# -----------------------------------------------------------
# LongRAG query server (FastAPI)
# - Retrieve parents via FAISS
# - Rerank parents via local cross-encoder (bge-reranker)
# - Expand with top semantic children (on-the-fly embed)
# - Assemble token-budgeted context
# - Generate answer via Ollama (llama3.1:8b)
# -----------------------------------------------------------

import os
import json
import time
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# ---------- Paths ----------
INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "parents.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

PARENTS_PATH = os.path.join("data", "processed", "parents", "parents.jsonl")
CHILDREN_PATH = os.path.join("data", "processed", "children", "children.jsonl")

# ---------- Models / Reranker ----------
EMBED_MODEL = "intfloat/e5-base-v2"   # must match build_index.py
USE_E5_PREFIX = EMBED_MODEL.startswith("intfloat/e5-")
E5_QUERY_PREFIX = "query: " if USE_E5_PREFIX else ""
E5_PASSAGE_PREFIX = "passage: " if USE_E5_PREFIX else ""

RERANKER_NAME = "BAAI/bge-reranker-base"  # local cross-encoder

# ---------- Retrieval knobs ----------
TOPK_ANN = 6               # FAISS ANN candidates
PARENTS_AFTER_RERANK = 4   # keep top N parents after reranker
CHILDREN_PER_PARENT = 3    # expand N best children per parent

# ---------- Token budget (rough) ----------
# We'll approximate "tokens" using words * 1.3 (quick heuristic).
MAX_SOURCE_TOKENS = 4000          # total budget for sources
TARGET_PARENT_TOKENS = 1200        # approx limit per parent block
TARGET_CHILD_TOKENS = 400          # approx limit per child block

# ---------- Ollama ----------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct"
OLLAMA_PARAMS = {
    # stream False: get full text in one response for simplicity
    "stream": False,
     "options": {
        "num_ctx": 8192,       # keep <= 8k on CPU unless you KNOW the model has more here
        "num_predict": 256,    # cap output length to avoid long runs
        "temperature": 0.2,
    }
}


# --------------- Utilities ---------------

def approx_token_count(text: str) -> int:
    # crude but fast: 1 token ≈ 0.75 words (=> words * 1.3 ≈ tokens)
    words = len(text.split())
    return int(words * 1.3)


def trim_to_tokens(text: str, max_tokens: int) -> str:
    # trim by words to respect rough token budget
    if approx_token_count(text) <= max_tokens:
        return text
    words = text.split()
    # try to keep approx fraction
    keep = max(50, int(max_tokens / 1.3))
    return " ".join(words[:keep]) + " ..."


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------- Load artifacts on startup ---------------

print(">> Loading FAISS & metadata...")
if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(f"Missing {INDEX_FILE}. Run app/build_index.py first.")
if not os.path.exists(META_FILE):
    raise FileNotFoundError(f"Missing {META_FILE}. Run app/build_index.py first.")
INDEX = faiss.read_index(INDEX_FILE)
META = json.load(open(META_FILE, "r", encoding="utf-8"))

print(">> Loading parents & children JSONL...")
PARENTS = {row["id"]: row for row in read_jsonl(PARENTS_PATH)}
CHILDREN_LIST = read_jsonl(CHILDREN_PATH) if os.path.exists(CHILDREN_PATH) else []
CHILDREN_BY_PARENT: Dict[str, List[Dict[str, Any]]] = {}
for c in CHILDREN_LIST:
    pid = c.get("parent_id", "")
    if pid:
        CHILDREN_BY_PARENT.setdefault(pid, []).append(c)

print(f">> Parents: {len(PARENTS)} | Children: {len(CHILDREN_LIST)}")

print(">> Loading embedding model:", EMBED_MODEL)
EMB = SentenceTransformer(EMBED_MODEL)

print(">> Loading reranker:", RERANKER_NAME)
RERANK_TOK = AutoTokenizer.from_pretrained(RERANKER_NAME)
RERANK_MODEL = AutoModelForSequenceClassification.from_pretrained(RERANKER_NAME)
RERANK_MODEL.eval()

# Build an ID order list from META so FAISS row -> parent_id works 1:1
PARENT_IDS = META["parent_ids"]
PARENTS_META = META["parents"]  # aligned with PARENT_IDS / FAISS rows


def embed_query(q: str) -> np.ndarray:
    txt = E5_QUERY_PREFIX + q
    v = EMB.encode([txt], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")


def embed_passages(texts: List[str]) -> np.ndarray:
    # used for child selection (on-the-fly)
    # NOTE: we prefix "passage: " if E5; that's ok for child scoring.
    if USE_E5_PREFIX:
        texts = [E5_PASSAGE_PREFIX + t for t in texts]
    X = EMB.encode(texts, normalize_embeddings=True)
    return np.asarray(X, dtype="float32")


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    pairs_a = [query] * len(candidates)
    pairs_b = [c["text"] for c in candidates]
    inputs = RERANK_TOK(pairs_a, pairs_b, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = RERANK_MODEL(**inputs).logits
    if logits.ndim == 0:
        logits = logits.view(1)
    elif logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    scores = torch.sigmoid(logits).detach().cpu().tolist()
    keep = min(top_n, len(candidates))
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:keep]
    reranked = [candidates[i] for i in order]
    for out, idx in zip(reranked, order):
        out["rerank_score"] = float(scores[idx])
    return reranked



def search_parents(query: str) -> List[Dict[str, Any]]:
    """First-pass FAISS ANN over parents, then rerank with cross-encoder."""
    qv = embed_query(query)
    D, I = INDEX.search(qv, TOPK_ANN)  # I: indices into PARENT_IDS
    cands = []
    for row_idx in I[0]:
        if row_idx < 0:
            continue
        pid = PARENT_IDS[row_idx]
        pdata = PARENTS.get(pid)
        if not pdata:
            continue
        # Trim parent text to a target length (for stable reranking)
        trimmed = trim_to_tokens(pdata.get("text", ""), TARGET_PARENT_TOKENS)
        cands.append({
            "id": pid,
            "text": trimmed,
            "source": pdata.get("source", ""),
            "path": pdata.get("path", ""),
            "children": pdata.get("children", []),
            "faiss_idx": int(row_idx)
        })
    if not cands:
        return []
    return rerank(query, cands, top_n=min(PARENTS_AFTER_RERANK, len(cands)))


def pick_children_for_parent(query_vec: np.ndarray, parent: Dict[str, Any], top_n: int) -> List[Dict[str, Any]]:
    """Select top-N children for a given parent using cosine sim (E5 embeddings)."""
    pid = parent["id"]
    kids = CHILDREN_BY_PARENT.get(pid, [])
    if not kids:
        return []

    texts = [k.get("text", "") for k in kids]
    if not any(texts):
        return []

    X = embed_passages(texts)  # (m, d) normalized
    # query_vec: (1, d) normalized
    sims = (X @ query_vec.T).reshape(-1)  # cosine via dot (both normalized)
    order = sims.argsort()[::-1][:top_n]
    chosen = []
    for idx in order:
        k = kids[idx]
        chosen.append({
            "id": k["id"],
            "parent_id": pid,
            "text": trim_to_tokens(k.get("text", ""), TARGET_CHILD_TOKENS),
            "source": k.get("source", k.get("metadata", {}).get("source", "")),
            "path": k.get("path", k.get("metadata", {}).get("path", "")),
            "sim": float(sims[idx])
        })
    return chosen


def assemble_sources(query: str) -> Dict[str, Any]:
    """Return a dict with 'context_text' and 'citations' based on budgets."""
    parents = search_parents(query)
    if not parents:
        return {"context_text": "", "citations": []}

    qv = embed_query(query)  # for child scoring
    budget_used = 0
    blocks: List[str] = []
    cites: List[Dict[str, str]] = []

    for p in parents:
        p_text = trim_to_tokens(p["text"], TARGET_PARENT_TOKENS)
        p_tok = approx_token_count(p_text)

        if budget_used + p_tok > MAX_SOURCE_TOKENS:
            # stop if adding this parent would exceed budget
            break

        blocks.append(f"[PARENT:{p['id']}] (src={p.get('source','')})\n{p_text}\n")
        cites.append({"type": "parent", "id": p["id"], "source": p.get("source", ""), "path": p.get("path", "")})
        budget_used += p_tok

        # expand children under budget
        chosen_children = pick_children_for_parent(qv, p, top_n=CHILDREN_PER_PARENT)
        for c in chosen_children:
            c_tok = approx_token_count(c["text"])
            if budget_used + c_tok > MAX_SOURCE_TOKENS:
                break
            blocks.append(f"[CHILD:{c['id']}] (from PARENT:{c['parent_id']})\n{c['text']}\n")
            cites.append({"type": "child", "id": c["id"], "parent_id": c["parent_id"],
                          "source": c.get("source", ""), "path": c.get("path", "")})
            budget_used += c_tok

    context_text = "\n".join(blocks).strip()
    return {"context_text": context_text, "citations": cites}


def call_ollama(question: str, sources: str) -> str:
    system = (
        "You are a grounded assistant. Use ONLY the provided sources.\n"
        "Cite spans as [PARENT:<id>] or [CHILD:<id>]. If not in sources, say you don't know."
    )
    prompt = f"{system}\n\n### Question\n{question}\n\n### Sources\n{sources}\n\n### Answer"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, **OLLAMA_PARAMS}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=900)  # was 300
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ReadTimeout:
        return "[SYSTEM] Timeout contacting Ollama after 900s. Try reducing context or using a smaller model."



# --------------- FastAPI ---------------

app = FastAPI(title="LongRAG (Ollama)")

class QueryIn(BaseModel):
    question: str
    topk_ann: Optional[int] = None
    parents_after_rerank: Optional[int] = None
    children_per_parent: Optional[int] = None
    max_source_tokens: Optional[int] = None


@app.post("/query")
def query_llm(body: QueryIn):
    # allow on-the-fly tuning
    global TOPK_ANN, PARENTS_AFTER_RERANK, CHILDREN_PER_PARENT, MAX_SOURCE_TOKENS
    if body.topk_ann is not None:
        TOPK_ANN = int(body.topk_ann)
    if body.parents_after_rerank is not None:
        PARENTS_AFTER_RERANK = int(body.parents_after_rerank)
    if body.children_per_parent is not None:
        CHILDREN_PER_PARENT = int(body.children_per_parent)
    if body.max_source_tokens is not None:
        MAX_SOURCE_TOKENS = int(body.max_source_tokens)

    t0 = time.time()
    assembled = assemble_sources(body.question)
    context_text = assembled["context_text"]
    citations = assembled["citations"]

    if not context_text:
        return {
            "answer": "",
            "citations": [],
            "message": "No context available (index empty or retrieval failed)",
            "latency_sec": round(time.time() - t0, 3)
        }

    answer = call_ollama(body.question, context_text)
    return {
        "answer": answer,
        "citations": citations,
        "latency_sec": round(time.time() - t0, 3),
        "budgets": {
            "max_source_tokens": MAX_SOURCE_TOKENS,
            "target_parent_tokens": TARGET_PARENT_TOKENS,
            "target_child_tokens": TARGET_CHILD_TOKENS
        },
        "params": {
            "topk_ann": TOPK_ANN,
            "parents_after_rerank": PARENTS_AFTER_RERANK,
            "children_per_parent": CHILDREN_PER_PARENT,
            "ollama_model": OLLAMA_MODEL
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "parents": len(PARENTS), "children": len(CHILDREN_LIST)}
