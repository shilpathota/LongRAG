# -----------------------------------------------------------
# Hybrid chunking compatible with LlamaIndex v0.11.x:
# - Semantic children (~512 tokens) using SemanticSplitterNodeParser
# - Manual packing of children into ~2048-token parents (ordered)
# Outputs JSONL for parents/children (no LlamaIndex node objects needed later)
# -----------------------------------------------------------

import os
import glob
import json
import uuid
from typing import List, Dict, Any

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

RAW_DIR = os.path.join("data", "raw")
OUT_PARENTS_DIR = os.path.join("data", "processed", "parents")
OUT_CHILDREN_DIR = os.path.join("data", "processed", "children")

# Targets (approx token counts; we approximate by words*1.3 below)
PARENT_TARGET_TOKENS = 2048
CHILD_TARGET_TOKENS = 512

# Semantic splitter config
SEMANTIC_BREAKPOINT_PERCENTILE = 95
HF_EMBED_MODEL = "BAAI/bge-base-en"


def ensure_dirs() -> None:
    os.makedirs(OUT_PARENTS_DIR, exist_ok=True)
    os.makedirs(OUT_CHILDREN_DIR, exist_ok=True)


def load_text_documents(raw_dir: str) -> List[Document]:
    docs: List[Document] = []
    files: List[str] = []
    for pat in ("*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(raw_dir, pat)))
    if not files:
        raise FileNotFoundError(
            f"No input files found in {raw_dir}. Put e.g. policy.txt there."
        )
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append(Document(text=text, metadata={"source": os.path.basename(fp), "path": fp}))
    return docs


def approx_tokens(s: str) -> int:
    # quick heuristic: tokens ≈ words * 1.3
    return int(len(s.split()) * 1.3)


def pack_children_into_parents(children_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Greedy pack: iterate children in order, group into ~PARENT_TARGET_TOKENS parents.
    """
    parents: List[Dict[str, Any]] = []
    cur_children: List[str] = []
    cur_text_parts: List[str] = []
    cur_tokens = 0

    def flush_parent():
        nonlocal cur_children, cur_text_parts, cur_tokens
        if not cur_children:
            return
        pid = "P_" + uuid.uuid4().hex[:12]
        parents.append({
            "id": pid,
            "text": "\n\n".join(cur_text_parts),
            "children": list(cur_children),
            "source": children_source(cur_children, children_rows),
            "path": children_path(cur_children, children_rows),
            "char_count": sum(len(t) for t in cur_text_parts),
        })
        cur_children = []
        cur_text_parts = []
        cur_tokens = 0

    for ch in children_rows:
        t = ch["text"]
        t_tokens = approx_tokens(t)
        # if a single child is bigger than parent target, still force it in alone
        if cur_tokens and (cur_tokens + t_tokens > PARENT_TARGET_TOKENS):
            flush_parent()
        cur_children.append(ch["id"])
        cur_text_parts.append(t)
        cur_tokens += t_tokens

    flush_parent()
    return parents


def children_source(child_ids: List[str], children_rows: List[Dict[str, Any]]) -> str:
    # pick the first non-empty source among these child rows
    lookup = {c["id"]: c for c in children_rows}
    for cid in child_ids:
        src = lookup.get(cid, {}).get("source", "")
        if src:
            return src
    return ""


def children_path(child_ids: List[str], children_rows: List[Dict[str, Any]]) -> str:
    lookup = {c["id"]: c for c in children_rows}
    for cid in child_ids:
        pth = lookup.get(cid, {}).get("path", "")
        if pth:
            return pth
    return ""


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print(">> Ensuring output directories...")
    ensure_dirs()

    print(f">> Loading documents from: {RAW_DIR}")
    docs = load_text_documents(RAW_DIR)
    print(f">> Loaded {len(docs)} document(s).")

    print(">> Creating semantic children (~512 tokens)…")
    embed = HuggingFaceEmbedding(model_name=HF_EMBED_MODEL)
    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=5,
        breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_PERCENTILE,
        embed_model=embed,
        # LlamaIndex will aim for semantic boundaries; we still cap roughly by size below
    )

    # produce Nodes from all docs
    nodes = semantic_splitter.get_nodes_from_documents(docs)

    # Convert to our JSON rows and gently cap children near CHILD_TARGET_TOKENS
    children_rows: List[Dict[str, Any]] = []
    for node in nodes:
        text = getattr(node, "text", "")
        # If child is too big, truncate to keep things bounded
        if approx_tokens(text) > CHILD_TARGET_TOKENS:
            words = text.split()
            keep = max(80, int(CHILD_TARGET_TOKENS / 1.3))
            text = " ".join(words[:keep]) + " ..."
        cid = "C_" + uuid.uuid4().hex[:12]
        md = getattr(node, "metadata", {}) or {}
        children_rows.append({
            "id": cid,
            "text": text,
            "metadata": md,
            "source": md.get("source", ""),
            "path": md.get("path", ""),
            # parent_id will be set after packing
        })

    print(f">> Children produced: {len(children_rows)}")

    print(">> Packing children into ~2048-token parents…")
    parents_rows = pack_children_into_parents(children_rows)

    # Fill back parent_id on children
    pid_by_child: Dict[str, str] = {}
    for p in parents_rows:
        for cid in p["children"]:
            pid_by_child[cid] = p["id"]
    for ch in children_rows:
        ch["parent_id"] = pid_by_child.get(ch["id"], "")

    # Write outputs
    parents_out = os.path.join(OUT_PARENTS_DIR, "parents.jsonl")
    children_out = os.path.join(OUT_CHILDREN_DIR, "children.jsonl")
    write_jsonl(parents_out, parents_rows)
    write_jsonl(children_out, children_rows)

    print(f">> Wrote parents:  {parents_out}  ({len(parents_rows)} records)")
    print(f">> Wrote children: {children_out} ({len(children_rows)} records)")
    print(">> Done.")


if __name__ == "__main__":
    main()
