# -----------------------------------------------------------
# Build FAISS index over PARENT nodes only (LongRAG pattern)
# Inputs :
#   data/processed/parents/parents.jsonl
#   data/processed/children/children.jsonl (for metadata only)
# Outputs:
#   index/parents.faiss
#   index/meta.json
# -----------------------------------------------------------

import os
import json
import glob
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# -------- Config --------
PARENTS_PATH = os.path.join("data", "processed", "parents", "parents.jsonl")
CHILDREN_PATH = os.path.join("data", "processed", "children", "children.jsonl")  # optional, for richer meta
INDEX_DIR = os.path.join("index")
INDEX_FILE = os.path.join(INDEX_DIR, "parents.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

# Embedding model (swap to BAAI/bge-base-en if you prefer)
EMBED_MODEL = "intfloat/e5-base-v2"

# If using E5 models, prepend "query: " for queries later (retrieval stage),
# and "passage: " for documents here. We'll store raw text, but embed with passage prefix.
USE_E5_PREFIX = EMBED_MODEL.startswith("intfloat/e5-")
PASSAGE_PREFIX = "passage: " if USE_E5_PREFIX else ""
# ------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)


def main():
    ensure_dirs()
    if not os.path.exists(PARENTS_PATH):
        raise FileNotFoundError(f"Missing parents file: {PARENTS_PATH} (run ingest.py first)")

    print(f">> Loading parents from {PARENTS_PATH}")
    parents = read_jsonl(PARENTS_PATH)
    print(f">> Found {len(parents)} parents")

    children_map = {}
    if os.path.exists(CHILDREN_PATH):
        print(f">> (Optional) Loading children from {CHILDREN_PATH} for metadata enrichment")
        children_rows = read_jsonl(CHILDREN_PATH)
        for c in children_rows:
            pid = c.get("parent_id", "")
            if pid:
                children_map.setdefault(pid, []).append(c["id"])

    # Build texts & metadata arrays in consistent ID order
    parent_ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for p in parents:
        pid = p["id"]
        txt = p.get("text", "")
        src = p.get("source", p.get("metadata", {}).get("source", ""))
        path = p.get("path", p.get("metadata", {}).get("path", ""))

        parent_ids.append(pid)
        texts.append(PASSAGE_PREFIX + txt)  # prefix for E5-style models
        metas.append({
            "id": pid,
            "source": src,
            "path": path,
            # prefer children list from parents.jsonl; fall back to recomputed children_map
            "children": p.get("children") or children_map.get(pid, []),
            "char_count": len(txt),
        })

    print(">> Loading embedding model:", EMBED_MODEL)
    emb = SentenceTransformer(EMBED_MODEL)

    print(">> Encoding parent texts (this can take a moment)…")
    X = emb.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    X = np.asarray(X, dtype="float32")

    # FAISS index with Inner Product (since vectors are L2-normalized, IP == cosine sim)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    faiss.write_index(index, INDEX_FILE)
    print(f">> Wrote FAISS index: {INDEX_FILE}  (ntotal={index.ntotal})")

    # Save metadata with the exact order matching FAISS vector rows
    meta = {
        "parent_ids": parent_ids,
        "parents": metas,  # list aligned with parent_ids / FAISS rows
        "embedding_model": EMBED_MODEL,
        "normalize": True,
        "similarity": "cosine_via_inner_product",
        "notes": "Parents only. Children are expanded at query time.",
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f">> Wrote metadata:     {META_FILE}")
    print(">> Done.")


if __name__ == "__main__":
    main()
