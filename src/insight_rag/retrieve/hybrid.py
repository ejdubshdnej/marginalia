"""混合检索：向量路 + BM25 路，用 RRF 融合（不用加权求和）。

为什么不用 α·cos + β·bm25：余弦在 0–1 有界，BM25 无界（可能十几）。
归一化系数本身是需要验证集才能调的假设，而本项目没有验证集。
RRF 只用排名，绕开量纲问题。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from insight_rag import config
from insight_rag.index import bm25_store, embedder, vector_store

_STATE: dict = {}


def load(corpus_dir: str | Path | None = None) -> dict:
    """惰性加载 chunks / faiss / bm25，进程内复用。"""
    if _STATE:
        return _STATE
    d = Path(corpus_dir) if corpus_dir is not None else config.INDEX_DIR
    chunks = [json.loads(line) for line in
              (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    index = vector_store.load(d / "faiss.index")
    tokens = bm25_store.load_tokens(d / "tokens.json")
    _STATE.update({
        "chunks": chunks,
        "index": index,
        "bm25": bm25_store.build_from_tokens(tokens),
    })
    return _STATE


def rrf_fuse(vec_hits: list[tuple[int, float]], bm25_hits: list[tuple[int, float]],
             k: int = config.RRF_K, top_k: int = config.TOP_K_FUSED) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion。分数 = Σ 1/(k + rank + 1)。"""
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(vec_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def hybrid_search(query: str, top_k: int = config.TOP_K_FUSED,
                  k: int = config.RRF_K, top_k_vec: int = config.TOP_K_VECTOR,
                  top_k_bm25: int = config.TOP_K_BM25,
                  state: dict | None = None) -> list[dict]:
    """两路检索 + RRF 融合。返回带 source 标记的块列表。"""
    st = state if state is not None else load()
    chunks = st["chunks"]

    qvec = embedder.embed_query(query)
    vec_hits = vector_store.search(st["index"], qvec, top_k_vec)
    bm_hits = bm25_store.search(st["bm25"], query, top_k_bm25)

    vec_ids = {i for i, _ in vec_hits}
    bm_ids = {i for i, _ in bm_hits}
    fused = rrf_fuse(vec_hits, bm_hits, k=k, top_k=top_k)

    out: list[dict] = []
    for idx, score in fused:
        if idx in vec_ids and idx in bm_ids:
            src = "both"
        elif idx in vec_ids:
            src = "vec"
        else:
            src = "bm25"
        c = chunks[idx]
        out.append({
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "paper": c["paper"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "text": c["text"],
            "type": c.get("type", "text"),
            "score": score,
            "source": src,
            "idx": idx,
        })
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    q = "混合检索的融合策略为什么选 RRF"
    for r in hybrid_search(q, top_k=10):
        print(r["chunk_id"], r["source"], round(r["score"], 4),
              r["text"][:40].replace("\n", " "))
