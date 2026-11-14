"""FAISS 向量库。IndexFlatIP + 已归一化向量 = 余弦相似度。"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

DIM = 768


def build(embeddings: np.ndarray) -> faiss.Index:
    emb = np.ascontiguousarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def save(index: faiss.Index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search(index: faiss.Index, qvec: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
    """返回 [(chunk_index, score), ...]，按分数降序。"""
    q = np.ascontiguousarray(qvec, dtype="float32")
    if q.ndim == 1:
        q = q[None, :]
    scores, idxs = index.search(q, top_k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
