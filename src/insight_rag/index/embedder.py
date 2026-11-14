"""BGE 向量化。查询侧加指令前缀，文档侧不加；一律 L2 归一化。"""
from __future__ import annotations

import numpy as np

from insight_rag import config

DEVICE = "cuda"
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(str(config.EMBED_MODEL), device=DEVICE)
        m.max_seq_length = 512
        _MODEL = m
    return _MODEL


def embed_docs(texts: list[str], batch: int = 32) -> np.ndarray:
    """(n, 768) float32，已 L2 归一化。文档侧不加前缀。"""
    if not texts:
        return np.zeros((0, 768), dtype="float32")
    m = _model()
    emb = m.encode(list(texts), batch_size=batch, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    return np.ascontiguousarray(emb, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """(1, 768)，加 BGE 中文检索指令前缀。"""
    m = _model()
    emb = m.encode([config.QUERY_PREFIX + text], convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    return np.ascontiguousarray(emb, dtype="float32")
