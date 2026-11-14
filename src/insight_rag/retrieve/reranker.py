"""CrossEncoder 重排（bge-reranker-base）。"""
from __future__ import annotations

import time

from insight_rag import config


class Reranker:
    def __init__(self, path=None):
        from sentence_transformers import CrossEncoder
        self.path = str(path or config.RERANK_MODEL)
        self.model = CrossEncoder(self.path, device="cuda", max_length=512)
        self.last_ms = 0.0

    def rerank(self, query: str, chunks: list[dict], top_k: int = config.TOP_K_RERANK) -> list[dict]:
        """对候选块打 (query, text) 相关性分，返回 top_k，并写回 rerank_score。"""
        if not chunks:
            return []
        pairs = [(query, c["text"]) for c in chunks]
        t0 = time.time()
        scores = self.model.predict(pairs, batch_size=20)
        self.last_ms = (time.time() - t0) * 1000
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))[:top_k]
        out = []
        for i in order:
            c = dict(chunks[i])
            c["rerank_score"] = float(scores[i])
            out.append(c)
        return out

    def warmup(self):
        self.model.predict([("预热", "预热文本")], batch_size=1)
