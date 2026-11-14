"""端到端问答：混合检索 → 重排 → 生成 → 引用校验。"""
from __future__ import annotations

import time

from insight_rag import config
from insight_rag.generate import prompt as P
from insight_rag.generate.llm import LLM
from insight_rag.retrieve import hybrid
from insight_rag.retrieve.reranker import Reranker

_STATE: dict = {}


def init(load_llm: bool = True, warmup: bool = True) -> dict:
    """一次性加载检索索引、重排模型、生成模型。"""
    if _STATE.get("ready"):
        return _STATE
    st = hybrid.load()
    rr = Reranker()
    rr.warmup()
    _STATE.update({"chunks": st["chunks"], "hybrid_state": st, "reranker": rr})
    if load_llm:
        llm = LLM(config.LLM_GGUF)
        if warmup:
            llm.warmup()
        _STATE["llm"] = llm
    _STATE["ready"] = True
    return _STATE


def ask(question: str, top_k: int = config.TOP_K_RERANK,
        max_tokens: int = config.MAX_TOKENS, temperature: float = config.TEMPERATURE,
        state: dict | None = None) -> dict:
    """返回 {answer, citations, latency:{retrieve_ms, generate_ms, total_ms}, citation_check}。"""
    st = state if state is not None else init()

    t0 = time.time()
    cand = hybrid.hybrid_search(question, top_k=config.TOP_K_FUSED,
                                state=st.get("hybrid_state"))
    top = st["reranker"].rerank(question, cand, top_k=top_k)
    t1 = time.time()

    user = P.build_prompt(question, top)
    resp = st["llm"].chat(
        [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temperature,
    )
    t2 = time.time()

    answer = resp["choices"][0]["message"]["content"]
    check = P.validate_citations(answer, len(top))

    citations = []
    for i, c in enumerate(top):
        citations.append({
            "n": i + 1,
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "paper": c["paper"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "text": c["text"],
            "rerank_score": c.get("rerank_score"),
            "source": c.get("source"),
        })

    return {
        "answer": answer,
        "citations": citations,
        "citation_check": check,
        "usage": resp.get("usage", {}),
        "latency": {
            "retrieve_ms": round((t1 - t0) * 1000, 1),
            "generate_ms": round((t2 - t1) * 1000, 1),
            "total_ms": round((t2 - t0) * 1000, 1),
        },
    }
