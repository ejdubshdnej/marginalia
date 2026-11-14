"""为 docs/DESIGN_TRADEOFFS.md 补实测数据（只读，不落盘）。

四处要写进文档的取舍，都必须有实测数字，不能写「我估计」：

1. 块大小 500 vs 300 vs 800 —— 三套切块各自重建索引，比 Recall@5 / nDCG@10。
2. RRF vs 加权求和 —— 加权和要先归一化再定 α；这里扫 α=0.3/0.5/0.7/0.9，
   看它能不能稳定超过 RRF，以及「最优 α」有多依赖这 15 道题。
3. FAISS Flat 的纯检索耗时 —— 与端到端 hybrid_search 分开量。
4. 分栏收益已在 tools/diag_g15.py 量过（nDCG +1.5pp），这里不重复。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 tools/diag_tradeoffs.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from insight_rag.config import CORPUS_PDF, INDEX_DIR, ROOT
from insight_rag.eval.metrics import ndcg_at_k, recall_at_k, page_size_map
from insight_rag.index import bm25_store, embedder, vector_store
from insight_rag.parse.chunker import chunk_pages
from insight_rag.parse.pdf_parser import parse_pdf
from insight_rag.retrieve.hybrid import rrf_fuse


def load_qa() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "eval" / "qa_set.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["gold_pages"]]


def score(qa: list[dict], search, page_sizes: dict[int, int] | None = None) -> tuple[float, float]:
    """page_sizes 必须与本次切块一致——不同切块方案每页块数不同，IDCG 也不同。"""
    recs, ndcgs = [], []
    for r in qa:
        hits = search(r["question"], 10)
        pages = [h["page"] for h in hits]
        recs.append(recall_at_k(pages, r["gold_pages"], 5))
        ndcgs.append(ndcg_at_k(hits, r["gold_chunks"], r["gold_pages"], 10,
                               page_sizes=page_sizes))
    n = len(qa)
    return sum(recs) / n, sum(ndcgs) / n


def weighted_fuse(vec_hits, bm_hits, chunk_count: int, alpha: float, top_k: int = 20):
    """归一化加权求和。余弦已在 [0,1] 附近，这里按 min-max 拉到 [0,1]；
    BM25 无界，必须按本次召回集 min-max 归一化。"""
    def norm(hits):
        if not hits:
            return {}
        vals = [s for _, s in hits]
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng <= 1e-12:
            return {i: 1.0 for i, _ in hits}
        return {i: (s - lo) / rng for i, s in hits}

    nv, nb = norm(vec_hits), norm(bm_hits)
    for i in nv:
        nv[i] = nv[i] * alpha
    for i in nb:
        nb[i] = nb[i] * (1 - alpha)
    merged = dict(nv)
    for i, s in nb.items():
        merged[i] = merged.get(i, 0.0) + s
    return sorted(merged.items(), key=lambda x: -x[1])[:top_k]


def main() -> int:
    qa = load_qa()
    print(f"qa {len(qa)} 条（有 gold）")

    # ---------- 1. 块大小对照 ----------
    print("\n" + "=" * 74)
    print("取舍一：块大小 300/60 · 500/100 · 800/160（同样分栏、同样 RRF）")
    print("=" * 74)
    pages = parse_pdf(CORPUS_PDF, use_column_split=True)
    size_results = {}
    for size, overlap in [(300, 60), (500, 100), (800, 160)]:
        cs = chunk_pages(pages, size=size, overlap=overlap, min_len=50)
        t0 = time.perf_counter()
        em = embedder.embed_docs([c["text"] for c in cs])
        build_s = time.perf_counter() - t0
        idx = vector_store.build(em)
        bm = bm25_store.build([c["text"] for c in cs])

        def search(q: str, k: int, _cs=cs, _idx=idx, _bm=bm):
            v = vector_store.search(_idx, embedder.embed_query(q), 20)
            b = bm25_store.search(_bm, q, 20)
            fused = rrf_fuse(v, b, top_k=max(k, 20))
            return [dict(_cs[i]) for i, _ in fused]

        r5, nd = score(qa, search, page_size_map(cs))
        size_results[f"{size}/{overlap}"] = (len(cs), r5, nd)
        print(f"  {size:3d}/{overlap:<3d} chunks {len(cs):5d}  Recall@5 {r5:.4f}  "
              f"nDCG@10 {nd:.4f}  建索引 {build_s:5.1f}s")

    # ---------- 2. RRF vs 加权求和 ----------
    print("\n" + "=" * 74)
    print("取舍二：RRF（k=60，无需调参）vs 归一化加权求和（α 要调）")
    print("=" * 74)
    chunks = [json.loads(l) for l in (INDEX_DIR / "chunks.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = vector_store.load(INDEX_DIR / "faiss.index")
    bm = bm25_store.build_from_tokens(bm25_store.load_tokens(INDEX_DIR / "tokens.json"))
    ps_main = page_size_map(chunks)

    # 先缓存每题的候选，避免重复编码
    cache = {}
    for r in qa:
        v = vector_store.search(idx, embedder.embed_query(r["question"]), 20)
        b = bm25_store.search(bm, r["question"], 20)
        cache[r["qid"]] = (v, b)

    def eval_rows(pick):
        recs, ndcgs, alpha_used = [], [], []
        for r in qa:
            v, b = cache[r["qid"]]
            fused, a = pick(v, b)
            hits = [dict(chunks[i]) for i, _ in fused[:10]]
            recs.append(recall_at_k([h["page"] for h in hits], r["gold_pages"], 5))
            ndcgs.append(ndcg_at_k(hits, r["gold_chunks"], r["gold_pages"], 10,
                                   page_sizes=ps_main))
            alpha_used.append(a)
        return sum(recs) / len(recs), sum(ndcgs) / len(ndcgs), alpha_used

    r5, nd, _ = eval_rows(lambda v, b: (rrf_fuse(v, b, top_k=20), 0.0))
    print(f"  RRF      Recall@5 {r5:.4f}  nDCG@10 {nd:.4f}   （无参数）")

    best = None
    for alpha in (0.3, 0.5, 0.7, 0.9):
        r5a, nda, _ = eval_rows(
            lambda v, b, a=alpha: (weighted_fuse(v, b, len(chunks), a), a))
        print(f"  α={alpha:<4}  Recall@5 {r5a:.4f}  nDCG@10 {nda:.4f}")
        if best is None or r5a > best[1]:
            best = (alpha, r5a, nda)
    print(f"  → 加权和最好的一组 α={best[0]}，Recall@5 {best[1]:.4f}；"
          f"RRF {r5:.4f}，差 {best[1] - r5:+.4f}")

    # ---------- 3. FAISS Flat 纯检索耗时 ----------
    print("\n" + "=" * 74)
    print("取舍三：FAISS IndexFlatIP 纯检索耗时（不含编码、不含 BM25）")
    print("=" * 74)
    qvecs = [embedder.embed_query(r["question"]) for r in qa]
    ts = []
    for qv in qvecs:
        t0 = time.perf_counter()
        vector_store.search(idx, qv, 20)
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"  n={len(ts)}  mean {statistics.mean(ts):.2f} ms  "
          f"p95 {sorted(ts)[int(0.95 * len(ts)) - 1]:.2f} ms  "
          f"max {max(ts):.2f} ms  （索引 {idx.ntotal} 条 × 768 维）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
