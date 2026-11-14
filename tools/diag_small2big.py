"""诊断：页级排序质量，以及 small-to-big（页级召回 + 页内展开）的实测表现。

问题背景
--------
修正 IDCG 构造后，块级 nDCG@10 = 0.3740，而目标门槛是 0.55。
oracle（页全对、页内按真实块分排）= 0.9347，random order 也有 0.8749，
说明门槛不是不可达——可达的前提是「top-10 块里大部分落在 gold 页」。

本脚本回答两个问题：
  1. 页级排序本身有多准？（top-1/3/5 页命中率）
  2. 若先用页级分数召回页、再把该页的块摊平进 top-10（small-to-big），
     nDCG 与 Recall 各自变成多少？

这是设计评估，不是调参：small-to-big 是 RAG 里的常规模式（父文档检索），
代价是上下文更冗余，收益是块序不再被跨页混排打散。

跑法：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 tools/diag_small2big.py
"""
from __future__ import annotations

import json

from insight_rag.config import ROOT, CHUNKS_PATH
from insight_rag.eval.metrics import ndcg_at_k, recall_at_k, page_size_map


def load_qa():
    rows = [json.loads(l) for l in (ROOT / "eval/qa_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["gold_pages"]]


def chunk_hits_to_pages(hits, idx2page):
    """块级命中 -> 页级排名。页分 = 该页命中块中的最高分。"""
    best: dict[int, float] = {}
    for h in hits:
        p = idx2page[h["idx"]]
        best[p] = max(best.get(p, -1e9), h["score"])
    return sorted(best.items(), key=lambda kv: -kv[1])


def expand(pages_order, chunks_by_page, top_k=10, per_page=None):
    """按页顺序摊平页内块，凑满 top_k。"""
    out = []
    for p, ps in pages_order:
        blocks = sorted(chunks_by_page.get(p, []), key=lambda c: c["char_start"])
        if per_page:
            blocks = blocks[:per_page]
        for c in blocks:
            out.append(c)
            if len(out) >= top_k:
                return out
    return out


def main() -> int:
    from insight_rag.retrieve import hybrid

    hybrid.load()
    st = hybrid._STATE
    chunks = st["chunks"]
    idx2page = {i: c["page"] for i, c in enumerate(chunks)}
    ps = page_size_map(chunks)
    chunks_by_page: dict[int, list] = {}
    for c in chunks:
        chunks_by_page.setdefault(c["page"], []).append(c)
    qa = load_qa()

    # ---- 1. 页级排序质量 ----
    hit1 = hit3 = hit5 = 0
    for r in qa:
        order = [p for p, _ in chunk_hits_to_pages(
            hybrid.hybrid_search(r["question"], top_k=50), idx2page)]
        gold = set(r["gold_pages"])
        if order[:1] and order[0] in gold:
            hit1 += 1
        if any(p in gold for p in order[:3]):
            hit3 += 1
        if any(p in gold for p in order[:5]):
            hit5 += 1
    n = len(qa)
    print(f"页级排序（由块级 max-pooling 聚合，候选池 top-50 块）")
    print(f"  页 Recall@1 {hit1/n:.4f}   @3 {hit3/n:.4f}   @5 {hit5/n:.4f}")
    print()

    # ---- 2. small-to-big ----
    print(f"{'策略':<28} {'Recall@5':>9} {'nDCG@10':>9} {'页精度':>8}")
    print("-" * 58)

    def evaluate(label, build_hits):
        nd, rc, pr = [], [], []
        for r in qa:
            hits = build_hits(r)
            pages = [h["page"] for h in hits]
            nd.append(ndcg_at_k(hits, r["gold_chunks"], r["gold_pages"], 10, page_sizes=ps))
            rc.append(recall_at_k(pages, r["gold_pages"], 5))
            g = set(r["gold_pages"])
            pr.append(sum(1 for p in pages if p in g) / max(len(pages), 1))
        print(f"{label:<28} {sum(rc)/n:9.4f} {sum(nd)/n:9.4f} {sum(pr)/n:8.4f}")

    def base(r):
        return hybrid.hybrid_search(r["question"], top_k=10)

    def s2b(all_pages):
        def f(r):
            order = chunk_hits_to_pages(hybrid.hybrid_search(r["question"], top_k=50), idx2page)
            if not all_pages:
                order = order[:1]
            return expand(order, chunks_by_page, top_k=10)
        return f

    evaluate("现状：块级 RRF top-10", base)
    evaluate("small-to-big：页1，页内全展开", s2b(all_pages=False))
    evaluate("small-to-big：页序摊平", s2b(all_pages=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
