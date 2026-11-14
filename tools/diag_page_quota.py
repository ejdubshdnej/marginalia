"""诊断：页配额检索——每页最多取 N 块，是否同时改善 nDCG 与 Recall。

动机
----
当前块级 RRF 的 top-10 来自约 8 个不同页（页精度 0.23–0.27），
而三档相关性指标（gold=2 / 同页=1 / 其余=0）的 IDCG 要求 top-10 尽量落在 gold 页。
纯块级检索天然把不同页的块混排，这是 0.3740 的直接原因。

「每页最多 N 块」不是调参凑分，它是 RAG 里常见的多样性约束：
同页相邻块有 100 字符重叠，重复返回多块只是浪费上下文预算。
本脚本测 N ∈ {1,2,3,4} × 取几页，看两项指标能否同时不退化。

跑法：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 tools/diag_page_quota.py
"""
from __future__ import annotations

import json

from insight_rag.config import CHUNKS_PATH, ROOT
from insight_rag.eval.metrics import ndcg_at_k, recall_at_k, page_size_map


def load_qa():
    rows = [json.loads(l) for l in (ROOT / "eval/qa_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["gold_pages"]]


def quota_pick(hits, idx2page, per_page: int, top_k: int = 10):
    """按原始分数顺序遍历，每页最多取 per_page 块，直到凑够 top_k。"""
    used: dict[int, int] = {}
    out = []
    for h in hits:  # 假设 hits 已按分数降序
        p = idx2page[h["idx"]]
        if used.get(p, 0) >= per_page:
            continue
        used[p] = used.get(p, 0) + 1
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def main() -> int:
    from insight_rag.retrieve import hybrid

    hybrid.load()
    st = hybrid._STATE
    chunks = st["chunks"]
    idx2page = {i: c["page"] for i, c in enumerate(chunks)}
    idx2chunk = {i: c for i, c in enumerate(chunks)}
    ps = page_size_map(chunks)
    qa = load_qa()

    # 候选池要足够大，否则配额约束常常凑不满 10 块
    POOL = 60

    def run(per_page: int):
        ndcgs, recs, precs = [], [], []
        for r in qa:
            raw = hybrid.hybrid_search(r["question"], top_k=POOL)
            picked = quota_pick(raw, idx2page, per_page)
            hits = [dict(idx2chunk[h["idx"]], score=h["score"]) for h in picked]
            pages = [h["page"] for h in hits]
            ndcgs.append(ndcg_at_k(hits, r["gold_chunks"], r["gold_pages"], 10, page_sizes=ps))
            recs.append(recall_at_k(pages, r["gold_pages"], 5))
            gold = set(r["gold_pages"])
            precs.append(sum(1 for p in pages if p in gold) / max(len(pages), 1))
        n = len(qa)
        return sum(recs) / n, sum(ndcgs) / n, sum(precs) / n

    print(f"可评题 {len(qa)}；候选池 top-{POOL}")
    print(f"{'每页上限':>8}  {'Recall@5':>9}  {'nDCG@10':>9}  {'页精度':>8}")
    print("-" * 42)
    base_r, base_n, base_p = None, None, None
    for n_pp in (999, 4, 3, 2, 1):
        r, nd, pr = run(n_pp)
        label = "无限制" if n_pp == 999 else str(n_pp)
        if n_pp == 999:
            base_r, base_n, base_p = r, nd, pr
        mark = ""
        if n_pp != 999:
            if nd > base_n:
                mark += " nDCG↑"
            if r < base_r - 1e-6:
                mark += " Recall↓"
        print(f"{label:>8}  {r:9.4f}  {nd:9.4f}  {pr:8.4f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
