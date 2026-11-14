"""四组消融：A 纯向量 / B 纯 BM25 / C RRF 融合 / D C+reranker。

判据：
  - C 的 Recall@5 ≥ max(A, B)  → 融合确实比单路好
  - D 的 nDCG@10 ≥ C 的 nDCG@10 → reranker 确实改善了排序质量
不成立就如实记录，不许调参凑数。

命令：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.eval.ablation \
      --qa eval/qa_set.jsonl --out runs/ablation.json --png runs/ablation.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from insight_rag.config import ROOT, CHUNKS_PATH
from insight_rag.eval.metrics import recall_at_k, ndcg_at_k, page_size_map

K_RET = 10  # 检索与 nDCG 统一看前 10


def load_qa(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_group(name: str, search_fn, rows: list[dict], top_k: int = K_RET,
              page_sizes: dict[int, int] | None = None) -> dict:
    """跑一组配置，返回 Recall@5 / nDCG@10 / MRR 与逐题明细。"""
    per_q = []
    t0 = time.perf_counter()
    for r in rows:
        hits = search_fn(r["question"], top_k)
        if not r["gold_pages"]:
            continue
        pages = [h["page"] for h in hits]
        # MRR：第一个 gold 页所在名次
        mrr = 0.0
        for rank, p in enumerate(pages, 1):
            if p in set(r["gold_pages"]):
                mrr = 1.0 / rank
                break
        per_q.append({
            "qid": r["qid"],
            "recall_at_5": recall_at_k(pages, r["gold_pages"], 5),
            "ndcg_at_10": ndcg_at_k(hits, r["gold_chunks"], r["gold_pages"], 10,
                                    page_sizes=page_sizes),
            "mrr": round(mrr, 4),
            "top_pages": pages,
        })
    dt = (time.perf_counter() - t0) / max(len(per_q), 1) * 1000
    n = len(per_q)
    return {
        "name": name,
        "n_questions": n,
        "recall_at_5": round(sum(x["recall_at_5"] for x in per_q) / n, 4),
        "ndcg_at_10": round(sum(x["ndcg_at_10"] for x in per_q) / n, 4),
        "mrr": round(sum(x["mrr"] for x in per_q) / n, 4),
        "latency_ms": round(dt, 1),
        "per_question": per_q,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="eval/qa_set.jsonl")
    ap.add_argument("--out", default="runs/ablation.json")
    ap.add_argument("--png", default="runs/ablation.png")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.is_absolute():
        qa_path = ROOT / qa_path
    rows = [r for r in load_qa(qa_path)]
    graded = [r for r in rows if r["gold_pages"]]
    print(f"qa_set {len(rows)} 条，其中可评检索的 {len(graded)} 条")

    from insight_rag.retrieve import hybrid
    from insight_rag.retrieve.reranker import Reranker

    hybrid.load()
    st = hybrid._STATE
    faiss_idx = st["index"]
    bm25 = st["bm25"]
    chunks = st["chunks"]
    embed_query = hybrid.embedder.embed_query
    bm25_search = hybrid.bm25_store.search
    vec_search = hybrid.vector_store.search

    # ---- A 纯向量 ----
    def search_vec(q: str, top_k: int):
        qv = embed_query(q)
        hits = vec_search(faiss_idx, qv, top_k)
        return [dict(chunks[i], score=s) for i, s in hits]

    # ---- B 纯 BM25 ----
    def search_bm25(q: str, top_k: int):
        hits = bm25_search(bm25, q, top_k)
        return [dict(chunks[i], score=s) for i, s in hits]

    # ---- C RRF ----
    def search_rrf(q: str, top_k: int):
        return hybrid.hybrid_search(q, top_k=top_k)

    print("\n加载 reranker ...")
    rr = Reranker()
    rr.warmup()

    # ---- D C + reranker ----
    def search_rrf_rr(q: str, top_k: int):
        cand = hybrid.hybrid_search(q, top_k=max(top_k, 20))
        return rr.rerank(q, cand, top_k=top_k)

    groups = []
    page_sizes = page_size_map(chunks)
    for name, fn in [("A 纯向量", search_vec), ("B 纯BM25", search_bm25),
                     ("C RRF融合", search_rrf), ("D C+reranker", search_rrf_rr)]:
        g = run_group(name, fn, graded, page_sizes=page_sizes)
        groups.append(g)
        print(f"  {name:14s} Recall@5 {g['recall_at_5']:.4f}  nDCG@10 {g['ndcg_at_10']:.4f}  "
              f"MRR {g['mrr']:.4f}  {g['latency_ms']:6.1f}ms/题")

    A, B, C, D = groups
    checks = {
        "C_recall_ge_max_AB": {
            "value_C": C["recall_at_5"],
            "max_AB": max(A["recall_at_5"], B["recall_at_5"]),
            "pass": C["recall_at_5"] >= max(A["recall_at_5"], B["recall_at_5"]) - 1e-9,
        },
        "D_ndcg_ge_C": {
            "value_D": D["ndcg_at_10"],
            "value_C": C["ndcg_at_10"],
            "pass": D["ndcg_at_10"] >= C["ndcg_at_10"] - 1e-9,
        },
    }
    all_pass = all(c["pass"] for c in checks.values())

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qa_set": str(qa_path.relative_to(ROOT)),
        "k_retrieve": K_RET,
        "groups": groups,
        "checks": checks,
        "all_pass": all_pass,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图 ----
    png = make_png(groups, Path(args.png) if not Path(args.png).is_absolute()
                   else Path(args.png))

    print("\n" + "=" * 66)
    print(f"{'组':<16}{'Recall@5':>11}{'nDCG@10':>11}{'MRR':>9}{'ms/题':>10}")
    print("-" * 66)
    for g in groups:
        print(f"{g['name']:<16}{g['recall_at_5']:>11.4f}{g['ndcg_at_10']:>11.4f}"
              f"{g['mrr']:>9.4f}{g['latency_ms']:>10.1f}")
    print("-" * 66)
    c1, c2 = checks["C_recall_ge_max_AB"], checks["D_ndcg_ge_C"]
    print(f"C ≥ max(A,B) on Recall@5: {c1['value_C']:.4f} vs {c1['max_AB']:.4f}  "
          f"{'✅' if c1['pass'] else '❌'}")
    print(f"D ≥ C on nDCG@10:         {c2['value_D']:.4f} vs {c2['value_C']:.4f}  "
          f"{'✅' if c2['pass'] else '❌'}")
    print("=" * 66)
    print(f"{'✅ 判据通过' if all_pass else '❌ 判据未通过（如实记录，不调参）'}")
    print(f"产物 {out}")
    if png:
        print(f"图  {png}")
    return 0 if all_pass else 1


def make_png(groups: list[dict], path: Path):
    """两张并排柱状图：Recall@5 与 nDCG@10。300 dpi。

    图内文字一律用 ASCII——matplotlib 默认字体 DejaVu Sans 没有 CJK 字形，
    用中文会渲染成方块并刷一屏 UserWarning。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"（跳过画图：{e}）")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    ascii_names = {"A 纯向量": "A vector", "B 纯BM25": "B BM25",
                   "C RRF融合": "C RRF", "D C+reranker": "D RRF+rerank"}
    names = [ascii_names.get(g["name"], g["name"]) for g in groups]
    colors = ["#8fb8de", "#e8a87c", "#7fb069", "#c05c7e"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [(axes[0], "recall_at_5", "Recall@5 (page-level)"),
                           (axes[1], "ndcg_at_10", "nDCG@10 (chunk-level)")]:
        vals = [g[key] for g in groups]
        bars = ax.bar(names, vals, color=colors[:len(names)], width=0.62)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.28 if max(vals) > 0 else 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("InsightRAG ablation - 15 answerable questions, k=10 retrieval", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    sys.exit(main())
