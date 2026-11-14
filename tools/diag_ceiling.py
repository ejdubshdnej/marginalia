"""判断 0.55 这个 nDCG 门槛在修正后的指标下是否可达。

三个参照点：
  1. 页全对、块随机：假设检索器总能找到正确的页，但不知道该页的哪个块是答案，
     前 10 位全来自 gold 页、gold 块位置随机。这是「页级完美检索器」的期望分。
  2. 页全对、块也对：gold 块排第一，其余同页块补满。= 1.0。
  3. 只命中一个同页块：说明指标有没有区分度。

再顺带量块级 Recall@5，判断检索器到底差在哪一层。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 tools/diag_ceiling.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_rag.config import INDEX_DIR, ROOT
from insight_rag.eval.metrics import ndcg_at_k, recall_at_k
from insight_rag.index import bm25_store, embedder, vector_store
from insight_rag.retrieve import reranker
from insight_rag.retrieve.hybrid import rrf_fuse

K = 10


def dcg(g):
    return sum(x / math.log2(i + 2) for i, x in enumerate(g))


def main() -> int:
    qa = [r for r in (json.loads(l) for l in (ROOT / "eval" / "qa_set.jsonl")
                      .read_text(encoding="utf-8").splitlines() if l.strip())
          if r["gold_pages"]]
    chunks = [json.loads(l) for l in (INDEX_DIR / "chunks.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    by_page = defaultdict(list)
    for c in chunks:
        by_page[c["page"]].append(c)
    page_sizes = {p: len(v) for p, v in by_page.items()}

    print(f"qa {len(qa)} 条，k={K}\n")

    # ---- 1. 页级完美检索器的期望 nDCG ----
    rng = random.Random(20261113)
    exp_scores = []
    for r in qa:
        gc, gp = set(r["gold_chunks"]), set(r["gold_pages"])
        pool = [c for p in gp for c in by_page.get(p, [])]
        if not pool:
            continue
        # 页内所有块，gold 块的相对位置随机 —— 模拟「找对页但排不准块」
        trials = []
        n_gold_in_pool = sum(1 for c in pool if c["chunk_id"] in gc)
        for _ in range(200):
            order = pool[:]
            rng.shuffle(order)
            hits = order[:K]
            trials.append(ndcg_at_k(hits, list(gc), list(gp), K, page_sizes=page_sizes))
        exp_scores.append(sum(trials) / len(trials))
    print("=== 1. 页级完美检索器（页全对、块序随机）的期望 nDCG@10 ===")
    print(f"  均值 {sum(exp_scores)/len(exp_scores):.4f}")
    print(f"  → 即使每道题都精准命中正确页，只要块序排不准，nDCG 也只有这个数")
    print(f"  → 门槛 0.55 {'可达' if sum(exp_scores)/len(exp_scores) >= 0.55 else '高于该上限，单靠页级定位达不到'}")

    # ---- 2. 只给 1 个同页块 ----
    print("\n=== 2. 区分度：只命中 1 个同页非 gold 块 ===")
    one = ndcg_at_k([{"chunk_id": "x", "page": qa[0]["gold_pages"][0]}],
                    qa[0]["gold_chunks"], qa[0]["gold_pages"], K,
                    page_sizes=page_sizes)
    print(f"  nDCG = {one:.4f}（旧实现恒为 1.0）")

    # ---- 3. 实际四组：nDCG 与块级 Recall ----
    idx = vector_store.load(INDEX_DIR / "faiss.index")
    bm = bm25_store.build_from_tokens(bm25_store.load_tokens(INDEX_DIR / "tokens.json"))
    rr = reranker.Reranker()

    def to_hits(p):
        return [chunks[i] for i, _ in p]

    def fn_vec(q, k=K):
        return to_hits(vector_store.search(idx, embedder.embed_query(q), k))

    def fn_bm(q, k=K):
        return to_hits(bm25_store.search(bm, q, k))

    def fn_rrf(q, k=K):
        return to_hits(rrf_fuse(vector_store.search(idx, embedder.embed_query(q), 20),
                                bm25_store.search(bm, q, 20), top_k=max(k, 20)))

    def fn_rrfrr(q, k=K):
        f = rrf_fuse(vector_store.search(idx, embedder.embed_query(q), 20),
                     bm25_store.search(bm, q, 20), top_k=20)
        return rr.rerank(q, [dict(chunks[i]) for i, _ in f], top_k=k)

    print(f"\n=== 3. 四组：nDCG@10 / 页级 Recall@5 / 块级 Recall@5 / 页精度 ===")
    print(f"{'组':12s} {'nDCG@10':>9s} {'页R@5':>8s} {'块R@5':>8s} {'页精度@10':>10s}")
    print("-" * 52)
    for name, fn in (("A 纯向量", fn_vec), ("B 纯BM25", fn_bm),
                     ("C RRF融合", fn_rrf), ("D C+reranker", fn_rrfrr)):
        nd, pr, cr, pp = [], [], [], []
        for r in qa:
            h = fn(r["question"])
            nd.append(ndcg_at_k(h, r["gold_chunks"], r["gold_pages"], K,
                                page_sizes=page_sizes))
            pr.append(recall_at_k([x["page"] for x in h], r["gold_pages"], 5))
            gcs = set(r["gold_chunks"])
            cr.append(1.0 if any(x["chunk_id"] in gcs for x in h[:5]) else 0.0)
            pp.append(sum(1 for x in h[:K] if x["page"] in set(r["gold_pages"])) / K)
        n = len(qa)
        print(f"{name:12s} {sum(nd)/n:>9.4f} {sum(pr)/n:>8.4f} "
              f"{sum(cr)/n:>8.4f} {sum(pp)/n:>10.4f}")

    print("\n为对照：页级 Recall@5 的理论上界 1.0，本项目实测 0.7333")
    return 0


if __name__ == "__main__":
    sys.exit(main())
