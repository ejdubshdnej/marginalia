"""reranker base vs large 对照：单次耗时 + 重排后的检索质量。

回答一个具体的选型问题：「为什么用 bge-reranker-base 而不是 large」。
本脚本在同一批候选上跑两个模型，给出延迟与 nDCG 的实测对照。
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")

import numpy as np

from insight_rag import config
from insight_rag.eval.metrics import ndcg_at_k, page_size_map, recall_at_k
from insight_rag.retrieve import hybrid


def load_graded():
    rows = []
    with open(config.ROOT / "eval" / "qa_set.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["gold_pages"]:
                rows.append(r)
    return rows


def run(model_dir: str, rows, chunks, ps):
    import insight_rag.retrieve.reranker as RR

    t0 = time.time()
    rr = RR.Reranker(path=str(config.ROOT / "models" / model_dir))
    rr.warmup()
    load_s = time.time() - t0

    lat = []
    recalls, ndcgs = [], []
    for r in rows:
        q = r["question"]
        hits = hybrid.hybrid_search(q, top_k=20)
        t0 = time.time()
        top = rr.rerank(q, hits, top_k=10)
        lat.append(rr.last_ms)
        pages = [h["page"] for h in top]
        recalls.append(recall_at_k(pages, r["gold_pages"], 5))
        ndcgs.append(
            ndcg_at_k(top, r["gold_chunks"], r["gold_pages"], 10, page_sizes=ps)
        )

    def pct(xs, p):
        return float(np.percentile(xs, p))

    return {
        "model": model_dir,
        "load_s": round(load_s, 1),
        "recall_at_5": round(float(np.mean(recalls)), 4),
        "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
        "latency_mean_ms": round(float(np.mean(lat)), 1),
        "latency_p95_ms": round(pct(lat, 95), 1),
        "latency_max_ms": round(float(np.max(lat)), 1),
    }


def main() -> int:
    rows = load_graded()
    print(f"graded questions: {len(rows)}")

    chunks_path = config.CHUNKS_PATH
    chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
    ps = page_size_map(chunks)
    hybrid.load()

    out = []
    for name in ("bge-reranker-base", "bge-reranker-large"):
        if not (config.ROOT / "models" / name).exists():
            print(f"SKIP {name} (未下载)")
            continue
        r = run(name, rows, chunks, ps)
        out.append(r)
        print(f"  {name:22s} nDCG@10 {r['ndcg_at_10']:.4f}  "
              f"R@5 {r['recall_at_5']:.4f}  "
              f"mean {r['latency_mean_ms']:.1f} ms  p95 {r['latency_p95_ms']:.1f} ms  "
              f"load {r['load_s']}s")

    Path("runs/reranker_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("-> runs/reranker_compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
