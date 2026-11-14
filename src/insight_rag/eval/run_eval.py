"""完整评测：跑 20 条 qa_set，输出 runs/eval_v1.json + 控制台汇总表。

流程：
  1. 按 eval/ANNOTATION_SPEC.md 的口径，对每条有 gold 的题取 top-10 混合检索候选
     （A 组：向量+BM25 → RRF → 截 10，不用 reranker，因为 Recall/nDCG 测的是检索器）
  2. recall_at_k(k=5) / ndcg_at_k(k=10)
  3. 对全部 20 条跑完整 RAG 链路（含 reranker + LLM）取答案，算 faithfulness
  4. unanswerable 单独统计「是否正确拒答」
  5. 随机基线：100 次抽 5 个随机块，按页级口径算 Recall@5

命令：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.eval.run_eval \
      --qa eval/qa_set.jsonl --out runs/eval_v1.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from insight_rag.config import ROOT, RUNS_DIR, CHUNKS_PATH
from insight_rag.eval.metrics import recall_at_k, ndcg_at_k, faithfulness, page_size_map

REFUSAL_MARKERS = ("文档未提及", "文档中没有", "未提及", "not mentioned", "no information")


def load_qa(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def random_baseline(rows: list[dict], chunks: list[dict], n_draws: int = 100, k: int = 5,
                    seed: int = 20261113) -> dict:
    """随机抽 k 块当检索结果，重复 n_draws 次，看页级 Recall@5 的均值。"""
    rng = random.Random(seed)
    graded = [r for r in rows if r["gold_pages"]]
    hits = []
    for _ in range(n_draws):
        for r in graded:
            pages = [rng.choice(chunks)["page"] for _ in range(k)]
            hits.append(recall_at_k(pages, r["gold_pages"], k))
    mean = sum(hits) / len(hits)
    return {"mean_recall_at_5": round(mean, 4), "n_draws": n_draws, "k": k,
            "n_questions": len(graded), "seed": seed}


def is_refusal(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if len(a) <= 24:
        return any(m.lower() in a for m in REFUSAL_MARKERS)
    return any(m.lower() in a for m in REFUSAL_MARKERS) and len(a) <= 120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="eval/qa_set.jsonl")
    ap.add_argument("--out", default="runs/eval_v1.json")
    ap.add_argument("--verbose-faith", action="store_true")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.is_absolute():
        qa_path = ROOT / qa_path
    rows = load_qa(qa_path)
    chunks = load_chunks()
    page_sizes = page_size_map(chunks)
    print(f"qa_set: {len(rows)} 条   chunks: {len(chunks)}")

    from insight_rag.retrieve import hybrid
    from insight_rag.generate import rag as rag_mod

    t0 = time.perf_counter()
    hybrid.load()
    rag_mod.init()
    print(f"模型就绪 {time.perf_counter() - t0:.1f}s")

    # ---- 1. 检索指标（A 组：RRF，不含 reranker） ----
    per_q: list[dict] = []
    t_ret = time.perf_counter()
    for r in rows:
        hits = hybrid.hybrid_search(r["question"], top_k=10)
        pages = [h["page"] for h in hits]
        rec = recall_at_k(pages, r["gold_pages"], 5) if r["gold_pages"] else None
        ndcg = ndcg_at_k(hits, r["gold_chunks"], r["gold_pages"], 10,
                         page_sizes=page_sizes) if r["gold_pages"] else None
        per_q.append({"qid": r["qid"], "type": r["type"], "recall_at_5": rec,
                      "ndcg_at_10": ndcg, "top_pages": pages,
                      "top_chunks": [h["chunk_id"] for h in hits]})
    ret_ms = (time.perf_counter() - t_ret) / len(rows) * 1000

    graded = [x for x in per_q if x["recall_at_5"] is not None]
    recall = sum(x["recall_at_5"] for x in graded) / len(graded)
    ndcg = sum(x["ndcg_at_10"] for x in graded) / len(graded)

    # ---- 2. 生成 + faithfulness ----
    ans_rows: list[dict] = []
    faiths: list[float] = []
    t_gen = time.perf_counter()
    for i, r in enumerate(rows, 1):
        res = rag_mod.ask(r["question"], top_k=5)
        ans = res["answer"]
        ctx = res["citations"]
        entry = {"qid": r["qid"], "type": r["type"], "question": r["question"],
                 "answer": ans, "citations": [c["chunk_id"] for c in ctx],
                 "latency_ms": res["latency"]["total_ms"],
                 "citation_check": res["citation_check"]}

        if r["type"] == "unanswerable":
            entry["refused"] = is_refusal(ans)
        else:
            f = faithfulness(ans, ctx, llm=rag_mod._STATE["llm"],
                             verbose=args.verbose_faith)
            entry["faithfulness"] = round(f, 3)
            faiths.append(f)
        ans_rows.append(entry)
        print(f"  [{i:2d}/20] {r['qid']} {r['type']:12s} "
              f"{entry.get('faithfulness', 'refuse=' + str(entry.get('refused')))} "
              f"{res['latency']['total_ms']:7.0f}ms")
    gen_ms = (time.perf_counter() - t_gen) / len(rows) * 1000

    faith_mean = sum(faiths) / len(faiths) if faiths else 0.0
    unans = [x for x in ans_rows if x["type"] == "unanswerable"]
    unans_ok = sum(1 for x in unans if x.get("refused"))
    unans_rate = unans_ok / len(unans) if unans else 0.0

    # ---- 3. 随机基线 ----
    rnd = random_baseline(rows, chunks)

    # ---- 4. 汇总 ----
    summary = {
        "recall_at_5": round(recall, 4),
        "ndcg_at_10": round(ndcg, 4),
        "faithfulness": round(faith_mean, 4),
        "unanswerable_refusal_rate": round(unans_rate, 4),
        "unanswerable_correct": f"{unans_ok}/{len(unans)}",
        "n_graded": len(graded),
        "random_baseline": rnd,
        "retrieve_ms_per_q": round(ret_ms, 1),
        "generate_ms_per_q": round(gen_ms, 1),
    }
    gates = {
        "recall_at_5": (summary["recall_at_5"], 0.60, ">="),
        "ndcg_at_10": (summary["ndcg_at_10"], 0.55, ">="),
        "faithfulness": (summary["faithfulness"], 0.70, ">="),
        "unanswerable": (summary["unanswerable_refusal_rate"], 0.60, ">="),
        "random_baseline": (rnd["mean_recall_at_5"], 0.030, "<="),
        "random_baseline_lo": (rnd["mean_recall_at_5"], 0.015, ">="),
    }
    summary["gates"] = {k: {"value": v[0], "limit": v[1], "op": v[2],
                            "pass": (v[0] >= v[1]) if v[2] == ">=" else (v[0] <= v[1])}
                        for k, v in gates.items()}
    all_pass = all(g["pass"] for g in summary["gates"].values())

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qa_set": str(qa_path.relative_to(ROOT)),
        "n_questions": len(rows),
        "summary": summary,
        "per_question": per_q,
        "answers": ans_rows,
        "all_gates_pass": all_pass,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'指标':<26}{'实测':>10}{'门槛':>12}   判定")
    print("-" * 62)
    names = {"recall_at_5": "Recall@5 (页级)", "ndcg_at_10": "nDCG@10 (块级)",
             "faithfulness": "Faithfulness", "unanswerable": "unanswerable 拒答率",
             "random_baseline": "随机基线 ≤0.030", "random_baseline_lo": "随机基线 ≥0.015"}
    for k, g in summary["gates"].items():
        op = "≥" if g["op"] == ">=" else "≤"
        print(f"{names[k]:<26}{g['value']:>10.4f}{op + str(g['limit']):>12}   "
              f"{'✅' if g['pass'] else '❌'}")
    print("-" * 62)
    print(f"{'unanswerable 答对':<26}{summary['unanswerable_correct']:>10}")
    print(f"{'检索耗时/题':<26}{summary['retrieve_ms_per_q']:>9.0f}ms")
    print(f"{'端到端耗时/题':<26}{summary['generate_ms_per_q']:>9.0f}ms")
    print("=" * 62)
    print(f"{'✅ 全部门槛通过' if all_pass else '❌ 有门槛未通过'}")
    print(f"产物 {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
