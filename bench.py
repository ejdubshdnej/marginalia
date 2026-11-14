"""端到端延迟基准：检索 / 重排 / 生成 / E2E 的 mean、p50、p95。

跑的是生产链路本身（`hybrid_search` → `Reranker.rerank` → `LLM.chat`），
题目取自 `eval/qa_set.jsonl`，所以这个脚本产出的数字就是 README 性能表的出处。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 bench.py
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 bench.py --reps 4 --json runs/bench_mine.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from insight_rag.config import MAX_TOKENS, ROOT, TOP_K_FUSED, TOP_K_RERANK  # noqa: E402
from insight_rag.generate.llm import LLM  # noqa: E402
from insight_rag.generate.prompt import SYSTEM, build_prompt  # noqa: E402
from insight_rag.generate.rag import init  # noqa: E402
from insight_rag.retrieve.hybrid import hybrid_search  # noqa: E402


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def load_questions(path: Path, limit: int | None) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [r["question"] for r in rows if r.get("question")]
    return questions[:limit] if limit else questions


def main() -> int:
    ap = argparse.ArgumentParser(description="InsightRAG 端到端延迟基准")
    ap.add_argument("--qa", default=str(ROOT / "eval" / "qa_set.jsonl"), help="题目文件（jsonl，取 question 字段）")
    ap.add_argument("--reps", type=int, default=4, help="整套题目重复几轮")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 道题")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--json", default=None, help="把汇总写到这个路径")
    args = ap.parse_args()

    questions = load_questions(Path(args.qa), args.limit)
    if not questions:
        print(f"[X] {args.qa} 里没有题")
        return 1

    print(f"装载模型（首轮会编译 CUDA kernel，约 30 s）…")
    t0 = time.time()
    st = init()
    print(f"  就绪，用时 {time.time() - t0:.1f}s；索引 {len(st['chunks'])} 块；共 {len(questions)} 道题 × {args.reps} 轮")

    retrieve_ms: list[float] = []
    generate_ms: list[float] = []
    tokens_per_s: list[float] = []
    samples: list[dict] = []

    llm: LLM = st["llm"]
    for rep in range(args.reps):
        for q in questions:
            t_a = time.perf_counter()
            fused = hybrid_search(q, top_k=TOP_K_FUSED, state=st["hybrid_state"])
            top = st["reranker"].rerank(q, fused, top_k=TOP_K_RERANK)
            t_b = time.perf_counter()

            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(q, top)}]
            out = llm.chat(messages, max_tokens=args.max_tokens, temperature=0.0)
            t_c = time.perf_counter()

            completion = out["usage"]["completion_tokens"]
            gen_s = t_c - t_b
            retrieve_ms.append((t_b - t_a) * 1000)
            generate_ms.append(gen_s * 1000)
            if gen_s > 0:
                tokens_per_s.append(completion / gen_s)
            samples.append({"question": q, "completion_tokens": completion})
        print(f"  第 {rep + 1}/{args.reps} 轮完成")

    e2e = [r + g for r, g in zip(retrieve_ms, generate_ms)]
    rows = [
        ("retrieve", retrieve_ms),
        ("generate", generate_ms),
        ("E2E", e2e),
    ]

    print(f"\n{'stage':10s} {'mean':>9s} {'p50':>9s} {'p95':>9s}   (ms, n={len(retrieve_ms)})")
    summary: dict[str, float] = {}
    for name, vals in rows:
        print(f"{name:10s} {statistics.mean(vals):9.1f} {pct(vals, 0.50):9.1f} {pct(vals, 0.95):9.1f}")
        summary[f"{name}_mean_ms"] = round(statistics.mean(vals), 1)
        summary[f"{name}_p50_ms"] = round(pct(vals, 0.50), 1)
        summary[f"{name}_p95_ms"] = round(pct(vals, 0.95), 1)
    print(f"gen avg   {statistics.mean(tokens_per_s):9.1f} tok/s")
    summary["gen_mean_tok_s"] = round(statistics.mean(tokens_per_s), 1)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "n": len(retrieve_ms), "reps": args.reps,
                        "questions": len(questions), "samples": samples},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n汇总已写入 {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
