"""把 runs/transfer/*/report.json 汇总成一张对比表，写 runs/transfer/SUMMARY.md。

只看已跑过的目录，缺的跳过——这样加一份新语料重跑一遍就有新行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs" / "transfer"

ORDER = ["style_guide_en", "speaker_manual_zh"]
LABEL = {
    "style_guide_en": "配色规范（英文，6 页）",
    "speaker_manual_zh": "产品说明书（中文，7 页）",
}


def main() -> int:
    runs = []
    for name in ORDER:
        p = BASE / name / "report.json"
        if p.is_file():
            runs.append((name, json.loads(p.read_text(encoding="utf-8"))))
    for d in sorted(BASE.iterdir()) if BASE.is_dir() else []:
        if d.is_dir() and d.name not in ORDER and (d / "report.json").is_file():
            runs.append((d.name, json.loads((d / "report.json").read_text(encoding="utf-8"))))

    if not runs:
        print("runs/transfer 下还没有报告")
        return 1

    md = ["# InsightRAG 迁移性能", "",
          "同一套代码、同一套模型（零训练），只换 PDF 重建索引。", "",
          "## 建库成本", "",
          "| 语料 | 页 | 字符 | 块 | 建库耗时 |", "|---|---|---|---|---|"]
    for name, r in runs:
        b = r["build"]
        md.append(f"| {LABEL.get(name, name)} | {b['pages']} | {b['chars']} | {b['chunks']} | "
                  f"{b['seconds']['total']} s |")

    md += ["", "## 问答质量", ""]
    head = "| 语料 | 题数 | 检索命中 | 生成命中 | 生成·宽松 | 引用诚实 | 拒答正确 | 假引用 | p50 |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    md += [head, sep]
    for name, r in runs:
        s = r.get("summary")
        if not s:
            continue
        f = lambda k: f"{s[k]['hits']}/{s[k]['n']}"
        lo = s.get("answer_hit_loose") or {}
        lo_s = f"{lo.get('hits')}/{lo.get('n')}" if lo else "—"
        md.append(f"| {LABEL.get(name, name)} | {s['n_questions']} | {f('retrieval_hit')} | "
                  f"{f('answer_hit')} | {lo_s} | {f('cite_hit')} | {f('refusal_ok')} | "
                  f"{s['fake_citations']} | {s['latency_ms']['p50']} ms |")

    md += ["",
           "**两个生成口径的差别**：「严格」要求答案里出现文档里的**专名/原词**；「宽松」也接受"
           "围绕同一实体的描述性措辞。样例文档的题目在原词层面已经能对上，所以两列相同。",
           "", "## 逐题失败项", "",
           "产品说明书那套的 z04（「环境噪声超过多少分贝时有效唤醒距离会下降？」）检索命中了"
           "正确页面，但模型答了「文档未提及」——原文写的是「环境噪声超过 65 分贝时有效距离降到"
           "约 2 米」，属于阈值型事实，模型没能从长句里把数字抽出来。这类题在原文库评测里同样"
           "是弱点（见 docs/REPORT.md §5）。",
           ""]

    md += ["", "## 对照：原论文库", "",
           "| 语料 | 页 | 块 | 建库耗时 | 页级 Recall@5 | nDCG@10 |",
           "|---|---|---|---|---|---|",
           "| 14 篇 RAG 论文（248 页） | 248 | 2421 | 13.4 s | 0.7333 | 0.3740 |", "",
           "原库 20 题评测来自 `runs/eval_v1.json`。两边判分口径不同（原库按 gold 页码算 recall，"
           "迁移演示按答案关键词是否出现在 top-k 块里），所以**这两行不能直接相减**，"
           "放在一起只为说明量级：建库成本 10–11 s，和原库 13.4 s 同一量级。", ""]

    (BASE / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    raise SystemExit(main())
