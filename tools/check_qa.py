"""校验 qa_set.jsonl：字段齐全、四类都有、gold_pages 合法、gold_chunks 全部能在索引里查到。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insight_rag.config import CHUNKS_PATH  # noqa: E402

REQUIRED = ["qid", "question", "type", "gold_pages", "gold_chunks", "difficulty"]
QUOTA = {"factoid": 8, "multi_hop": 4, "table": 3, "unanswerable": 5}


def main(path: str = "eval/qa_set.jsonl") -> int:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {c["chunk_id"]: c for c in chunks}
    n_pages = max(c["page"] for c in chunks) + 1

    errs: list[str] = []
    for r in rows:
        for f in REQUIRED:
            if f not in r:
                errs.append(f"{r.get('id','?')}: 缺字段 {f}")
        for cid in r.get("gold_chunks") or []:
            if cid not in by_id:
                errs.append(f"{r["qid"]}: gold_chunk 查不到 {cid}")
            elif by_id[cid]["page"] not in r.get("gold_pages", []):
                errs.append(f"{r["qid"]}: {cid} 的页 {by_id[cid]['page']} 不在 gold_pages")
        for pg in r.get("gold_pages") or []:
            if not (0 <= pg < n_pages):
                errs.append(f"{r["qid"]}: gold_page {pg} 越界（共 {n_pages} 页）")

    types = Counter(r.get("type") for r in rows)
    print(f"条数: {len(rows)}   类型: {dict(types)}")
    for t, q in QUOTA.items():
        if types.get(t, 0) != q:
            errs.append(f"配额 {t} 应为 {q}，实际 {types.get(t,0)}")
    unmatched = [r for r in rows if r["type"] != "unanswerable" and not r.get("gold_pages")]
    for r in unmatched:
        errs.append(f"{r["qid"]}: 非 unanswerable 但 gold_pages 为空")

    if errs:
        print(f"\n❌ {len(errs)} 个问题:")
        for e in errs:
            print("  -", e)
        return 1
    print("\n✅ 全部通过：字段齐、四类配额对、gold_pages 合法、gold_chunks 100% 命中")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
