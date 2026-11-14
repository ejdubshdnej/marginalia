"""校验 qa_set.jsonl 的完整性。

检查项（对应 eval/ANNOTATION_SPEC.md §7）：
1. 行数 = 20
2. 四类配额 factoid 8 / multi_hop 4 / table 3 / unanswerable 5
3. 字段齐全，gold_pages 全是 0..247 的整数
4. gold_chunks 100% 能在 chunks.jsonl 里查到，且其页在 gold_pages 内
5. 非 unanswerable 题 gold_pages 非空；unanswerable 题 gold_pages / gold_chunks 为空
6. unanswerable 题的关键词在全语料出现 0 次（防止 Paris 型假拒答）
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insight_rag.config import CHUNKS_PATH  # noqa: E402

REQUIRED = ["qid", "question", "gold_pages", "gold_chunks", "gold_answer",
            "type", "difficulty"]
QUOTA = {"factoid": 8, "multi_hop": 4, "table": 3, "unanswerable": 5}

# unanswerable 题的「答案词」：这些短语在 2421 块里必须一次都不出现。
ABSENT_TERMS = {
    "q016": ["mercury", "boiling point"],
    "q017": ["beethoven", "symphony", "symphonies"],
    "q018": ["1.5 pro", "1.5-pro", "gemini 1.5"],
    "q019": ["vitamin", "dose"],
    "q020": ["carbon-14", "half-life", "half life", "radioactive"],
}


def main(path: str = "eval/qa_set.jsonl") -> int:
    src = Path(path)
    if not src.is_absolute():
        src = ROOT / src
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {c["chunk_id"]: c for c in chunks}
    n_pages = max(c["page"] for c in chunks) + 1
    blob = " ".join(c["text"] for c in chunks).lower()

    errs: list[str] = []
    if len(rows) != 20:
        errs.append(f"行数应为 20，实际 {len(rows)}")

    for r in rows:
        qid = r.get("qid", "?")
        for f in REQUIRED:
            if f not in r:
                errs.append(f"{qid}: 缺字段 {f}")
        pages = r.get("gold_pages") or []
        gc = r.get("gold_chunks") or []
        for cid in gc:
            if cid not in by_id:
                errs.append(f"{qid}: gold_chunk 查不到 {cid}")
            elif by_id[cid]["page"] not in pages:
                errs.append(f"{qid}: {cid} 的页 {by_id[cid]['page']} 不在 gold_pages")
        for pg in pages:
            if not isinstance(pg, int) or not (0 <= pg < n_pages):
                errs.append(f"{qid}: gold_page {pg!r} 非法（共 {n_pages} 页）")
        if r.get("type") == "unanswerable":
            if pages or gc:
                errs.append(f"{qid}: unanswerable 的 gold_pages/gold_chunks 必须为空")
            for term in ABSENT_TERMS.get(qid, []):
                n = blob.count(term)
                if n:
                    errs.append(f"{qid}: 答案词 {term!r} 在语料出现 {n} 次，不是真 unanswerable")
        elif not pages:
            errs.append(f"{qid}: 非 unanswerable 但 gold_pages 为空")

    types = Counter(r.get("type") for r in rows)
    print(f"条数 {len(rows)}   类型 {dict(types)}")
    for t, q in QUOTA.items():
        if types.get(t, 0) != q:
            errs.append(f"配额 {t} 应为 {q}，实际 {types.get(t, 0)}")

    if errs:
        print(f"\n❌ {len(errs)} 个问题：")
        for e in errs:
            print("  -", e)
        return 1

    hard = sum(1 for r in rows if r.get("difficulty") == "hard")
    print(f"难度 hard {hard} 条 / {len(rows)}")
    print("\n✅ 校验通过：20 行、四类配额对、gold_pages 合法、gold_chunks 100% 命中、unanswerable 答案词 0 次")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
