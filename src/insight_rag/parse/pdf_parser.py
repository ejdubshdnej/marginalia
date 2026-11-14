"""PDF 解析：逐页取文本，可选双栏重排。

接口契约（后续模块按此签名调用，不许改）：
    parse_pdf(path, use_column_split=True) -> [{"page": int, "text": str, "paper": str}, ...]
    column_split(page, mid=306.0) -> str
"""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from insight_rag import config

_MANIFEST_CACHE: list[dict] | None = None


def load_manifest(path: Path | str | None = None) -> list[dict]:
    """读论文清单，用于给每一页标注所属论文。"""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None and path is None:
        return _MANIFEST_CACHE
    p = Path(path) if path is not None else config.MANIFEST
    data = json.loads(p.read_text(encoding="utf-8"))
    papers = data["papers"]
    if path is None:
        _MANIFEST_CACHE = papers
    return papers


def paper_of_page(page_no: int, papers: list[dict] | None = None) -> str:
    """页码 -> 论文标题。清单缺页时返回 'unknown'，不抛异常。"""
    papers = papers if papers is not None else load_manifest()
    for p in papers:
        if p["page_start"] <= page_no <= p["page_end"]:
            return p["title"]
    return "unknown"


def column_split(page, mid: float = config.COLUMN_MID) -> str:
    """把一个 pymupdf.Page 的文本块按「左栏 -> 跨栏 -> 右栏」重排。

    实测唯一有效的方案：按块的 x 边界归类，各自按 (y, x) 排序后拼接。
    """
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    left = sorted([b for b in blocks if b[2] <= mid + 5], key=lambda b: (b[1], b[0]))
    right = sorted([b for b in blocks if b[0] >= mid - 5], key=lambda b: (b[1], b[0]))
    mid_ = [b for b in blocks if b not in left and b not in right]
    return "\n".join(b[4] for b in left + sorted(mid_, key=lambda b: b[1]) + right)


def parse_pdf(path: str | Path, use_column_split: bool = True,
              manifest_path: str | Path | None = None) -> list[dict]:
    """逐页解析。返回页级记录，text 为该页最终采用的正文。"""
    path = Path(path)
    papers = load_manifest(manifest_path)
    doc = pymupdf.open(path)
    out: list[dict] = []
    try:
        for pno, page in enumerate(doc):
            text = column_split(page) if use_column_split else page.get_text()
            out.append({
                "page": pno,
                "text": text,
                "paper": paper_of_page(pno, papers),
            })
    finally:
        doc.close()
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    pages = parse_pdf(config.CORPUS_PDF)
    print("pages", len(pages), "chars", sum(len(p["text"]) for p in pages))
    print(pages[0]["paper"], "|", pages[0]["text"][:120].replace("\n", " "))
