"""页内切块。char_start / char_end 是**页内**字符偏移，溯源靠它。"""
from __future__ import annotations

from typing import Iterable

from insight_rag import config


def make_chunk_id(page_no: int, idx_in_page: int) -> str:
    """格式 "p{page}_b{idx:03d}"，例：p12_b003。必须能反解出页码。"""
    return f"p{page_no}_b{idx_in_page:03d}"


def parse_chunk_id(chunk_id: str) -> tuple[int, int]:
    """chunk_id -> (page, idx)。反解失败抛 ValueError。"""
    page_s, block_s = chunk_id.split("_b")
    if not page_s.startswith("p"):
        raise ValueError(f"非法 chunk_id: {chunk_id}")
    return int(page_s[1:]), int(block_s)


def chunk_page_text(text: str, size: int = config.CHUNK_SIZE,
                    overlap: int = config.CHUNK_OVERLAP,
                    min_len: int = config.MIN_CHUNK_LEN) -> list[tuple[int, int, str]]:
    """对单页正文做滑动窗口切块，返回 [(char_start, char_end, text), ...]。

    窗口步长 = size - overlap；末尾窗口若已覆盖到页尾则不再重复产出。
    """
    stride = size - overlap
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    for s in range(0, max(1, n - overlap), stride):
        t = text[s:s + size]
        if len(t.strip()) < min_len:
            continue
        spans.append((s, s + len(t), t))
    return spans


def chunk_pages(pages: Iterable[dict], size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP,
                min_len: int = config.MIN_CHUNK_LEN) -> list[dict]:
    """pages -> 带完整元数据的块列表。

    每块：chunk_id / page / paper / char_start / char_end / text / type
    """
    out: list[dict] = []
    for p in pages:
        page_no = p["page"]
        paper = p.get("paper", "unknown")
        for idx, (cs, ce, t) in enumerate(chunk_page_text(p["text"], size, overlap, min_len)):
            out.append({
                "chunk_id": make_chunk_id(page_no, idx),
                "page": page_no,
                "paper": paper,
                "char_start": cs,
                "char_end": ce,
                "text": t,
                "type": "text",
            })
    return out


if __name__ == "__main__":
    import sys

    from insight_rag.parse.pdf_parser import parse_pdf

    sys.stdout.reconfigure(encoding="utf-8")
    pages = parse_pdf(config.CORPUS_PDF)
    for sz, ov in ((500, 100), (300, 60), (800, 160)):
        ch = chunk_pages(pages, sz, ov)
        print(f"size={sz} ov={ov} -> {len(ch)} 块")
    ch = chunk_pages(pages)
    print("sample:", ch[0]["chunk_id"], ch[0]["page"], ch[0]["char_start"], ch[0]["char_end"])
