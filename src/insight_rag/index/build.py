"""建库 CLI：解析 → 切块 → 向量化 → 落盘（faiss + chunks.jsonl + tokens.json）。

用法：
    PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.index.build

与 api/main.py 的 POST /index 做的是同一件事，区别是这个入口不需要起服务，
适合首次建库或 CI 里重建索引。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from insight_rag import config
from insight_rag.index import bm25_store, embedder, vector_store
from insight_rag.parse.chunker import chunk_pages
from insight_rag.parse.pdf_parser import parse_pdf


def build(pdf_path: Path | None = None, verbose: bool = True) -> dict:
    """建库并落盘，返回统计信息。"""
    pdf_path = Path(pdf_path) if pdf_path is not None else config.CORPUS_PDF
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()

    t0 = time.perf_counter()
    pages = parse_pdf(pdf_path)
    parse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_pages(pages)
    chunk_s = time.perf_counter() - t0

    texts = [c["text"] for c in chunks]

    t0 = time.perf_counter()
    embedder.embed_docs(texts[:32])  # warm-up：模型加载 + CUDA kernel 编译
    warm_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb = embedder.embed_docs(texts)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = vector_store.build(emb)
    vector_store.save(index, config.FAISS_PATH)
    save_s = time.perf_counter() - t0

    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    t0 = time.perf_counter()
    bm25_store.save_tokens(texts, config.TOKENS_PATH)
    bm25_store.build_from_tokens(bm25_store.load_tokens(config.TOKENS_PATH))
    bm25_s = time.perf_counter() - t0

    total = time.perf_counter() - t_all
    size_mb = Path(config.FAISS_PATH).stat().st_size / 1024 / 1024

    info = {
        "pages": len(pages),
        "chunks": len(chunks),
        "dim": int(emb.shape[1]),
        "faiss_mb": round(size_mb, 2),
        "seconds": {
            "parse": round(parse_s, 2),
            "chunk": round(chunk_s, 2),
            "warmup": round(warm_s, 2),
            "embed": round(embed_s, 2),
            "faiss_save": round(save_s, 2),
            "bm25": round(bm25_s, 2),
            "total": round(total, 2),
        },
    }

    if verbose:
        print(f"pages  {info['pages']}")
        print(f"chunks {info['chunks']}  dim {info['dim']}")
        print(f"embed  {info['seconds']['embed']}s  "
              f"({info['chunks'] / max(info['seconds']['embed'], 1e-9):.1f} 块/秒)")
        print(f"faiss  {info['faiss_mb']} MB -> {config.FAISS_PATH}")
        print(f"chunks -> {config.CHUNKS_PATH}")
        print(f"tokens -> {config.TOKENS_PATH}")
        print(f"total  {info['seconds']['total']}s")

    return info


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not config.CORPUS_PDF.exists():
        print(f"FAIL 语料不存在: {config.CORPUS_PDF}")
        return 1
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
