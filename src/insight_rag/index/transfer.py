"""任意 PDF → 独立索引目录。迁移演示与网页上传共用这一份实现。

为什么不复用 parse_pdf()：它把页码硬映射进 14 篇论文的 manifest，换文档后
paper 名会全错。这里逐页直接调 column_split()（双栏重排），单栏文档同样正常。
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from insight_rag import config
from insight_rag.index import bm25_store, embedder, vector_store
from insight_rag.parse.chunker import chunk_pages
from insight_rag.parse.pdf_parser import column_split

INDEX_FILES = ("faiss.index", "chunks.jsonl", "tokens.json")


def safe_name(name: str) -> str:
    """目录名必须是纯 ASCII。

    原因：faiss 的 C++ 层用 fopen 打开路径，Windows 上走的是 ANSI 代码页，
    中文路径会直接报 "could not open ... for writing"。所以这里只留
    [A-Za-z0-9_-]，中文名保留在 meta.json 的 title 里给界面显示。
    """
    raw = (name or "").strip()
    if not raw:
        return "doc"
    s = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("._-")[:48]
    if not s:
        # 全中文这类取不出 ASCII 的名字，用内容哈希当目录名，稳定且不重复
        s = "doc-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
    return s


def write_title(d: Path, title: str) -> None:
    """把人类可读的名字存进 meta.json，目录名保持 ASCII。"""
    meta = {"title": title or d.name}
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_title(d: Path) -> str:
    p = d / "meta.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("title") or d.name
        except (OSError, json.JSONDecodeError):
            pass
    return d.name


def parse_any_pdf(path: Path) -> list[dict]:
    """解析任意 PDF，返回 [{page, text, paper}]。"""
    import pymupdf

    doc = pymupdf.open(path)
    stem = path.stem
    out: list[dict] = []
    try:
        for pno, page in enumerate(doc):
            out.append({"page": pno, "text": column_split(page), "paper": stem})
    finally:
        doc.close()
    return out


def index_ready(d: Path) -> bool:
    return all((d / f).is_file() and (d / f).stat().st_size > 0 for f in INDEX_FILES)


def build_to(pdf: Path, out_dir: Path, verbose: bool = False, title: str = "") -> dict:
    """把 pdf 建库到 out_dir。**不读也不写 config 里的路径**，只写 out_dir。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    pages = parse_any_pdf(pdf)
    parse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_pages(pages)
    chunk_s = time.perf_counter() - t0
    texts = [c["text"] for c in chunks]
    if not chunks:
        raise ValueError(f"{pdf.name} 里没有可索引的文字（可能是扫描件，需要先 OCR）")

    # 首次调用会加载 BGE 模型并编译 CUDA kernel，单独计时，别混进 embed 里
    t0 = time.perf_counter()
    embedder.embed_docs(texts[:32])
    warm_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb = embedder.embed_docs(texts)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = vector_store.build(emb)
    vector_store.save(index, out_dir / "faiss.index")
    faiss_s = time.perf_counter() - t0

    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    t0 = time.perf_counter()
    bm25_store.save_tokens(texts, out_dir / "tokens.json")
    bm25_store.build_from_tokens(bm25_store.load_tokens(out_dir / "tokens.json"))
    bm25_s = time.perf_counter() - t0

    info = {
        "pdf": str(pdf),
        "name": out_dir.name,
        "title": title or out_dir.name,
        "dir": str(out_dir),
        "pdf_mb": round(pdf.stat().st_size / 1024 / 1024, 2),
        "pages": len(pages),
        "chars": sum(len(p["text"]) for p in pages),
        "chunks": len(chunks),
        "dim": int(emb.shape[1]),
        "faiss_mb": round((out_dir / "faiss.index").stat().st_size / 1024 / 1024, 2),
        "seconds": {
            "parse": round(parse_s, 2),
            "chunk": round(chunk_s, 2),
            "warmup": round(warm_s, 2),
            "embed": round(embed_s, 2),
            "faiss_save": round(faiss_s, 2),
            "bm25": round(bm25_s, 2),
            "total": round(time.perf_counter() - t_all, 2),
        },
    }
    write_title(out_dir, info["title"])
    if verbose:
        print(f"  {info['pages']} 页 / {info['chunks']} 块 / {info['seconds']['total']}s -> {out_dir}")
    return info


# ---------------------------------------------------------------- 索引清单

def read_meta(d: Path) -> dict:
    """读一个索引目录的概况：块数、页数、语料名、各文件大小与时间。"""
    d = Path(d)
    meta: dict = {
        "dir": str(d),
        "name": d.name,
        "title": read_title(d),
        "ready": index_ready(d),
        "files": [],
        "chunks": 0,
        "pages": 0,
        "papers": [],
    }
    for f in INDEX_FILES:
        p = d / f
        if p.is_file():
            st = p.stat()
            meta["files"].append({"name": f, "bytes": st.st_size, "mtime": int(st.st_mtime)})

    cp = d / "chunks.jsonl"
    if cp.is_file():
        pages: set[int] = set()
        papers: set[str] = set()
        n = 0
        try:
            with open(cp, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    c = json.loads(line)
                    n += 1
                    if "page" in c:
                        pages.add(c["page"])
                    if c.get("paper"):
                        papers.add(c["paper"])
        except (OSError, json.JSONDecodeError):
            pass
        meta["chunks"] = n
        meta["pages"] = len(pages)
        meta["papers"] = sorted(papers)
    return meta


def list_indexes(active: Path | None = None) -> list[dict]:
    """列出可用索引：默认库 + runs/transfer/*。

    只列「三件套齐全」的目录，半成品不出现，免得点进去起不来。
    """
    active = Path(active or config.INDEX_DIR).resolve()
    base = config.RUNS_DIR / "transfer"
    cands: list[Path] = [config.ROOT / "runs" / "index"]
    if base.is_dir():
        cands += sorted(p for p in base.iterdir() if p.is_dir())

    out: list[dict] = []
    for d in cands:
        if not index_ready(d):
            continue
        m = read_meta(d)
        if d.resolve() == (config.ROOT / "runs" / "index").resolve():
            m["title"] = "论文库（14 篇 RAG 论文）"   # 默认库没有 meta.json，给个牌子
        m["active"] = d.resolve() == active
        m["built_at"] = max((f["mtime"] for f in m["files"]), default=0)
        out.append(m)
    return out


def remove_index(d: Path) -> bool:
    """删掉一套（仅限 runs/transfer 下的），防止误删默认库。"""
    d = Path(d).resolve()
    base = (config.RUNS_DIR / "transfer").resolve()
    if base not in d.parents:
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not d.exists()
