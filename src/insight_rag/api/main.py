"""FastAPI 服务：问答 + 索引管理（上传建库 / 切换 / 删除）。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from insight_rag import config
from insight_rag.api import registry
from insight_rag.generate import rag

MAX_UPLOAD_MB = 40


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SwitchRequest(BaseModel):
    dir: str


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    latency: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.init()
    yield


app = FastAPI(title="InsightRAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEB_DIR = config.ROOT / "web"


def _health() -> dict:
    from insight_rag.index import transfer

    st = rag._STATE
    d = registry.current()
    is_default = d == registry.DEFAULT_DIR.resolve()
    return {
        "status": "ok",
        "model": config.LLM_GGUF.name,
        "chunks": len(st.get("chunks", [])),
        "index_dir": str(d),
        "index_name": "论文库（14 篇 RAG 论文）" if is_default else transfer.read_title(d),
        "is_default": is_default,
    }


@app.get("/health")
def health() -> dict:
    return _health()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return rag.ask(req.question, top_k=req.top_k)


@app.post("/index")
def rebuild_index() -> dict:
    """重建**当前**索引。默认库重跑语料，迁移库重跑它自己的那份 PDF。"""
    import json
    import time

    from insight_rag.index import bm25_store, embedder, transfer, vector_store
    from insight_rag.parse.chunker import chunk_pages
    from insight_rag.parse.pdf_parser import parse_pdf

    t0 = time.time()
    cur = registry.current()
    src = cur / "source.pdf"          # 迁移库建的时候会留一份源 PDF
    if src.is_file():
        # 迁移语料不能用 parse_pdf：它按 manifest 把页码映射成 14 篇论文的名字，
        # 换过的文档会全标成错的论文。这里和建库走同一个解析入口。
        pages = transfer.parse_any_pdf(src)
    elif cur == registry.DEFAULT_DIR.resolve():
        pages = parse_pdf(config.CORPUS_PDF)
    else:
        # 迁移库却没有 source.pdf（例如用命令行建的），拿默认语料重跑会把
        # 论文块写进这一套迁移索引里——宁可报错，也不能悄悄写坏它。
        raise HTTPException(
            400,
            "这套迁移索引没有留下源 PDF，不能就地重建。"
            "请用「迁移新文档」重新上传建一套，或用命令行 tools/transfer_demo.py 重建。",
        )
    chunks = chunk_pages(pages)
    texts = [c["text"] for c in chunks]

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb = embedder.embed_docs(texts)
    index = vector_store.build(emb)
    vector_store.save(index, config.FAISS_PATH)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    bm25_store.save_tokens(texts, config.TOKENS_PATH)

    # 重载索引但**不重载模型**：reranker 和 llm 实例还在，重载一次要多花 33 s
    registry.activate(cur)

    return {"chunks": len(chunks), "elapsed": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- 索引管理

@app.get("/indexes")
def list_indexes() -> dict:
    from insight_rag.index import transfer

    return {
        "active": str(registry.current()),
        "busy": registry.is_busy(),
        "default_dir": str(registry.DEFAULT_DIR),
        "items": transfer.list_indexes(registry.current()),
    }


@app.post("/indexes/upload")
async def upload_index(file: UploadFile = File(...), name: str = Form("")) -> dict:
    """收一个 PDF，建一套独立索引。建完**不自动切换**——切换是显式动作。"""
    blob = await file.read()
    if not blob:
        raise HTTPException(400, "文件是空的")
    if len(blob) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"超过 {MAX_UPLOAD_MB} MB，请先在本地压缩或用命令行建库")
    if not blob.startswith(b"%PDF"):
        raise HTTPException(400, "不是 PDF（文件头不是 %PDF）")
    if registry.is_busy()["building"]:
        raise HTTPException(409, "另一个建库任务正在跑，等它完成")

    try:
        info = registry.build_from_upload(blob, file.filename or "upload.pdf", name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"建库失败：{type(e).__name__}: {e}") from e
    return {"ok": True, "info": info}


@app.post("/indexes/switch")
def switch_index(req: SwitchRequest) -> dict:
    d = registry.resolve(req.dir)
    if d is None:
        raise HTTPException(400, "只允许切换 runs/index 和 runs/transfer 下的索引")
    if d == registry.current():
        return {"ok": True, "unchanged": True, **_health()}
    try:
        info = registry.activate(d)
    except FileNotFoundError as e:
        raise HTTPException(404, f"索引文件缺失：{e}") from e
    except Exception as e:
        raise HTTPException(500, f"切换失败：{type(e).__name__}: {e}") from e
    return {"ok": True, **info, **_health()}


@app.delete("/indexes")
def delete_index(dir: str) -> dict:
    d = registry.resolve(dir)
    if d is None:
        raise HTTPException(400, "只允许删除 runs/transfer 下的索引")
    if not registry.remove(d):
        raise HTTPException(400, "正在使用的索引或默认库不能删")
    return {"ok": True}


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index_page():
        return FileResponse(str(WEB_DIR / "index.html"))
