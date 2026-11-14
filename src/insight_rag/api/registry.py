"""索引注册表：列出 / 建新 / 切换 / 删除。

切换是这里的关键设计：**不重载模型**。reranker 和 llm 各占几秒到几十秒
（LLM warm-up 实测 33 s），换索引只换检索状态，这两个必须留着复用。
切换实测 < 1 s，因为只有 chunks 解析 + BM25 重建。
"""
from __future__ import annotations

import threading
from pathlib import Path

from insight_rag import config
from insight_rag.generate import rag
from insight_rag.retrieve import hybrid

DEFAULT_DIR = config.ROOT / "runs" / "index"
TRANSFER_DIR = config.RUNS_DIR / "transfer"

# 同一时刻只能有一次建库；上传两个文件并发建库会把显存和 I/O 都挤爆
_BUILD_LOCK = threading.Lock()
_BUSY = {"building": False, "name": ""}


def is_busy() -> dict:
    return dict(_BUSY)


def resolve(rel: str) -> Path | None:
    """把前端传来的相对路径解析成真实的索引目录。

    只允许两处：默认库 runs/index，和 runs/transfer 下的子目录。
    别的路径一律拒绝——这个参数来自 HTTP，不能让它指向任意目录。
    """
    if not rel:
        return None
    p = Path(rel)
    d = (p if p.is_absolute() else config.ROOT / p).resolve()
    if d == DEFAULT_DIR.resolve():
        return d
    t = TRANSFER_DIR.resolve()
    if t in d.parents and d.is_dir():
        return d
    return None


def current() -> Path:
    return Path(config.INDEX_DIR).resolve()


def _sync_config(d: Path) -> None:
    """让 config 里的三个路径跟着走。

    hybrid.load() 不带参数时读 config.INDEX_DIR，所以这里改了就全链路生效；
    剩下几处 `from insight_rag.config import CHUNKS_PATH` 是 by-value 导入，
    只在评测脚本里用，服务进程内不受影响。
    """
    config.INDEX_DIR = d
    config.FAISS_PATH = d / "faiss.index"
    config.CHUNKS_PATH = d / "chunks.jsonl"
    config.TOKENS_PATH = d / "tokens.json"


def activate(d: Path) -> dict:
    """切换到 d，返回切换后的健康状况。模型实例原样保留。"""
    d = Path(d).resolve()
    old = dict(rag._STATE)

    _sync_config(d)
    hybrid._STATE.clear()
    st = hybrid.load(d)

    rag._STATE.clear()
    rag._STATE.update({
        "chunks": st["chunks"],
        "hybrid_state": st,
        "reranker": old.get("reranker"),
        "llm": old.get("llm"),
        "ready": True,
    })

    # 复用不到模型就补（首次切换前服务还没被问过的情况）
    if rag._STATE["reranker"] is None:
        from insight_rag.retrieve.reranker import Reranker
        rag._STATE["reranker"] = Reranker()
        rag._STATE["reranker"].warmup()
    if rag._STATE["llm"] is None:
        from insight_rag.generate.llm import LLM
        rag._STATE["llm"] = LLM(config.LLM_GGUF)
        rag._STATE["llm"].warmup()

    return {
        "active": str(d),
        "name": d.name,
        "chunks": len(st["chunks"]),
    }


def build_from_upload(pdf_bytes: bytes, filename: str, display_name: str = "") -> dict:
    """存 PDF → 建库到 runs/transfer/<name>。产物独立，绝不覆盖 runs/index。

    目录名必须是纯 ASCII（faiss 的 C++ 层在 Windows 上打不开中文路径），
    人类可读的名字存在 meta.json 的 title 里。
    """
    from insight_rag.index import transfer

    title = (display_name or Path(filename).stem or "doc").strip()
    with _BUILD_LOCK:
        _BUSY.update({"building": True, "name": title})
        try:
            name = transfer.safe_name(title)
            out = TRANSFER_DIR / name
            if out.exists():
                # 同名就加后缀，不覆盖别人已经建好的索引
                i = 2
                while (TRANSFER_DIR / f"{name}_{i}").exists():
                    i += 1
                name = f"{name}_{i}"
                out = TRANSFER_DIR / name
            out.mkdir(parents=True, exist_ok=True)

            src = out / "source.pdf"
            src.write_bytes(pdf_bytes)

            info = transfer.build_to(src, out, verbose=False, title=title)
            info["source"] = src.name
            return info
        except Exception:
            # 半成品目录直接清掉，免得列表里出现一套用不了的索引
            import shutil

            shutil.rmtree(out, ignore_errors=True)
            raise
        finally:
            _BUSY.update({"building": False, "name": ""})


def remove(d: Path) -> bool:
    from insight_rag.index import transfer

    d = Path(d).resolve()
    if d == DEFAULT_DIR.resolve() or d == current():
        return False          # 默认库和正在用的库都不许删
    return transfer.remove_index(d)
