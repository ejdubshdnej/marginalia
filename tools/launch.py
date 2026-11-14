#!/usr/bin/env python
"""InsightRAG 一键启动：预检 → 缺索引则建库 → 起服务 → 等就绪 → 开浏览器。

推荐用法是双击仓库根目录的 `start.bat`（它只做两件事：切到本目录、调这个脚本）。
命令行等价写法：

    .venv\\Scripts\\python.exe -X utf8 tools\\launch.py
    .venv\\Scripts\\python.exe -X utf8 tools\\launch.py --check          # 只体检，不起服务
    .venv\\Scripts\\python.exe -X utf8 tools\\launch.py --port 8900
    .venv\\Scripts\\python.exe -X utf8 tools\\launch.py --rebuild        # 强制重建索引
    .venv\\Scripts\\python.exe -X utf8 tools\\launch.py --no-browser

设计取舍（都是踩过坑换来的）：
- **不用 subprocess 管道**：受限环境下 `os.pipe()` 可能失败，所以子进程直接继承
  控制台句柄，uvicorn 的日志原样打在同一个窗口里，失败现场不用另找日志文件。
- **不在 batch 里写中文**：cmd 按当前代码页解析 .bat，中文容易乱行；start.bat 保持
  纯 ASCII，所有中文提示从 Python 输出。
- **就绪判定看 /health 而不是看端口通**：`lifespan` 里 `rag.init()` 是阻塞的，模型
  warm-up 期间端口可能已经 listen 但请求会挂住，端口通不等于能问。
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from insight_rag import config  # noqa: E402  （只依赖 pathlib，导入瞬时完成）

DEFAULT_PORT = 8899
DEFAULT_TIMEOUT = 300.0        # 模型 warm-up 实测 31 s，留足 CPU 回退的余量
MAX_PORT_PROBES = 20           # 端口被别的程序占着时，往后找多少个

REQUIRED_IMPORTS = (
    ("fastapi", "FastAPI 服务"),
    ("uvicorn", "ASGI 服务器"),
    ("faiss", "向量检索"),
    ("torch", "CUDA 张量"),
    ("sentence_transformers", "BGE 向量 / 重排"),
    ("llama_cpp", "本地生成（GGUF）"),
    ("fitz", "PDF 解析"),
    ("rank_bm25", "BM25 检索"),
    ("jieba", "中文分词"),
)


# --------------------------------------------------------------------------- #
# 终端彩色：只在真控制台上开，重定向到文件时返回 False，避免日志里塞转义符
# --------------------------------------------------------------------------- #
def _enable_ansi() -> bool:
    if os.name != "nt":
        return sys.stdout.isatty()
    try:
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if not k32.SetConsoleMode(handle, mode.value | 0x0004):  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return False
        return True
    except Exception:
        return False


COLOR = _enable_ansi()


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def _rel(path: Path) -> str:
    """显示成相对仓库根，路径短、一眼看出是哪套索引。"""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ok(msg: str) -> None:
    print(f"  {_paint('✓', '32')} {msg}")


def warn(msg: str) -> None:
    print(f"  {_paint('!', '33')} {msg}")


def bad(msg: str) -> None:
    print(f"  {_paint('✗', '31')} {msg}")


def section(step: str, title: str) -> None:
    print()
    print(_paint(f"[{step}] {title}", "36"))


def rule() -> None:
    print(_paint("─" * 62, "90"))


# --------------------------------------------------------------------------- #
# 预检
# --------------------------------------------------------------------------- #
def check_packages() -> list[str]:
    """用 find_spec 而不是 import：torch 真导入要 2–4 s，体检不该等这么久。"""
    missing = []
    for mod, label in REQUIRED_IMPORTS:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(f"{mod}（{label}）")
    return missing


def check_models() -> list[str]:
    missing = []
    for path, label in (
        (config.EMBED_MODEL, "向量模型 bge-base-zh-v1.5"),
        (config.RERANK_MODEL, "重排模型 bge-reranker-base"),
    ):
        if not path.is_dir():
            missing.append(f"{label} → {path}")
    if not config.LLM_GGUF.is_file():
        missing.append(f"生成模型 GGUF → {config.LLM_GGUF}")
    return missing


def index_files_ok() -> bool:
    return all(p.is_file() and p.stat().st_size > 0 for p in
               (config.FAISS_PATH, config.CHUNKS_PATH, config.TOKENS_PATH))


def chunk_count() -> int:
    try:
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


# --------------------------------------------------------------------------- #
# 端口
# --------------------------------------------------------------------------- #
def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def probe_health(port: int, timeout: float = 1.5) -> dict | None:
    """端口上是不是「已经起好的 InsightRAG」。是就返回 /health，否则 None。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and data.get("status") == "ok" and "chunks" in data:
        return data
    return None


def pick_port(start: int) -> tuple[int, dict | None]:
    """返回 (端口, 该端口上已有服务的 health 或 None)。

    端口被占且对方的 /health 能应答 → 认为已经开着，直接复用；
    被别的程序占着 → 往后找一个空端口。
    """
    port = start
    for _ in range(MAX_PORT_PROBES):
        if not port_in_use(port):
            return port, None
        health = probe_health(port)
        if health is not None:
            return port, health
        port += 1
    raise SystemExit(f"从 {start} 开始连续 {MAX_PORT_PROBES} 个端口都被占用，请用 --port 指定")


# --------------------------------------------------------------------------- #
# 建库 / 起服务
# --------------------------------------------------------------------------- #
def child_env() -> dict:
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + old if old else "")
    env["PYTHONUTF8"] = "1"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 模型全在本地，只防漏网
    # 子进程是全新解释器，--index-dir 改的模块属性传不过去，靠环境变量重放。
    # config.py 里这是唯一一处读环境变量的地方，不设就等于默认 runs/index。
    if config.INDEX_DIR.resolve() != (ROOT / "runs" / "index").resolve():
        env["INSIGHT_RAG_INDEX_DIR"] = str(config.INDEX_DIR)
    return env


def build_index() -> bool:
    print(_paint("  · 建库中（首次 10–60 s），下面是子进程的输出：", "90"))
    cmd = [sys.executable, "-X", "utf8", "-m", "insight_rag.index.build"]
    rc = subprocess.call(cmd, cwd=str(ROOT), env=child_env())
    if rc != 0:
        bad(f"建库失败（exit {rc}），索引没有生成")
        return False
    ok("索引已生成")
    return True


def rebuild_via_api(port: int, timeout: float = 600.0) -> bool:
    """调已在运行的实例的 POST /index 重建索引（服务端会重新加载进程内状态）。"""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/index", data=b"{}", method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        bad(f"重建失败：{type(e).__name__}: {e}")
        return False
    ok(f"重建完成：{data.get('chunks')} 块，用时 {data.get('elapsed')}s")
    return True


def spawn_server(port: int, host: str) -> subprocess.Popen:
    cmd = [
        sys.executable, "-X", "utf8", "-m", "uvicorn", "insight_rag.api.main:app",
        "--host", host, "--port", str(port),
        "--log-level", "info", "--no-access-log",
    ]
    # 不接管 stdout/stderr：子进程继承本窗口，uvicorn 的日志和本脚本的输出交错显示，
    # 服务崩了现场就在上面，不需要再去翻 runs/*.log。
    return subprocess.Popen(cmd, cwd=str(ROOT), env=child_env())


def wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> tuple[dict | None, str | None]:
    """轮询 /health 直到就绪。

    进度提示按整行输出、每 10 s 一行，不用 `\\r` 原地刷新：uvicorn 和 transformers
    也在往同一个控制台写日志，`\\r` 行会被它们的输出直接拼成
    「… 40sINFO: Application startup complete.」这种糊在一起的烂行。
    """
    t0 = time.time()
    next_tick = 10.0
    while True:
        if proc.poll() is not None:
            return None, f"服务进程提前退出（exit {proc.returncode}），报错见上方日志"
        health = probe_health(port, timeout=1.5)
        if health is not None:
            return health, None
        elapsed = time.time() - t0
        if elapsed > timeout:
            return None, f"等了 {timeout:.0f}s 仍未就绪（模型加载或 CUDA 初始化卡住？）"
        if elapsed >= next_tick:
            print(f"  · 仍在加载（已 {elapsed:.0f}s / 上限 {timeout:.0f}s）", flush=True)
            next_tick += 10.0
        time.sleep(0.4)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            # line_buffering：stdout 被重定向到管道/文件时也要逐行出，否则进度条在跑
            # 而「就绪」那几行还压在缓冲里，看上去像卡死。
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

    ap = argparse.ArgumentParser(add_help=True, description="InsightRAG 一键启动")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"起始端口（默认 {DEFAULT_PORT}，被占用时自动顺延）")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1，局域网访问用 0.0.0.0）")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"等待就绪的秒数（默认 {DEFAULT_TIMEOUT:.0f}）")
    ap.add_argument("--rebuild", action="store_true", help="强制重建索引再启动")
    ap.add_argument("--index-dir", default="", help="换一套索引目录启动（如 runs/transfer/theme），默认 runs/index")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--check", action="store_true", help="只做体检，不启动服务")
    args = ap.parse_args(argv)

    if args.index_dir:
        # 演示用：把索引指到别处，生产 runs/index 一个字节不动。
        # 直接改 config 模块的属性，比重新导入更直白，且所有模块都是
        # `from insight_rag import config` + 调用时取属性，改了就立刻生效。
        d = Path(args.index_dir)
        if not d.is_absolute():
            d = (ROOT / d).resolve()
        config.INDEX_DIR = d
        config.FAISS_PATH = d / "faiss.index"
        config.CHUNKS_PATH = d / "chunks.jsonl"
        config.TOKENS_PATH = d / "tokens.json"

    print()
    print(_paint("  InsightRAG · 论文问答与溯源", "1;36"))
    print(_paint("  混合检索 BM25+向量 → RRF → CrossEncoder 重排 → Qwen3-4B 生成", "90"))
    rule()

    # ---- 1. 环境 ----
    section("1/4", "环境检查")
    failures: list[str] = []

    if not (ROOT / ".venv" / "Scripts" / "python.exe").is_file():
        failures.append("找不到 .venv/Scripts/python.exe —— 按 docs/SETUP.md §2 建虚拟环境")
    else:
        ok(f"解释器 {sys.version.split()[0]}  ·  .venv")

    missing_pkgs = check_packages()
    if missing_pkgs:
        for m in missing_pkgs:
            failures.append(f"缺少依赖 {m}")
    else:
        ok(f"依赖 {len(REQUIRED_IMPORTS)}/{len(REQUIRED_IMPORTS)} 全部可导入")

    missing_models = check_models()
    if missing_models:
        for m in missing_models:
            failures.append(f"缺少模型 {m}")
        failures.append("模型不入仓库，下载方式见 docs/SETUP.md §3")
    else:
        ok("模型齐全：bge-base-zh-v1.5 / bge-reranker-base / Qwen3-4B-Q4_K_M.gguf")

    # ---- 2. 索引 ----
    section("2/4", "索引")
    if args.index_dir:
        ok(f"索引目录 {_rel(config.INDEX_DIR)}（--index-dir）")
    if args.rebuild and args.index_dir:
        bad("--rebuild 只能重建默认的 runs/index；迁移用的索引请用 tools/transfer_demo.py 重建")
        return 1
    if args.rebuild:
        warn("--rebuild：强制重建")
        needs_build = True
    elif index_files_ok():
        ok(f"索引就绪：{chunk_count()} 块")
        needs_build = False
    else:
        warn(f"{_rel(config.INDEX_DIR)} 缺文件，需要建库")
        needs_build = True
        if args.index_dir:
            # 换索引目录时不能拿默认语料去建库——那等于在新目录里塞了另一份论文索引，
            # 和迁移演示想要的正好相反。让用户先把那套索引建出来。
            failures.append(
                f"{_rel(config.INDEX_DIR)} 里没有索引。先跑 tools/transfer_demo.py "
                f"--pdf <那个 PDF> --name <名字>，再用 --index-dir {_rel(config.INDEX_DIR)} 启动"
            )
        elif not config.CORPUS_PDF.is_file():
            failures.append(f"语料缺失 {config.CORPUS_PDF} —— 见 data/CORPUS.md")

    if failures:
        print()
        for f in failures:
            bad(f)
        print()
        print(_paint("  启动中止。修好上面几项后重新运行 start.bat。", "33"))
        return 1

    if args.check:
        print()
        ok("体检通过，未启动服务（去掉 --check 即可启动）")
        return 0

    # ---- 3. 端口（先定下来，重建策略取决于它）----
    section("3/4", "端口")
    port, existing = pick_port(args.port)
    if port != args.port:
        warn(f"{args.port} 被其它程序占用，改用 {port}")
    if existing is not None:
        ok(f"{port} 上已有 InsightRAG 在跑（{existing.get('chunks')} 块）")
    else:
        ok(f"端口 {port} 空闲")

    # ---- 4. 建库（按需）----
    section("4/4", "建库")
    if not needs_build:
        ok("跳过：索引已存在（要重建就加 --rebuild）")
    elif existing is not None:
        # 服务已经在跑：索引文件归它的进程管，此时在本地再写一遍只会让内存里的
        # 旧状态和磁盘对不上。走它自己的 POST /index 就地重建，行为与网页上一致。
        warn("服务已在运行，改用它的 POST /index 就地重建（实测约 18 s，期间这个实例不能提问）")
        if not rebuild_via_api(port):
            return 1
    else:
        if not build_index():
            return 1

    # 已有实例：直接复用，不再起第二个
    if existing is not None:
        url = f"http://127.0.0.1:{port}"
        print()
        rule()
        print(f"  {_paint('服务地址', '90')}  {_paint(url, '1;32')}")
        rule()
        if not args.no_browser:
            webbrowser.open(url)
            print(_paint("  已尝试打开浏览器（没弹出就手动点上面的地址）", "90"))
        return 0

    t0 = time.time()
    print()
    print(_paint("  启动服务", "36"))
    proc = spawn_server(port, args.host)
    print(f"  · 进程 PID {proc.pid}，模型加载中（含 warm-up，实测约 42 s）")

    try:
        health, err = wait_ready(port, proc, args.timeout)
        if health is None:
            print()
            bad(err or "未知原因")
            _terminate(proc)
            return 1

        ready_s = time.time() - t0
        url = f"http://127.0.0.1:{port}"
        rule()
        ok(f"就绪，用时 {ready_s:.1f}s")
        print()
        print(f"  {_paint('服务地址', '90')}  {_paint(url, '1;32')}")
        print(f"  {_paint('索引', '90')}      {health.get('chunks')} 块 · {health.get('model', '?')}")
        print(f"  {_paint('停止', '90')}      按 Ctrl+C，或直接关掉这个窗口")
        rule()

        if not args.no_browser:
            webbrowser.open(url)
            print(_paint("  已尝试打开浏览器（没弹出就手动点上面的地址）", "90"))

        # 前台守着子进程：Ctrl+C 会同时发给两个进程，uvicorn 自己会优雅退出
        proc.wait()
        print()
        ok("服务已退出")
        return 0
    except KeyboardInterrupt:
        print()
        print(_paint("  收到 Ctrl+C，正在停止服务…", "33"))
        _terminate(proc)
        ok("已停止")
        return 0


def _terminate(proc: subprocess.Popen, grace: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
