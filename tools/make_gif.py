"""录一段 README 用的 GIF：提问 → 答案 → 点引用跳到出处。

不走 playwright，用 Edge 的 DevTools Protocol 直连（CDP over websocket）。
注意：不使用 subprocess(capture_output=True)，子进程 stdout/stderr 走 DEVNULL。

用法：
    .venv/Scripts/python.exe -X utf8 tools/make_gif.py
产出：
    web/demo.gif
"""
import base64
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx
import uvicorn
import websockets

from insight_rag.api.main import app

PORT = 8899
CDP_PORT = 9333
BASE = f"http://127.0.0.1:{PORT}"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = ROOT / "web" / "demo.gif"
FRAME_DIR = ROOT / "runs" / "gif_frames"

QUESTION = "How is the faithfulness score computed in the RAGAS evaluation framework?"


def serve():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def wait_server(timeout=240):
    client = httpx.Client(timeout=180.0)
    for _ in range(timeout):
        try:
            if client.get(f"{BASE}/health").status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


class CDP:
    """极简 CDP 客户端：发命令、等指定 method 的事件。"""

    def __init__(self, ws):
        self.ws = ws
        self.n = 0
        self.events = []

    async def send(self, method, params=None):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(msg))
        mid = self.n
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            if data.get("id") == mid:
                if "error" in data:
                    raise RuntimeError(f"{method}: {data['error']}")
                return data.get("result", {})
            if "method" in data:
                self.events.append(data)

    async def shot(self, path):
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        path.write_bytes(base64.b64decode(res["data"]))
        return path

    async def js(self, expr, await_promise=False):
        res = await self.send(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": await_promise},
        )
        return res.get("result", {}).get("value")


async def record(cdp, frames, label, seconds, fps=4.0):
    """在 seconds 秒内按 fps 抓帧。"""
    n = max(1, int(seconds * fps))
    gap = seconds / n
    for _ in range(n):
        t0 = time.perf_counter()
        p = FRAME_DIR / f"f{len(frames):03d}.png"
        await cdp.shot(p)
        frames.append((p, label))
        rest = gap - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


async def drive():
    from PIL import Image

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("*.png"):
        old.unlink()

    # 等 Edge 的 CDP 端口起来
    targets = None
    for _ in range(60):
        try:
            r = httpx.get(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=3.0)
            targets = r.json()
            break
        except Exception:
            time.sleep(1)
    if not targets:
        raise RuntimeError("Edge CDP 端口没起来")
    page = next((t for t in targets if t["type"] == "page"), targets[0])
    ws_url = page["webSocketDebuggerUrl"]

    frames = []
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        cdp = CDP(ws)
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        await cdp.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
        )
        await cdp.send("Page.navigate", {"url": BASE + "/"})
        time.sleep(3.5)

        # 1) 空态 + 输入问题
        await cdp.js(
            "(()=>{const q=document.getElementById('q');"
            "q.value=%s;q.dispatchEvent(new Event('input'));q.focus();})()"
            % json.dumps(QUESTION, ensure_ascii=False)
        )
        await record(cdp, frames, "type", 1.2)

        # 2) 点提问，抓「检索中」+ 答案出现
        await cdp.js("document.getElementById('go').click()")
        await record(cdp, frames, "ask", 9.0, fps=3.0)

        # 3) 点第一个引用，滚到出处
        clicked = await cdp.js(
            "(()=>{const el=document.querySelector('.cite:not(.fake)');"
            "if(!el) return null; el.click(); return el.dataset.n;})()"
        )
        print("clicked citation:", clicked)
        await record(cdp, frames, "cite", 3.0, fps=4.0)

        answer = await cdp.js("document.getElementById('answer').innerText.slice(0,200)")
        print("answer preview:", (answer or "")[:160].replace("\n", " "))

    # 拼 GIF
    imgs = []
    for p, _ in frames:
        im = Image.open(p).convert("RGB")
        im.thumbnail((1000, 625), Image.LANCZOS)
        imgs.append(im)
    if not imgs:
        raise RuntimeError("没抓到帧")

    # 每帧 250ms；开头/结尾多留一会儿
    durations = []
    for i, (_, label) in enumerate(frames):
        durations.append(300 if i == len(frames) - 1 else 250)

    for ncol in (64, 48, 32):
        q = [im.quantize(colors=ncol, method=Image.MEDIANCUT) for im in imgs]
        q[0].save(
            OUT,
            save_all=True,
            append_images=q[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
        size_mb = OUT.stat().st_size / 1024 / 1024
        print(f"colors={ncol} -> {size_mb:.2f} MB, {len(q)} frames")
        if size_mb <= 9.5:
            break
    print("GIF:", OUT, f"{OUT.stat().st_size/1024/1024:.2f} MB")


def main():
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    if not wait_server():
        print("FAIL 服务没起来")
        return 1
    print("server ready")

    # 如果已经有一个 Edge 的 CDP 端口活着，直接复用，不再另起浏览器
    global CDP_PORT
    reused = False
    for port in (9335, 9333, 9334):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/json/version", timeout=2.0)
            if r.status_code == 200:
                CDP_PORT = port
                reused = True
                print(f"reuse existing CDP on {port}: {r.json().get('Browser')}")
                break
        except Exception:
            continue

    edge = None
    if not reused:
        edge = subprocess.Popen(
            [
                EDGE,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                f"--remote-debugging-port={CDP_PORT}",
                "--remote-allow-origins=*",
                "--user-data-dir=" + str(ROOT / "runs" / "edge-profile"),
                "--window-size=1440,900",
                "--hide-scrollbars",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        import asyncio

        asyncio.run(drive())
    finally:
        if edge is not None:
            edge.terminate()
            try:
                edge.wait(timeout=10)
            except Exception:
                edge.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
