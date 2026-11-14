"""发布前自检：确认这个仓库里没有不该公开的东西。

不依赖 GPU、不加载模型，纯文本扫描，几秒跑完。任何一项失败都会打印具体
文件:行号并返回非零退出码，可以直接挂进提交前检查。

    .venv/Scripts/python.exe -X utf8 tools/check_publish.py
    .venv/Scripts/python.exe -X utf8 tools/check_publish.py --strict   # 警告也算失败
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIX = {".py", ".md", ".txt", ".json", ".jsonl", ".bat", ".sh", ".yml", ".yaml",
               ".toml", ".cfg", ".ini", ".html", ".js", ".css", ".gitignore", ""}
SKIP_DIRS = {".git", ".venv", "models", "runs", "__pycache__", ".pytest_cache", "node_modules",
             ".mypy_cache", ".idea", ".vscode"}

# 内部项目语境的词。出现这些说明文字还是写给内部看的，不是写给外人看的。
BANNED_WORDS = [
    "手册", "立项", "简历", "面试", "求职", "答辩", "交付版",
    "本机", "沙箱", "受限令牌", "harness", "DSHworker", "DeepseekDHS", "office-runtime",
    "问题清单", "项目编号", "Agent执行手册",
]

# Gate / G14 / 阶段3 这类内部编号
BANNED_PATTERNS = [
    (re.compile(r"\bGate\b"), "内部验收编号 Gate"),
    (re.compile(r"\bG(?:0|1[0-7]|[3-9])\b"), "内部验收编号 G0–G17"),
    (re.compile(r"阶段\s*[0-9一二三四五六七八九十]"), "内部阶段编号"),
    (re.compile(r"\bM[0-9]{1,2}\b(?=\s*[·、:：])"), "内部里程碑编号"),
]

# 绝对路径：公开仓库里不该出现任何人的本地目录
ABS_PATH = [
    re.compile(r"[A-Za-z]:\\\\?(?:DSHworker|Users|App|DSH)"),
    re.compile(r"[A-Za-z]:/(?:DSHworker|Users/|App/)"),
    re.compile(r"/Users/[A-Za-z]"),
    re.compile(r"/home/[A-Za-z]"),
]

# 个人信息
PII = [
    (re.compile(r"1[3-9]\d{9}"), "疑似手机号"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"), "疑似邮箱（agent@ 之类占位除外）"),
    (re.compile(r"张梓锋|cyoushihou"), "个人姓名/账号"),
]

SECRETS = [
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "GitHub classic token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "私钥"),
]

ALLOWED_EMAILS = {"agent@insight-rag.local", "noreply@", "example.com", "@example"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if p.suffix.lower() in {".pdf", ".png", ".gif", ".jpg", ".jpeg", ".gguf", ".index",
                                ".docx", ".pyc", ".bin"}:
            continue
        yield p, rel


def read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def check_text(r: Report) -> None:
    for p, rel in iter_files():
        text = read_text(p)
        if text is None:
            continue
        if p.suffix == "" and p.name != ".gitignore":
            continue
        if rel.as_posix() == "tools/check_publish.py":
            continue  # 本文件里的词表本身就是这些词
        for i, line in enumerate(text.splitlines(), 1):
            for word in BANNED_WORDS:
                if word in line:
                    r.err(f"{rel}:{i} 内部用词 {word!r}")
            for pat, why in BANNED_PATTERNS:
                if pat.search(line):
                    r.err(f"{rel}:{i} {why}")
            for pat in ABS_PATH:
                if pat.search(line):
                    r.err(f"{rel}:{i} 绝对路径")
            for pat, why in PII:
                m = pat.search(line)
                if not m:
                    continue
                hit = m.group(0)
                if any(a in hit or a in line for a in ALLOWED_EMAILS):
                    continue
                r.err(f"{rel}:{i} {why}: {hit}")
            for pat, why in SECRETS:
                if pat.search(line):
                    r.err(f"{rel}:{i} 疑似凭据: {why}")


def check_python(r: Report) -> None:
    for p, rel in iter_files():
        if p.suffix != ".py":
            continue
        src = read_text(p)
        if src is None:
            r.err(f"{rel} 不是合法 UTF-8")
            continue
        try:
            ast.parse(src, filename=str(rel))
        except SyntaxError as e:
            r.err(f"{rel}:{e.lineno} 语法错误: {e.msg}")


def check_bat_ascii(r: Report) -> None:
    for p, rel in iter_files():
        if p.suffix != ".bat":
            continue
        raw = p.read_bytes()
        bad = [(i, b) for i, b in enumerate(raw) if b > 0x7F]
        if bad:
            r.err(f"{rel} 含非 ASCII 字节（第 {bad[0][0]} 字节），cmd 会按代码页解析出错")
        if b"\r\n" not in raw:
            r.err(f"{rel} 不是 CRLF 换行")
        if raw.startswith(b"\xef\xbb\xbf"):
            r.err(f"{rel} 带 UTF-8 BOM")


LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s#]+)")


def check_links(r: Report) -> None:
    for p, rel in iter_files():
        if p.suffix != ".md":
            continue
        text = read_text(p)
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for target in LINK_RE.findall(line):
                if re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
                    continue
                dest = (p.parent / target).resolve()
                try:
                    dest.relative_to(ROOT.resolve())
                except ValueError:
                    r.err(f"{rel}:{i} 链接指向仓库外: {target}")
                    continue
                if not dest.exists():
                    r.err(f"{rel}:{i} 链接目标不存在: {target}")


REQUIRED = {
    "README.md": ["## 快速开始", "## 指标", "## 复现", "LICENSE"],
    "LICENSE": ["MIT License"],
    "NOTICE": ["PyMuPDF", "AGPL"],
    ".gitattributes": ["*.py", "eol=lf"],
}


def check_required(r: Report) -> None:
    for name, needles in REQUIRED.items():
        p = ROOT / name
        if not p.is_file():
            r.err(f"缺少文件 {name}")
            continue
        text = read_text(p) or ""
        for n in needles:
            if n not in text:
                r.err(f"{name} 里找不到 {n!r}")


def check_tracked(r: Report) -> None:
    idx = ROOT / ".git" / "index"
    if not idx.is_file():
        r.warn("不是 git 仓库（或还没有提交），跳过 tracked 检查")
        return
    for name in ("data/demo/style_guide_en.pdf", "data/demo/speaker_manual_zh.pdf"):
        if not (ROOT / name).is_file():
            r.err(f"样例文档缺失 {name}")
    for name in ("runs/index/faiss.index", "models"):
        if (ROOT / name).exists():
            r.warn(f"{name} 存在于工作区——确认它在 .gitignore 里，不要提交")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="警告也算失败")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args()

    r = Report()
    check_text(r)
    check_python(r)
    check_bat_ascii(r)
    check_links(r)
    check_required(r)
    check_tracked(r)

    if args.json:
        print(json.dumps({"errors": r.errors, "warnings": r.warns}, ensure_ascii=False, indent=2))
    else:
        for e in r.errors:
            print(f"[X] {e}")
        for w in r.warns:
            print(f"[!] {w}")
        print()
        if r.errors:
            print(f"未通过：{len(r.errors)} 个错误，{len(r.warns)} 个警告")
        elif r.warns:
            print(f"通过（{len(r.warns)} 个警告）")
        else:
            print("通过：没有发现问题")

    if r.errors:
        return 1
    if args.strict and r.warns:
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    raise SystemExit(main())
