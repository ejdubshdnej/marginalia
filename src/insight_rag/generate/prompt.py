"""Prompt 组装与引用校验。"""
from __future__ import annotations

import re

SYSTEM = """你是文档问答助手。只依据下面给出的资料回答。
资料中没有的信息，直接回答"文档未提及"，不要推测。
每个论断后用 [编号] 标注来源。
答案控制在 120 字以内，直接给结论，不要复述资料原文，不要分点铺开。"""

USER = """资料：
{context}

问题：{question}
"""


def build_context(chunks: list[dict]) -> str:
    """上下文块带固定编号 [1] [2] [3]。"""
    return "\n\n".join(f"[{i + 1}] {c['text'].strip()}" for i, c in enumerate(chunks))


def build_prompt(question: str, chunks: list[dict]) -> str:
    return USER.format(context=build_context(chunks), question=question)


_CITE = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str) -> list[int]:
    """抽答案里出现的 [n] 编号，去重保序。"""
    seen: list[int] = []
    for m in _CITE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def validate_citations(answer: str, n_chunks: int) -> dict:
    """校验引用：编号必须落在 1..n_chunks，否则记为自造引用。"""
    cited = extract_citations(answer)
    valid = [n for n in cited if 1 <= n <= n_chunks]
    fake = [n for n in cited if not (1 <= n <= n_chunks)]
    return {
        "cited": cited,
        "valid": valid,
        "fake": fake,
        "n_fake": len(fake),
        "has_citation": bool(cited),
    }
