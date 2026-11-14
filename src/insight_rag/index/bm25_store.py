"""BM25 关键词路。jieba 分词 + BM25Okapi。

坑：BM25Okapi 和 jieba 分词结果都不能直接 pickle（tokenizer 含 lambda）。
这里存 tokens.json，每次重建（2400 块重建约 3 秒）。
"""
from __future__ import annotations

import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

# 停用词：只去掉高频虚词，保留 "为什么/指标" 这类检索有意义的词
_STOP = set("的了和是在与及对就都也很而或被把从到以为之其中这那些个么呢吧啊")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in jieba.lcut(text):
        t = t.strip()
        if not t or t in _STOP:
            continue
        toks.append(t.lower())
    return toks


def build(texts: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in texts])


def save_tokens(texts: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    toks = [tokenize(t) for t in texts]
    path.write_text(json.dumps(toks, ensure_ascii=False), encoding="utf-8")


def load_tokens(path: str | Path) -> list[list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_from_tokens(tokens: list[list[str]]) -> BM25Okapi:
    return BM25Okapi(tokens)


def search(bm25: BM25Okapi, query: str, top_k: int = 20) -> list[tuple[int, float]]:
    """返回 [(chunk_index, score), ...]，按分数降序；分数 <= 0 的丢弃。"""
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    out: list[tuple[int, float]] = []
    for i in order[:top_k]:
        s = float(scores[i])
        if s > 0:
            out.append((i, s))
    return out
