"""pytest 共享 fixture。解析只跑一次，多个测试复用（每次解析约 0.55 s）。

语料 PDF 不随仓库分发（版权原因），首次克隆后需要先跑 build_corpus.py 下载并
合并。没跑之前，依赖语料的测试会被跳过而不是报错——这样 21 个测试里有 13 个
纯函数测试仍然能跑，用来验证「改一行代码有没有弄坏指标」。
"""
from __future__ import annotations

import pytest

from insight_rag.config import CORPUS_PDF
from insight_rag.parse.pdf_parser import parse_pdf

SKIP_REASON = (
    "缺少语料 PDF：data/corpus_rag_papers.pdf。"
    "先跑 `python -X utf8 build_corpus.py` 下载并合并 14 篇论文，再跑这些测试。"
)


def _require_corpus() -> None:
    if not CORPUS_PDF.is_file():
        pytest.skip(SKIP_REASON, allow_module_level=False)


@pytest.fixture(scope="session")
def corpus_pages():
    """分栏解析后的 248 页（与生产索引同一路径）。"""
    _require_corpus()
    return parse_pdf(CORPUS_PDF, use_column_split=True)


@pytest.fixture(scope="session")
def corpus_pages_raw():
    """不分栏的 248 页，用于对照。"""
    _require_corpus()
    return parse_pdf(CORPUS_PDF, use_column_split=False)
