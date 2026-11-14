"""单元测试。全部可在无 GPU、无模型权重的情况下跑完。

覆盖解析、切块、融合、指标四个环节，外加几个边界用例：
  - test_parse_no_blank：解析后没有空白页
  - test_chunk_count：块数与基线口径一致
  - test_chunk_id_roundtrip：chunk_id 编解码互逆
  - test_rrf_fuse：RRF 融合的算术正确
"""
from __future__ import annotations

import pytest

from insight_rag.parse.chunker import (
    chunk_page_text,
    chunk_pages,
    make_chunk_id,
    parse_chunk_id,
)
from insight_rag.retrieve.hybrid import rrf_fuse


# ────────────────────────────── 解析 ──────────────────────────────


def test_parse_no_blank(corpus_pages):
    """248 页里没有空白页——空白页会污染向量库（零向量的余弦无意义）。"""
    assert len(corpus_pages) == 248
    blanks = [p["page"] for p in corpus_pages if not p["text"].strip()]
    assert blanks == [], f"发现空白页：{blanks[:10]}"


def test_parse_page_count_stable(corpus_pages):
    """页码连续且从 0 开始，没有重复。"""
    pages = [p["page"] for p in corpus_pages]
    assert pages == list(range(248))


def test_parse_has_text(corpus_pages):
    """每页至少有 100 字符（实测最短页也远超这个数）。"""
    short = [(p["page"], len(p["text"])) for p in corpus_pages if len(p["text"]) < 100]
    assert short == [], f"疑似解析失败的页：{short[:10]}"


# ────────────────────────────── 切块 ──────────────────────────────


def test_chunk_count(corpus_pages):
    """500/100 切块得到 2,421 块；300/60 → 3,996；800/160 → 1,536。

    这三个数与不分栏解析的 2407/3969/1523 不同，差在分栏解析补了换行符
    （总字符 939,507 → 945,423），属口径差异，不是解析错误。
    """
    counts = {}
    for size, overlap in [(500, 100), (300, 60), (800, 160)]:
        counts[size] = len(chunk_pages(corpus_pages, size=size, overlap=overlap, min_len=50))
    assert counts[500] == 2421, counts
    assert counts[300] == 3996, counts
    assert counts[800] == 1536, counts


def test_chunk_no_oversize(corpus_pages):
    """除块本身不足一个 size 的情况，任何块的字符数不得超过 size。"""
    chunks = chunk_pages(corpus_pages, size=500, overlap=100, min_len=50)
    over = [(c["chunk_id"], len(c["text"])) for c in chunks if len(c["text"]) > 500]
    assert over == [], f"超长块：{over[:5]}"


def test_chunk_min_len(corpus_pages):
    """碎片（< 50 字符）必须被丢弃。"""
    chunks = chunk_pages(corpus_pages, size=500, overlap=100, min_len=50)
    too_short = [(c["chunk_id"], len(c["text"])) for c in chunks if len(c["text"]) < 50]
    assert too_short == [], f"残留碎片：{too_short[:5]}"


def test_chunk_page_no_blank(corpus_pages):
    """每页至少产出一个块——有页没块说明这一页的文本被切块逻辑吞了。"""
    chunks = chunk_pages(corpus_pages, size=500, overlap=100, min_len=50)
    have = {c["page"] for c in chunks}
    missing = sorted(set(range(248)) - have)
    assert missing == [], f"没有块的页：{missing[:10]}"


# ───────────────────────── chunk_id 编解码 ─────────────────────────


def test_chunk_id_roundtrip():
    """make_chunk_id / parse_chunk_id 互逆，且格式固定为 p{page}_b{idx:03d}。"""
    assert make_chunk_id(0, 0) == "p0_b000"
    assert make_chunk_id(126, 7) == "p126_b007"
    assert make_chunk_id(247, 12) == "p247_b012"

    for page, idx in [(0, 0), (126, 7), (247, 12), (35, 100)]:
        cid = make_chunk_id(page, idx)
        assert parse_chunk_id(cid) == (page, idx)


def test_chunk_id_roundtrip_all(corpus_pages):
    """2,421 个真实 chunk_id 全部能解回自己的页码。"""
    chunks = chunk_pages(corpus_pages, size=500, overlap=100, min_len=50)
    for c in chunks:
        page, _ = parse_chunk_id(c["chunk_id"])
        assert page == c["page"], f"{c['chunk_id']} 解出页 {page}，实际 {c['page']}"


def test_chunk_id_bad_format():
    """非法格式必须报错，不能静默返回错误页码。"""
    with pytest.raises(ValueError):
        parse_chunk_id("b000_p1")


# ─────────────────────────────── RRF ───────────────────────────────


def test_rrf_fuse():
    """RRF 分数 = Σ 1/(k + rank + 1)，手工验算两条。"""
    # 两路都命中的块：1/61 + 1/61
    # 只有向量路命中的块：1/61；BM25 独有的块：1/61
    fused = rrf_fuse([(10, 0.9), (20, 0.8)], [(10, 12.0), (30, 9.0)], k=60, top_k=10)
    scores = dict(fused)

    assert scores[10] == pytest.approx(1 / 61 + 1 / 61)
    assert scores[20] == pytest.approx(1 / 62)
    assert scores[30] == pytest.approx(1 / 62)
    # 两路都命中的块排第一
    assert fused[0][0] == 10


def test_rrf_fuse_ordering():
    """返回结果必须按分数降序。"""
    fused = rrf_fuse([(1, 1.0), (2, 0.5)], [(3, 5.0), (1, 4.0)], k=60, top_k=10)
    vals = [s for _, s in fused]
    assert vals == sorted(vals, reverse=True)


def test_rrf_fuse_top_k():
    """top_k 截断生效。"""
    vec = [(i, 1.0) for i in range(20)]
    bm = [(i, 1.0) for i in range(20)]
    assert len(rrf_fuse(vec, bm, k=60, top_k=5)) == 5


def test_rrf_fuse_empty():
    """两路都空时返回空列表，不报错。"""
    assert rrf_fuse([], [], k=60, top_k=10) == []


def test_rrf_fuse_one_side_empty():
    """一路空时退化为另一路的排名，分数严格递减。"""
    fused = rrf_fuse([], [(5, 3.0), (6, 2.0), (7, 1.0)], k=60, top_k=10)
    assert [i for i, _ in fused] == [5, 6, 7]
    vals = [s for _, s in fused]
    assert vals == sorted(vals, reverse=True)


# ───────────────────────── 指标函数（顺带守） ─────────────────────────


def test_recall_at_k():
    from insight_rag.eval.metrics import recall_at_k

    assert recall_at_k([1, 2, 3, 4, 5], [3], 5) == 1.0
    assert recall_at_k([1, 2, 3, 4, 5], [99], 5) == 0.0
    assert recall_at_k([1, 2, 3, 4, 5, 6], [6], 5) == 0.0
    assert recall_at_k([1, 2, 3, 4, 5, 6], [6], 6) == 1.0
    assert recall_at_k([], [1], 5) == 0.0


def test_ndcg_at_k_perfect():
    from insight_rag.eval.metrics import ndcg_at_k

    # IDCG 现在按「gold 块 + 同页块填满 k 位」构造。这里 gold 页只有 1 个块，
    # 必须传 page_sizes 告诉指标该页没有更多块可填，否则理想排序被高估。
    hits = [{"chunk_id": "p1_b000", "page": 1}]
    assert ndcg_at_k(hits, ["p1_b000"], [1], 10,
                     page_sizes={1: 1}) == pytest.approx(1.0)


def test_ndcg_idcg_not_built_from_retrieved():
    """回归：IDCG 不能拿检索结果自身构造。

    旧实现 `ideal = sorted(检索回来的 gains)`：只检回一个同页非 gold 块时
    DCG == IDCG，nDCG 恒为 1.0，指标失去区分度。
    """
    from insight_rag.eval.metrics import ndcg_at_k

    only_same_page = [{"chunk_id": "p1_b999", "page": 1}]
    v = ndcg_at_k(only_same_page, ["p1_b000"], [1], 10,
                  page_sizes={1: 10})
    assert v < 0.5, f"只命中同页非 gold 块却拿到 {v:.3f}，IDCG 又用检索结果构造了"


def test_ndcg_at_k_same_page_lower():
    """同页但非 gold 块的分数必须低于 gold 块本身。"""
    from insight_rag.eval.metrics import ndcg_at_k

    gold = [{"chunk_id": "p1_b000", "page": 1}]
    same_page = [{"chunk_id": "p1_b001", "page": 1}]
    g = ndcg_at_k(gold, ["p1_b000"], [1], 10, page_sizes={1: 10})
    s = ndcg_at_k(same_page, ["p1_b000"], [1], 10, page_sizes={1: 10})
    assert g > s > 0


def test_split_claims():
    from insight_rag.eval.metrics import split_claims

    # "第三句？" 只有 3 个字符，按 split_claims 的 len >= 6 规则被丢弃——
    # 这是设计行为（太短的片段当不了论断），不是丢句。
    claims = split_claims("第一句话说完了。第二句话也很重要！第三句？")
    assert claims == ["第一句话说完了", "第二句话也很重要"]
    # 短碎片被丢弃
    assert split_claims("好。这是一句足够长的论断内容。") == ["这是一句足够长的论断内容"]


def test_faithfulness_refusal_is_one():
    """拒答类答案没有论断，按 RAGAS 定义记 1.0。"""
    from insight_rag.eval.metrics import faithfulness

    assert faithfulness("文档未提及。", [{"text": "无关内容"}], llm=None) == 1.0
