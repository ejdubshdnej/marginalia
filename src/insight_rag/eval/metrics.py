"""三个检索与生成质量指标。定义固定，改之前先看 docs/EVAL_PROTOCOL.md。

recall_at_k(retrieved_pages, gold_pages, k=5)
ndcg_at_k(retrieved, gold_chunks, gold_pages, k=10)
faithfulness(answer, context_chunks) -> float  # LLM 判官
"""
from __future__ import annotations

import math
import re


def page_size_map(chunks: list[dict]) -> dict[int, int]:
    """{page: 该页块数}。给 ndcg_at_k 的 page_sizes 用，避免各调用点各写一遍。

    不传 page_sizes 时，IDCG 假设同页块足够填满 k 位；若某页只有 7 个块，
    最好的检索结果也拿不到 1.0——理想排序被高估了。传了才与语料真实情况一致。
    """
    sizes: dict[int, int] = {}
    for c in chunks:
        p = c.get("page")
        if p is not None:
            sizes[p] = sizes.get(p, 0) + 1
    return sizes


def recall_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int = 5) -> float:
    """前 k 个结果里，任一块的 page ∈ gold_pages 即算命中，返回 1.0 / 0.0。

    gold_pages 为空（unanswerable 题）时应调用方跳过，这里返回 0.0 以暴露误用。
    """
    if not gold_pages:
        return 0.0
    topk = retrieved_pages[:k]
    return 1.0 if any(p in set(gold_pages) for p in topk) else 0.0


def ndcg_at_k(retrieved: list[dict], gold_chunks: list[str], gold_pages: list[int],
              k: int = 10, page_sizes: dict[int, int] | None = None) -> float:
    """三档相关性：gold_chunk=2，同页其它块=1，其余=0。

    retrieved 每项需含 chunk_id / page。返回 DCG@k / IDCG@k。

    **IDCG 由 gold 独立构造，不能用检索结果自身**（2026-11-13 修正）。
    旧实现拿 `sorted(检索回来的 gains)` 当理想排序，等价于把「本次检索的上限」
    当成「理论最优」——只检回一个同页非 gold 块时 DCG == IDCG，nDCG 恒为 1.0，
    指标失去区分度。

    理想排序（标准 IR 做法）：前 n_gold 位放 gold 块（rel=2），其后用同页块
    （rel=1）填，仍不足 k 位则补 0。

    page_sizes: 可选的 {page: 该页块数}。给了就按真实可用的同页块数封顶，
    否则假定同页块足够填满 k 位——对块数少的页会把理想排序估高，
    使完美检索也拿不到 1.0。自检：把 retrieved 换成理想排序，nDCG 必为 1.0。
    """
    gold_chunk_set = set(gold_chunks or [])
    gold_page_set = set(gold_pages or [])
    if not gold_page_set:
        return 0.0

    def rel(item: dict) -> int:
        if item.get("chunk_id") in gold_chunk_set:
            return 2
        if item.get("page") in gold_page_set:
            return 1
        return 0

    gains = [rel(it) for it in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    n_gold = min(len(gold_chunk_set), k)
    if page_sizes is None:
        n_same = k - n_gold
    else:
        # gold_chunks 必须落在 gold_pages 上（validate_qa 已强制），故此式精确
        total_on_gold_pages = sum(page_sizes.get(p, 0) for p in gold_page_set)
        n_same = max(0, total_on_gold_pages - len(gold_chunk_set))
        n_same = min(n_same, k - n_gold)
    ideal = ([2] * n_gold + [1] * n_same + [0] * k)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


_SENT = re.compile(r"[。！？；\n]+|(?<=[.!?])\s+")


def split_claims(answer: str) -> list[str]:
    """把答案切成论断（句级）。太短的片段丢弃。"""
    parts = [p.strip() for p in _SENT.split(answer) if p and p.strip()]
    return [p for p in parts if len(p) >= 6]


JUDGE_PROMPT = """下面是一段参考资料和一个论断。判断这个论断能否从参考资料中直接推断出来。

参考资料：
{context}

论断：{claim}

只回答一个词：support（能推断）或 no_support（不能推断）。不要解释。"""


def faithfulness(answer: str, context_chunks: list[dict], llm=None,
                 verbose: bool = False) -> float:
    """答案中每个论断能在检索块里找到支撑的比例。

    LLM 判官：逐条论断问本地模型 support / no_support。
    拒答类答案（"文档未提及"）没有论断，视为 1.0（无幻觉可判）。
    """
    claims = split_claims(answer)
    if not claims:
        return 1.0
    if llm is None:
        raise ValueError("faithfulness 需要 llm 判官")

    context = "\n\n".join(c["text"][:900] for c in context_chunks[:5])
    supported = 0
    for c in claims:
        p = JUDGE_PROMPT.format(context=context, claim=c)
        out = llm(p, max_tokens=8, temperature=0.0).strip().lower()
        ok = out.startswith("support")
        supported += ok
        if verbose:
            print(f"    {'✓' if ok else '✗'} {out[:12]:12s} {c[:60]}")
    return supported / len(claims)
