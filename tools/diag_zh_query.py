"""验证：中文提问在英文语料上到底能不能检索到。

评测集最初设计成中文题（如「混合检索的融合方式为什么选 RRF？」），
但语料是 14 篇英文论文，最终评测集改成了英文并写进 EVAL_PROTOCOL.md。
这个决定必须先被实测证伪或证实，不能靠「感觉中文不行」。

做法：把 20 条题的题干译成中文，分别用
  (a) zh 嵌入 + 中文 query（现状生产路径）
  (b) zh 嵌入 + 中文 query + 中文指令前缀
  (c) en 嵌入 + 中文 query（跨语言是否为 en 模型更强）
量页级 Recall@5，与英文题干对照。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 tools/diag_zh_query.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_rag.config import INDEX_DIR, ROOT
from insight_rag.eval.metrics import recall_at_k
from insight_rag.index import bm25_store, embedder, vector_store

ZH_QUESTIONS = {
    "q001": "RAGAS 里 faithfulness 分数是怎么算的？",
    "q002": "DPR 里的 in-batch negatives 是怎么构造训练样本的？",
    "q003": "Transformer 的点积注意力为什么要除以根号 dk？",
    "q004": "HyDE 是怎么用一个假设文档来做零样本稠密检索的？",
    "q005": "Self-RAG 的 reflection token 分哪几类，各起什么作用？",
    "q006": "Sentence-BERT 用什么网络结构让句子向量可以直接比较？",
    "q007": "Lost in the Middle 发现模型对长上下文里不同位置的信息利用有什么规律？",
    "q008": "RAPTOR 是怎么自底向上构建摘要树的？",
    "q009": "DPR 训练时负样本从哪来，和 in-batch negatives 什么关系？",
    "q010": "BEIR 用哪些指标评测，为什么用这些而不是精确率？",
    "q011": "Lost in the Middle 的现象在 RAG 综述里怎么被总结的？",
    "q012": "Self-RAG 怎么决定要不要检索，又怎么评判自己生成的质量？",
    "q013": "注意力论文的表 3 里改 head 数和改 key 维度分别对结果有什么影响？",
    "q014": "DPR 的表 2 里 top-20 和 top-100 accuracy 分别说明什么？",
    "q015": "BEIR 表里 BM25 和 DPR 在 Hole@10 上差多少？",
}


def load_qa() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "eval" / "qa_set.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["gold_pages"]]


def main() -> int:
    qa = load_qa()
    chunks = [json.loads(l) for l in (INDEX_DIR / "chunks.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = vector_store.load(INDEX_DIR / "faiss.index")
    bm = bm25_store.build_from_tokens(bm25_store.load_tokens(INDEX_DIR / "tokens.json"))

    def recall_vec(q: str, embed_fn, top: int = 10) -> float:
        hits = vector_store.search(idx, embed_fn(q), top)
        return recall_at_k([chunks[i]["page"] for i, _ in hits], [], 5)  # 占位，下面真算

    def measure(questions: list[str], embed_fn, use_bm25: bool) -> tuple[float, int]:
        recs, bm_hits = [], 0
        for r, q in zip(qa, questions):
            v = vector_store.search(idx, embed_fn(q), 20)
            if use_bm25:
                b = bm25_store.search(bm, q, 20)
                bm_hits += 1 if b else 0
                from insight_rag.retrieve.hybrid import rrf_fuse
                fused = rrf_fuse(v, b, top_k=10)
                pages = [chunks[i]["page"] for i, _ in fused]
            else:
                pages = [chunks[i]["page"] for i, _ in v[:10]]
            recs.append(recall_at_k(pages, r["gold_pages"], 5))
        return sum(recs) / len(recs), bm_hits

    en_qs = [r["question"] for r in qa]
    zh_qs = [ZH_QUESTIONS[r["qid"]] for r in qa]

    print(f"qa {len(qa)} 条\n")
    print("=" * 78)
    print("纯向量路（不用 BM25，排除分词影响）")
    print("=" * 78)
    r5, _ = measure(en_qs, embedder.embed_query, use_bm25=False)
    print(f"  en 题干 + zh 嵌入（现状）      Recall@5 {r5:.4f}")
    r5, _ = measure(zh_qs, embedder.embed_query, use_bm25=False)
    print(f"  zh 题干 + zh 嵌入（无前缀）    Recall@5 {r5:.4f}")

    # 中文指令前缀
    from insight_rag.config import QUERY_PREFIX
    def zh_prefixed(q: str):
        return embedder.embed_query(QUERY_PREFIX + q)
    r5, _ = measure(zh_qs, zh_prefixed, use_bm25=False)
    print(f"  zh 题干 + zh 嵌入（加前缀）    Recall@5 {r5:.4f}")

    # en 模型跑中文题干（跨语言）
    from sentence_transformers import SentenceTransformer
    m_en = SentenceTransformer(str(ROOT / "models" / "bge-base-en-v1.5"), device="cuda")
    en_idx_emb = m_en.encode([c["text"] for c in chunks], batch_size=64,
                             normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True).astype("float32")
    en_idx = vector_store.build(en_idx_emb)

    def en_embed(q: str):
        return m_en.encode([q], normalize_embeddings=True,
                           convert_to_numpy=True).astype("float32")[0]

    recs = []
    for r, q in zip(qa, zh_qs):
        hits = vector_store.search(en_idx, en_embed(q), 10)
        recs.append(recall_at_k([chunks[i]["page"] for i, _ in hits], r["gold_pages"], 5))
    print(f"  zh 题干 + en 嵌入（跨语言）    Recall@5 {sum(recs) / len(recs):.4f}")

    print("\n" + "=" * 78)
    print("混合检索（含 BM25）")
    print("=" * 78)
    r5, n = measure(en_qs, embedder.embed_query, use_bm25=True)
    print(f"  en 题干  en={n}/15 题有 BM25 命中   Recall@5 {r5:.4f}")
    r5, n = measure(zh_qs, embedder.embed_query, use_bm25=True)
    print(f"  zh 题干  {n}/15 题有 BM25 命中   Recall@5 {r5:.4f}")
    print("\n（BM25 命中数 = 该题 BM25 返回非空结果的条数；0 表示关键词路完全失效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
