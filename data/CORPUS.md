# InsightRAG — 主语料说明

## 结论

主语料是 **248 页、93.9 万字符的「RAG 技术论文集」PDF**，由 14 篇 RAG 领域核心论文按主题顺序合并而成。

选择它而不是「自己找一份文档」的理由：

1. **内容与项目强相关** —— 拿 RAG 领域的 14 篇奠基论文建索引，问的是「混合检索为什么选 RRF」这类真问题，比用一本无关说明书有说服力。
2. **评测集天然可写** —— 每篇论文的摘要、方法、实验结论都是标准答案，方便标注 gold_pages。
3. **多跳问题有素材** —— 14 篇论文互相引用（比如 Self-RAG 引用 DPR、RAGAS 引用 RAG 原始论文），可以出真·multi_hop 题。
4. **表格题有素材** —— BEIR、DPR、RAGAS 都有结果对比表。

## 构成

| # | 论文 | 主题 | 页码（0-based） | 页数 |
|---|---|---|---|---|
| 1 | Attention Is All You Need | transformer | 0–14 | 15 |
| 2 | Sentence-BERT | embedding | 15–25 | 11 |
| 3 | Dense Passage Retrieval | retrieval | 26–38 | 13 |
| 4 | RAG for Knowledge-Intensive NLP | rag | 39–57 | 19 |
| 5 | BEIR Benchmark | evaluation | 58–81 | 24 |
| 6 | HyDE: Precise Zero-Shot Dense Retrieval | retrieval | 82–92 | 11 |
| 7 | Query Rewriting for Retrieval | retrieval | 93–105 | 13 |
| 8 | Lost in the Middle | long_context | 106–123 | 18 |
| 9 | RAGAS Evaluation | evaluation | 124–131 | 8 |
| 10 | Self-RAG | rag | 132–161 | 30 |
| 11 | RAG for LLMs: A Survey | survey | 162–182 | 21 |
| 12 | Corrective RAG | rag | 183–198 | 16 |
| 13 | RAPTOR | long_context | 199–221 | 23 |
| 14 | GraphRAG: From Local to Global | rag | 222–247 | 26 |

完整清单（含 arxiv id）在 `data/papers_manifest.json`，该文件是评测时定位 gold_pages 的权威依据。

## 文本层质量（实测）

```
页数        248
总字符      939,507
中位/页     3,936
最小/页     518
最大/页     9,074
空白页      0 (0.0%)
解析耗时    0.55s
```

**零空白页**意味着不需要 OCR，`page.get_text()` 直接可用。这符合「有文本层、非扫描件」的要求。

## 重新生成

`build_corpus.py` 可完整复现（下载 → 合并 → 加书签）：

```bash
.venv/Scripts/python.exe -X utf8 build_corpus.py
```

- 单篇缓存到 `data/papers/<arxiv_id>.pdf`，已存在则跳过
- 合并后写入 PDF 书签（每篇论文一个一级书签），前端溯源可以按书签显示论文名
- 输出 `data/corpus_rag_papers.pdf`（18.8 MB）+ `data/papers_manifest.json`

> 下载走 `https://arxiv.org/pdf/{id}`，失败回退 `export.arxiv.org`。必须显式传 `ssl.create_default_context(cafile=certifi.where())`，否则会报 `CERTIFICATE_VERIFY_FAILED`。

## 为什么不用「1 篇 transformer + 2 篇综述」的方案

最初设想的是「Attention Is All You Need + 几篇 RAG 综述合成一本」，执行时扩展成 14 篇，原因：

- 单靠 1 篇 transformer + 2 篇综述，页数只有 60 页左右，规模太小，评测波动大
- 只有综述的话，问题容易停留在「综述里怎么分类」这种浅层题，出不了需要跨论文综合的 multi_hop 题
- 14 篇覆盖了 transformer / embedding / retrieval / rag / evaluation / long_context 六个主题，消融实验时不同主题的检索难度差异本身就是有价值的分析材料

