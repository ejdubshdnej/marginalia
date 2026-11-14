# 架构说明

这份文档回答三个问题：系统由哪些部分组成、一次提问要经过哪些步骤、为什么是这么分的。

---

## 1. 数据流

```
                   ┌──────────── 离线，跑一次 ────────────┐
  data/corpus_rag_papers.pdf
        │ pymupdf 逐页取文本块
        ▼
  column_split()  双栏页按 x 坐标分左右，全宽块单列
        │
        ▼
  chunk_pages()   500 字符窗口 / 100 字符重叠 / 按页切，不跨页
        │
        ├──► embed_docs()  →  bge-base-zh-v1.5  →  768 维、L2 归一化  →  IndexFlatIP
        │                                                              runs/index/faiss.index
        ├──► tokenize()    →  jieba 分词 + 去停用词                  →  BM25Okapi
        │                                                              runs/index/tokens.json
        └──► 块元数据（chunk_id / page / paper / 字符偏移 / 文本）    →  runs/index/chunks.jsonl

                   ┌──────────── 在线，每次提问 ──────────┐
  question
        │
        ├──► embed_query()（加中文指令前缀）──► FAISS top-20 ──┐
        └──► tokenize() ──► BM25 top-20 ──────────────────────┤
                                                               ▼
                                                    RRF 融合，k=60
                                                    score = Σ 1/(k + rank + 1)
                                                               │ top-20
                                                               ▼
                                                    bge-reranker-base 交叉编码重排
                                                               │ top-5
                                                               ▼
                                                    Qwen3-4B-Instruct-2507（GGUF, Q4_K_M）
                                                    中文 system prompt，要求 ≤120 字、论断带 [n]
                                                               │
                                                               ▼
                                                    validate_citations() → {cited, valid, fake}
                                                               │
                                                               ▼
                                                    {answer, citations[], latency{}}
```

关键设计：**检索和生成完全解耦**。检索部分（解析、切块、编码、索引）不依赖大模型，可以单独评测；生成部分只吃 `chunks + query`，换生成模型不影响检索指标。评测时也是分开报的。

---

## 2. 模块职责

```
src/insight_rag/
├── config.py              所有超参和路径。其他模块不许硬编码数字，一律从这里取
├── parse/
│   ├── pdf_parser.py      双栏切分、论文清单映射（page → paper 名）
│   └── chunker.py         定长窗口切块，按页切不跨页，块 id 编码页号与页内序号
├── index/
│   ├── embedder.py        bge 向量化。文档侧不加前缀，查询侧加中文指令前缀
│   ├── vector_store.py    FAISS IndexFlatIP 的建/存/读/查
│   ├── bm25_store.py      jieba 分词 + BM25Okapi。只存 tokens，不 pickle 模型对象
│   ├── build.py           建库 CLI：解析 → 切块 → 编码 → 落盘，返回统计
│   └── transfer.py        迁移演示：任意 PDF → 独立索引目录，不碰默认索引
├── retrieve/
│   ├── hybrid.py          双路召回 + RRF 融合。state 可注入，便于一个进程挂多套索引
│   └── reranker.py        bge-reranker-base 交叉编码重排
├── generate/
│   ├── prompt.py          中文 system prompt、上下文拼装、引用校验
│   ├── llm.py             llama.cpp 封装，一次预热消除 CUDA kernel 编译开销
│   └── rag.py             把上面几步串起来，提供 ask()
├── eval/
│   ├── metrics.py         Recall@k / nDCG@k / faithfulness
│   ├── run_eval.py        20 题完整评测，输出 runs/eval_v1.json
│   └── ablation.py        四组消融（纯向量 / 纯 BM25 / RRF / +重排）
└── api/
    ├── main.py            FastAPI 路由与生命周期
    └── registry.py        索引注册表：切换、上传建库、删除，以及安全边界
```

几个值得单独说明的：

**`config.py` 里的 `INDEX_DIR` 是唯一读环境变量的地方。** 设了 `INSIGHT_RAG_INDEX_DIR` 就换索引目录，不设就和默认行为完全一致。这是迁移演示能在不覆盖 `runs/index` 的前提下换语料的机制。

**`hybrid.py` 的 `state` 参数。** `hybrid.load()` 把索引存在模块级 `_STATE` 里，一个进程只能挂一套。但 `hybrid_search(query, state=...)` 接受外部传入的 `{chunks, index, bm25}`，所以评测脚本和迁移演示可以在同一进程里挂多套索引，不需要起多个服务。

**`registry.py` 是唯一直接改写 `rag._STATE` 的地方。** 切换索引时它保留 `reranker` 和 `llm` 实例，只换检索状态——否则每次切换都要重跑 30 秒的模型预热。

---

## 3. 擦边的地方，以及为什么这么定

**为什么按页切块而不是全文切。** 块要能指回页码，跨页的块就没法给出唯一出处。代价是每页末尾会有不到 500 字符的短块，实测 2421 块里有相当一部分是这样。

**为什么查询侧加前缀、文档侧不加。** bge 系列的中文版模型是在「指令 + 句子」的配对数据上训练的，查询侧不加指令前缀会掉点。文档侧加了反而变差，因为训练时文档侧就是裸文本。

**BM25 为什么存 tokens 而不是 pickle 模型。** `BM25Okapi` 和 jieba 的分词结果都不能 pickle（tokenizer 里有 lambda），所以落盘的是分词后的 token 列表，加载时用 `build_from_tokens()` 重建。

**为什么用 RRF 而不是加权求和。** 向量分和 BM25 分的量纲完全不同，加权求和要先归一化并定一个 α。实测扫过几个 α，最优值对题目很敏感，不稳定。RRF 只用排名，不需要归一化。代价是丢掉了分数的幅度信息——这也是消融里「融合后召回反而不如纯 BM25」的原因之一（见 `docs/DESIGN_TRADEOFFS.md`）。

**为什么生成走 llama.cpp 而不是 transformers。** Windows 上 bitsandbytes 缺优化内核，4bit 路径 GPU 利用率只有 30% 左右。同一模型换 llama.cpp + Q4_K_M，速度是 88.8 tok/s vs 18.4 tok/s。代价是 GGUF 量化本身有精度损失，且 chat template 的兼容性要单独确认。

---

## 4. 服务端接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 健康检查，返回当前索引的块数、名字、路径 |
| POST | `/ask` | 提问，返回答案、引用列表、各阶段耗时 |
| POST | `/index` | 就地重建当前索引 |
| GET | `/indexes` | 列出所有可用索引及当前激活项 |
| POST | `/indexes/upload` | 上传 PDF，新建一套独立索引 |
| POST | `/indexes/switch` | 切换到另一套索引（不重载模型） |
| DELETE | `/indexes?dir=` | 删除一套迁移索引（默认库和当前库不给删） |
| GET | `/`、`/static` | 前端页面与静态资源 |

`/ask` 的返回里 `citation_check` 是 `{cited, valid, fake, n_fake, has_citation}`：`cited` 是答案里出现过的编号，`valid` 是这些编号在返回的引用列表范围内的，`fake` 是越界的。假引用数在所有评测里都是 0，但这是靠提示词约束 + 事后校验两层，不是靠模型自觉。

---

## 5. 目录结构

```
insight-rag/
├── src/insight_rag/          源码（见上）
├── tests/test_pipeline.py    21 个单元测试，不需要 GPU、不需要模型权重
├── eval/
│   ├── qa_set.jsonl          20 道标注题，四类配额
│   ├── validate_qa.py        评测集完整性校验
│   ├── ANNOTATION_SPEC.md    标注规范
│   └── transfer/             迁移演示的题目集
├── data/
│   ├── CORPUS.md             语料说明与构建方式
│   ├── papers_manifest.json  14 篇论文的清单（题名 / arXiv ID / 页码范围）
│   └── demo/                 两份虚构样例 PDF，由 tools/make_demo_pdfs.py 生成
├── docs/
│   ├── REPORT.md             技术报告（完整的方法、实验与讨论）
│   ├── EVAL_PROTOCOL.md      评测协议与指标定义
│   ├── DESIGN_TRADEOFFS.md   每个决策的备选方案与实测依据
│   ├── SETUP.md              环境搭建
│   └── ARCHITECTURE.md       本文
├── tools/                    启动器、基准、诊断、图表生成、样例文档生成
├── web/                      前端页面（原生 JS，无构建步骤）
├── runs/                     索引与评测产物（索引不进仓库）
└── start.bat                 Windows 一键启动
```

---

## 6. 一条容易被忽略的约定

`tests/` 里的 21 个测试全部不碰 GPU、不加载模型权重——解析、切块、RRF、指标计算都是纯函数，可以直接验。这样「改一行代码会不会弄坏指标」能在 1 秒内回答，不用等 30 秒的模型预热。

模型相关的行为由 `tools/` 里的诊断脚本覆盖，那些要跑真模型，慢，不进单元测试。
