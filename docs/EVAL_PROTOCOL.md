# 评测协议（EVAL_PROTOCOL）

本文件规定 InsightRAG 的三个指标怎么算、gold 标到什么粒度、判官用什么 prompt、以及任何人拿到这个仓库后怎么把论文里的数字重算出来。

**原则**：README 和论文里出现的每一个指标数字，都必须能用本文件末尾的命令重新跑出来。跑不出来的一律不写。

---

## 1. 评测集

`eval/qa_set.jsonl`，20 条，每行一个 JSON 对象：

```json
{"qid":"q001","question":"...","gold_pages":[126],"gold_chunks":["p126_b007","p126_b004"],
 "gold_answer":"...","type":"factoid","difficulty":"easy"}
```

| 字段 | 含义 |
|---|---|
| `qid` | 题号，`q001`–`q020` |
| `question` | 英文问句（原因见 §1.2） |
| `gold_pages` | 能回答本题的**页号**列表（0-based，0–247）。unanswerable 题为空列表 |
| `gold_chunks` | 上述页面里**直接包含答案**的块 id。unanswerable 题为空列表 |
| `gold_answer` | 参考答案，用于人读和抽查，**不参与指标计算** |
| `type` | `factoid` / `multi_hop` / `table` / `unanswerable` |
| `difficulty` | `easy` / `medium` / `hard` |

配额：factoid 8 / multi_hop 4 / table 3 / unanswerable 5。

> **`gold_pages` 的一条约定**：一般要求全部非空，但
> 20 条里有 5 条 unanswerable。unanswerable 的语义就是「语料里没有答案」，其 `gold_pages`
> **必然是空列表**——若填成非空，这道题就不再是 unanswerable，拒答率也无从测得。
> 本项目的处理：**非 unanswerable 的 15 条 `gold_pages` 100% 非空**（实测 15/15），
> unanswerable 的 5 条为空。这条冲突已写进 `eval/validate_qa.py` 第 5 条校验：
> 非 unanswerable 为空 → 报错；unanswerable 非空 → 报错。实测通过。

### 1.1 语料

`data/papers.pdf`，248 页，14 篇 RAG 领域英文论文合并成的一本 PDF，939,507 字符，0 空白页。

| 页码范围 | 论文 |
|---|---|
| 0–14 | Attention Is All You Need |
| 15–25 | Sentence-BERT |
| 26–38 | DPR |
| 39–57 | RAG |
| 58–81 | BEIR |
| 82–92 | HyDE |
| 93–105 | Query Rewriting |
| 106–123 | Lost in the Middle |
| 124–131 | RAGAS |
| 132–161 | Self-RAG |
| 162–182 | RAG Survey |
| 183–198 | Corrective RAG |
| 199–221 | RAPTOR |
| 222–247 | GraphRAG |

### 1.2 为什么题目是英文

语料是英文。本项目的嵌入模型是 `bge-base-zh-v1.5`（中文模型），BM25 用 jieba 分词——两者都为中文查询设计。这会带来多大损失，本项目实测过，结论比「中文不行」要细：

| 题干 | 纯向量 Recall@5 | 混合（含 BM25）Recall@5 | BM25 空返回 |
|---|---|---|---|
| 英文（本评测集） | 0.6000 | 0.7333 | 0/15 |
| 中文，**保留英文专有名词** | 0.5333 | **0.7333** | 0/15 |
| 中文，**纯中文无英文术语** | **0.2667** | **0.2667** | **15/15** |
| 中文题干 + 英文嵌入模型（跨语言） | 0.4000 | — | — |

关键在中间那两行：**中文提问能不能检索到，取决于题干里有没有英文专有名词**。问「RAGAS 的 faithfulness 怎么算」时，jieba 会把 `RAGAS`、`faithfulness` 切成英文词，BM25 照常命中，指标与英文题干持平（0.7333）；一旦改成全中文（「忠实度分数怎么算」），BM25 对 2,421 个英文块**全部空返回**，融合退化成纯向量，Recall@5 掉到 **0.2667**——只剩英文题干的 36%。

另外两条实测：

- **换英文嵌入模型不能救跨语言**：`bge-base-en-v1.5` 跑中文题干只有 **0.4000**，比中文模型（0.5333）更差。跨语言检索不是换个模型就能解决的。
- **重排同样受影响**：`bge-reranker-base` 是中文重排模型。同一条英文段落，英文 query 打 **+0.5291**，中文 query 打 **+0.0003～+0.0021**，差两个数量级。

所以评测集必须用英文。不是因为中文一定失败，而是因为**中文的成败取决于题干的用词习惯**——同一道题换个说法指标就变，这种评测集不可复现。中文仍可在前端作为可选输入，但它的指标不进任何报告（见 `docs/DESIGN_TRADEOFFS.md` 第 3 条）。

---

## 2. 三个指标的定义

记检索返回的前 k 个块为 $R_k$，某题标注的 `gold_pages` 为 $G_p$，`gold_chunks` 为 $G_c$。

### 2.1 Recall@5（页级，主指标）

$$
\text{Recall@5} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[\exists\, r \in R_5^{(i)},\ \text{page}(r) \in G_p^{(i)}\big]
$$

前 5 个结果里任一块的页号落在 gold 页集合内，该题记 1，否则记 0。$N$ = 15（有 gold 的题数，unanswerable 题不参与检索指标）。

实现：`src/insight_rag/eval/metrics.py::recall_at_k(retrieved_pages, gold_pages, k=5)`。

**为什么页级而不是块级**：块级 gold 极难标准——同一页里往往有两三个块都能支撑答案，只标其中一个会让指标偏低且不可复现。页级是「找对了地方没有」，与产品体验对应（用户点开就能看到答案），且对标注者的主观判断不敏感。

### 2.2 nDCG@10（块级，排序质量）

三档相关性：

| 情况 | $rel$ |
|---|---|
| 块 id ∈ `gold_chunks` | 2 |
| 页号 ∈ `gold_pages` 但块不在 `gold_chunks` 里 | 1 |
| 其余 | 0 |

$$
\text{DCG@}10 = \sum_{i=1}^{10} \frac{rel_i}{\log_2(i+1)},\qquad
\text{nDCG@}10 = \frac{\text{DCG@}10}{\text{IDCG@}10}
$$

其中 IDCG 是**独立于检索结果**构造的理想排序——这一点必须写死，否则指标会自证：

$$
\text{IDCG@}10 = \sum_{i=1}^{10}\frac{rel^*_i}{\log_2(i+1)},\qquad
rel^* = \underbrace{[2,\dots,2]}_{n_{\text{gold 块}}} \oplus
\underbrace{[1,\dots,1]}_{n_{\text{同页块}}} \oplus [0,\dots]
$$

$n_{\text{同页块}}$ 由语料真实情况决定：该页总块数减去其中属于 `gold_chunks` 的块数，且不超过 $10-n_{\text{gold 块}}$（见 `metrics.py::page_size_map`）。

> **修过一个 bug（必须记录）**：初版实现写成 `ideal = sorted(本次检索回来的 gains, reverse=True)`，即拿检索结果自身构造 IDCG。后果是「只检回一个同页非 gold 块」时 DCG 与 IDCG 相等，nDCG 恒为 **1.0**，指标完全没有区分度。这直接让 nDCG 虚高到 0.6501（真值为 0.3740）。修正后四组消融的 nDCG 全部下降约 0.27。
>
> 为什么这么容易写错：三档相关的 IDCG 不像二值相关那样只有「gold 全在前」一种理想排序，必须显式指定「理想排序里同页块能占几位」，而这一位要问语料（每页切成几块）而不是问检索结果。

实现：`metrics.py::ndcg_at_k(retrieved, gold_chunks, gold_pages, k=10, page_sizes=None)`。
`page_sizes` 是 `{page: 该页块数}`，由 `page_size_map(chunks)` 生成；不传时退化为「同页块足够填满 k 位」的乐观假设，会让分数略高（本项目实测 +0.011~+0.014）。

**为什么三档**：只用「是否 gold 块」会丢掉「同一页的相邻块」这个明显优于无关块的信号；三档让同页块拿到部分分数，同时 gold 块始终排在它前面。$k=10$ 而 Recall 用 5，是因为排序质量在前 10 位仍有区分度，而前 5 已被 Recall 覆盖。

**这个指标的隐含要求**（修完 bug 才看清）：一页平均切 9.8 块，所以 IDCG 里「同页块」能占满剩余 9 位。要拿高分，top-10 必须**密集落在 gold 页**，而块级检索器天然把不同页的块混排（这正是 RAG 想要的多样性）。实测页精度只有 0.23–0.27，即 top-10 里平均 1.2–1.5 块在 gold 页，nDCG 因此停在 0.37。这不是检索器找不到（块级 Recall@5 有 0.80），是**指标奖励「页内密集」而检索器被设计成「跨页分散」**。详见 `DESIGN_TRADEOFFS.md` 第 10 条。

### 2.3 Faithfulness（生成忠实度，LLM 判官）

$$
\text{Faithfulness} = \frac{\text{被支撑的论断数}}{\text{论断总数}}
$$

流程：

1. 把答案按句切分（`metrics.py::split_claims`，正则 `[。！？；\n]+|(?<=[.!?])\s+`，丢弃长度 < 6 的碎片）；
2. 对每条论断，取前 5 个检索块各前 900 字符拼成参考资料；
3. 用本地 Qwen3-4B 当判官，逐条问 `support` / `no_support`；
4. 统计 `support` 的比例。

**拒答类答案**（「文档未提及」这类）切不出论断，按定义 **Faithfulness = 1.0**——没有论断就没有可证伪的幻觉。这是 RAGAS 原定义（claims 为空时定义为 1）的直译，不是本项目发明的宽松口径。

实现：`metrics.py::faithfulness(answer, context_chunks, llm, verbose=False)`。

### 2.4 判官 prompt 全文

以下是 `metrics.py::JUDGE_PROMPT` 的**逐字**内容，任何改动都会让历史数字失效：

```
下面是一段参考资料和一个论断。判断这个论断能否从参考资料中直接推断出来。

参考资料：
{context}

论断：{claim}

只回答一个词：support（能推断）或 no_support（不能推断）。不要解释。
```

判官调用参数：`max_tokens=8`、`temperature=0.0`，取输出首词判断 `out.startswith("support")`。

**判官的已知偏差**：本地 4B 模型做判官偏宽松，倾向于把「部分相关」判成 support。因此本项目 Faithfulness 的绝对值（0.90）不能与论文里用 GPT-4 当判官的数字直接比较；只有同一判官下的**组间对比**有意义（即消融里 D 组 vs C 组）。

### 2.5 拒答率（unanswerable）

对 5 条 unanswerable 题，答案命中拒答模式且长度 ≤ 120 字符即算正确：

```
"文档未提及" / "未提及" / "not mentioned" / "no information"
```

实现：`run_eval.py::is_refusal`。

**这套判据的局限（必须写进论文）**：这 5 条题的答案词在 2,421 个块里出现 **0 次**（构建时用 `eval/validate_qa.py` 的 `ABSENT_TERMS` 强制校验：mercury / boiling point / vitamin d / beethoven / symphony / carbon-14 / half-life 等）。模型说「文档未提及」是必然的，**测的是语料边界，不是幻觉抑制**。真正困难的幻觉场景（语料有相关信息但不足以回答）无法客观判定，一条都没有出。

---

## 3. gold 标注规则

这是整套评测里最容易做错、做错后最不可挽回的一步。

### 3.1 规则一：gold 标到「页」，不标到「整篇论文」

同一题如果把相关论文的整段页码范围都标 gold，随机基线会从 **0.020** 涨到 **0.310**（差 15 倍），指标彻底失效——因为随便检索 5 页都可能命中。

### 3.2 规则二：只标「不看它就答不出」的页

判据：把该页从这个 20 条题的语料里删掉，这道题还能不能答对。不能，才是 gold。

**这条规则是本项目实测踩出来的**：首版 qa_set 里 15 道题共标了 23 个 gold 页（1.53 页/题），随机基线 **0.0413**，超出容许区间 [0.015, 0.030] 37%。

根因是**把「相关」当成了「回答本题」**：

| 题 | 误标的页 | 真实情况 |
|---|---|---|
| q007 | p106 + p171 | p171 是 RAG Survey 里复述同一结论的一页，只有 p106 是原始论文 |
| q004 | p82 + p83 | p82 只是摘要，真正讲 hypothetical document 的是 p83 |
| q010 | p62 + p66 | p66 只讲 TREC-COVID 复现方法，不回答「BEIR 用什么指标」 |

修正后压到 17 页（1.13 页/题），理论随机值 $5 \times 1.13 / 248 \approx 0.0228$，实测 0.0267 落入区间。

**代价**：Recall@5 从 0.8000 降到 0.7333。这是题目变难导致的真实下降，**不是模型退化**，两组数字不能混用。

### 3.3 规则三：unanswerable 题的答案词必须在全语料出现 0 次

不能凭直觉判断「这个话题论文里应该没有」。本项目实测的反例：问「2022 世界杯冠军是谁」，直觉上 14 篇 RAG 论文不可能讲足球——但 `world cup` 在语料里出现 1 次、`fifa` 出现 2 次，因为 GraphRAG 论文拿世界杯当实体消歧的例子。这道题最终被 `validate_qa.py` 拦下并替换。

**做法**：先写答案词，再跑全语料计数，确认 0 次才允许入集。

另一个反例（来自 API 冒烟测试）：问「法国首都」，模型答 `Paris [3]`。看起来是幻觉，实际是对的——第 3 个块的原文写着「if the question is 'What is the capital of France?', a direct answer would be 'Paris'」。**RAG 系统忠于原文，不等于忠于世界知识**，这类题不能当幻觉证据。

### 3.4 规则四：gold_chunks 必须是真实存在的块 id

格式 `p{页号}_b{页内序号:03d}`，例如 `p126_b007` = 第 126 页第 8 个块。校验器会把它拿去 2,421 个块里查，查不到就报错并列出 id。

---

## 4. 随机基线

衡量「gold 标得够不够细」的自检指标。

做法：对每道有 gold 的题，从 2,421 个块里**均匀随机抽 5 块**，看是否命中 gold 页；重复 100 次取均值。随机种子固定 `20261113`，保证任何人复跑得到同一个数。

理论值：$5 \times \bar{|G_p|} / 248$，其中 $\bar{|G_p|}$ 是平均每题 gold 页数。

| 平均 gold 页/题 | 理论随机基线 | 判定 |
|---|---|---|
| 1.00 | 0.0202 | ✅ 在区间内 |
| 1.13（本项目） | 0.0228 | ✅ 实测 0.0267 |
| 1.53（首版误标） | 0.0308 | ❌ 实测 0.0413 |

区间 **[0.015, 0.030]**。低于 0.015 说明 gold 标得过窄（等于变相调难），高于 0.030 说明标得过粗（Recall 会虚高）。

---

## 5. 门槛

| 指标 | 门槛 | 实测 | 状态 |
|---|---|---|---|
| 页级 Recall@5 | ≥ 0.60 | 0.7333 | ✅ |
| 块级 nDCG@10 | ≥ 0.55 | **0.3740** | ❌ |
| Faithfulness | ≥ 0.70 | 0.9000 | ✅ |
| unanswerable 拒答率 | ≥ 0.60 | 1.0000 (5/5) | ✅ |
| 随机基线 | 0.015–0.030 | 0.0267 | ✅ |

**nDCG 这一项不达标，如实记录，不调参凑。** 允许的解释是「说清楚是检索问题还是生成问题」，实测结论是**两者都不是**：块级 Recall@10 有 0.80（块找得到），Faithfulness 0.90（生成忠实），卡住的是「三档相关 + 页内密集 IDCG」这个指标与「跨页分散」的检索器之间的结构性错配。四种改法都试过，没有一种能同时保住 Recall：

| 策略 | Recall@5 | nDCG@10 | 页精度 |
|---|---|---|---|
| 块级 RRF（现状） | 0.7333 | 0.3740 | 0.27 |
| 每页最多 4 块 | 0.7333 | 0.3487 | 0.24 |
| 每页最多 1 块 | 0.8000 | 0.1952 | 0.11 |
| 页优先再摊平（small-to-big） | **0.5333** | **0.4757** | 0.53 |
| oracle：页全对、页内按真实块分排 | 1.0 | 0.9347 | 1.0 |

`0.9347` 说明门槛**数学上可达**，但只有「页级召回全对」才拿得到；当前页级 Recall@1 仅 0.5333。追 nDCG 的唯一通路是换更强的页级排序（更大模型、交叉编码器），不在本项目的算力与时间预算内。

注意这里的门槛与「在更大标注集上得到的历史值」不是一回事：后者需要 150 条量级的标注，本项目的 20 条评测集难度更高、样本更小，因此门槛定在 0.60 / 0.55。**两者不可互相冒充。**
另外，**外部基线里没有 nDCG 的任何数字**——0.55 与「目标 0.75」都没有历史值可对照，IDCG 的构造方式也没有统一约定。这意味着 0.55 更接近一个拍出来的门槛，而不是从可复现的度量约定里推出来的。

---

## 6. 复现命令

以下命令均在仓库根目录 `insight-rag/` 下、Windows + Bash 环境执行。

### 6.1 建索引（首次或语料变更后）

```bash
PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.index.build_index
```

产出 `runs/index/{faiss.index, chunks.jsonl, tokens.json}`。

### 6.2 校验评测集

```bash
PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.eval.validate_qa
```

期望输出：`✅ 校验通过：20 行、四类配额对、gold_pages 合法、gold_chunks 100% 命中、unanswerable 答案词 0 次`

### 6.3 跑评测

```bash
PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.eval.run_eval \
    --qa eval/qa_set.jsonl --out runs/eval_v1.json
```

产出 `runs/eval_v1.json`，含每题明细（`per_question`）与汇总（`summary`）。控制台输出汇总表。

单题耗时参考：检索 53 ms、端到端 1054 ms（含 faithfulness 判官约 230 ms），模型就绪 42.6 s。

### 6.4 跑消融

```bash
PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.eval.ablation \
    --qa eval/qa_set.jsonl --out runs/ablation.json --png runs/ablation.png
```

产出 `runs/ablation.json` + `runs/ablation.png`（300 dpi）。

### 6.5 跑测试

```bash
.venv/Scripts/python.exe -X utf8 -m pytest -q
```

---

## 7. 复现性说明

| 项 | 状态 |
|---|---|
| 随机种子 | `random_baseline` 固定 `20261113`；BM25/FAISS 无随机性 |
| 嵌入模型 | 本地 `models/bge-base-zh-v1.5`，不联网下载 |
| 重排模型 | 本地 `models/bge-reranker-base` |
| 生成模型 | 本地 `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`，`temperature=0.1` |
| GPU | 单一 RTX 5070 12GB；嵌入/重排/LLM 同时驻留 |
| 判官温度 | `0.0` |

**不完全可复现的部分**：LLM 判官在 `temperature=0.0` 下仍可能有极少量波动（llama.cpp 的采样实现与显卡非确定性），Faithfulness 在 ±0.05 内浮动属正常。所有数字以 `runs/eval_v1.json` 里记录的那一次为准。
