# 标注规范 · ANNOTATION_SPEC

本文件定义 `qa_set.jsonl` 的标注规则。规则在标注**之前**写死，标注过程中不改。
改规则等于改指标，改指标等于指标失效。

---

## 1. 什么算命中

**页级命中**：检索返回的前 k 个块中，任一块的 `page` ∈ `gold_pages`，即算该题召回成功。
`recall_at_k` 按此口径统计，k=5。

**块级命中**：检索返回的块中，任一块的 `chunk_id` ∈ `gold_chunks`，用于 nDCG 的相关性分档。

**为什么不用「块级」当 Recall 主指标**：同一段内容常被切在相邻块的边界上，
块级指标会把「命中邻块」判成失败，系统性低估。页级更稳。

---

## 2. gold 标到页 + 块序号

**`gold_pages` 必须写到页码，`gold_chunks` 必须写到 `p{page}_b{idx:03d}` 级别的块号。
禁止标成整篇论文。**

这不是洁癖，是算术：

| 标注粒度 | 随机抽取 5 块的命中率 |
|---|---|
| 整篇论文 | 0.310 |
| 页级 | 0.020 |

差 15 倍。若把 gold 标成整篇论文，随机基线会从 0.020 涨到 0.310，
而系统实测 Recall@5 只有 0.8 左右——**随机基线和系统指标的差距从 40 倍缩到 2.6 倍**，
指标立刻失去区分度，任何人一看就知道评测是凑的。

**标注方法**：
1. 用 `tools/find_gold.py` 按关键词在 2421 个块里搜候选
2. 人工读候选块的原文，确认它确实能回答问题
3. 记下该块的 `page` 和 `chunk_id`
4. 抄进 `qa_set.jsonl`

---

## 3. unanswerable 怎么判

**判对**：答案包含「未提及」「没有」「无法确定」「文档未提及」等明确拒答表述，且**未给出具体事实**。
**判错**：答出了具体内容（日期、数字、人名、地点），无论是否带引用。

**命题时必须先验证「语料里真的没有」**，否则测的不是幻觉抑制能力。

本项目的真实反例（发生过）：
问 `What is the capital of France`，系统答 `Paris [3]`。
追查发现 `p246_b006`（GraphRAG 论文）原文写着
*"if the question is 'What is the capital of France?', a direct answer would be 'Paris'"*。
系统答对了、引用也对，但这条题**不能当 unanswerable 用**——它考的是「语料里有没有」，不是「模型会不会编」。

**固化流程**：每条 unanswerable 题入库前，必须跑一次全语料关键词检索，确认候选答案词在 2421 个块里出现 0 次。校验脚本见 `eval/validate_qa.py`。

---

## 4. 难度分级

| 级别 | 判据 |
|---|---|
| easy | 单个块即可完整回答，答案在该块内连续出现 |
| medium | 需要 2 个块拼接，或需要跨段落对照 |
| hard | 需要跨论文（不同 PDF 篇目），或答案在表格里，或需要聚合多个数字 |

难度只用于分组分析，不参与是否达标的判定。

---

## 5. 类型配额（20 条版）

| 类型 | 条数 | 考什么 |
|---|---|---|
| factoid | 8 | 基础召回：单点事实、定义、公式 |
| multi_hop | 4 | 需 2 个块：对比、因果链、跨节推导 |
| table | 3 | 解析质量：答案只在表格里，正文无 |
| unanswerable | 5 | 幻觉抑制：语料无答案，必须拒答 |

---

## 6. 引用标注格式

字段固定，不许增删：

```json
{"qid":"q001","question":"...","gold_pages":[12,13],"gold_chunks":["p12_b003","p13_b001"],
 "gold_answer":"...","type":"factoid","difficulty":"medium"}
```

- `gold_pages`：整数列表，**全部 < 248**
- `gold_chunks`：字符串列表，格式 `p{page}_b{idx:03d}`，**必须能在 chunks.jsonl 里查到**
- `gold_answer`：一句话参考答案，不是标准答案（评测不用它做精确匹配，只用于人工判断 faithfulness）
- unanswerable 题：`gold_pages` / `gold_chunks` 填 `[]`，`gold_answer` 填 `"文档未提及"`

---

## 7. 校验

标完必须跑：

```bash
.venv/Scripts/python.exe -X utf8 eval/validate_qa.py
```

它会检查：行数、四类配额、`gold_pages` 非空且 < 248、`gold_chunks` 100% 能在 2421 块里找到、
unanswerable 题的答案词在全语料出现 0 次。任何一条不过，回上一步重标。
