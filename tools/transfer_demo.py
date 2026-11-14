"""迁移演示：把一个**从没见过的 PDF** 丢进来，不改一行检索/生成代码，直接建库并问答。

验证的是「这套 RAG 的迁移能力」——三个模型全是官方原版、零训练，所以迁移的唯一成本
就是换语料后重新建一次索引。这个脚本把这件事量化出来：

  建库成本（解析 / 切块 / 向量化 / 落盘秒数）
  → 检索是否找得到（top-k 块里有没有答案关键词）
  → 生成是否答得对（答案里有没有答案关键词）
  → 引用是否诚实（引用的块里到底有没有那个词；该拒答时是否拒答）

刻意**不碰生产索引**：产物写到 --out-dir（默认 runs/transfer/<name>），
生产 runs/index 一个字节都不动。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe -X utf8 tools/transfer_demo.py \
      --pdf path/to/doc.pdf --name theme --questions eval/transfer/theme_showcase.jsonl

  # 只建库不提问，纯看建库成本
  ... --pdf doc.pdf --name doc --questions ""
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_rag import config
from insight_rag.eval.run_eval import is_refusal
from insight_rag.generate import rag
from insight_rag.generate.llm import LLM
from insight_rag.index import bm25_store, embedder, vector_store
from insight_rag.parse.chunker import chunk_pages
from insight_rag.parse.pdf_parser import column_split
from insight_rag.retrieve.reranker import Reranker


# ---------------------------------------------------------------- 解析

def parse_any_pdf(path: Path) -> list[dict]:
    """解析任意 PDF，返回 [{page, text, paper}]。

    不用 parse_pdf()：它会把页码硬映射进 14 篇论文的 manifest，换文档后 paper 名会错。
    这里直接逐页调 column_split()（双栏重排），单栏文档同样正常。
    """
    import pymupdf

    doc = pymupdf.open(path)
    stem = path.stem
    out: list[dict] = []
    try:
        for pno, page in enumerate(doc):
            out.append({"page": pno, "text": column_split(page), "paper": stem})
    finally:
        doc.close()
    return out


# ---------------------------------------------------------------- 建库

def build_transfer_index(pdf: Path, out_dir: Path, verbose: bool = True) -> dict:
    """在 out_dir 建一套独立索引，返回建库报告。不写 config 里的生产路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    pages = parse_any_pdf(pdf)
    parse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_pages(pages)
    chunk_s = time.perf_counter() - t0
    texts = [c["text"] for c in chunks]

    # 首次调用会加载 BGE 模型并编译 CUDA kernel，单独计时，别混进 embed 里
    t0 = time.perf_counter()
    embedder.embed_docs(texts[:32])
    warm_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb = embedder.embed_docs(texts)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = vector_store.build(emb)
    vector_store.save(index, out_dir / "faiss.index")
    faiss_s = time.perf_counter() - t0

    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    t0 = time.perf_counter()
    bm25_store.save_tokens(texts, out_dir / "tokens.json")
    bm = bm25_store.build_from_tokens(bm25_store.load_tokens(out_dir / "tokens.json"))
    bm25_s = time.perf_counter() - t0

    total = time.perf_counter() - t_all
    faiss_mb = (out_dir / "faiss.index").stat().st_size / 1024 / 1024

    info = {
        "pdf": str(pdf),
        "pdf_mb": round(pdf.stat().st_size / 1024 / 1024, 2),
        "pages": len(pages),
        "chars": sum(len(p["text"]) for p in pages),
        "chunks": len(chunks),
        "dim": int(emb.shape[1]),
        "faiss_mb": round(faiss_mb, 2),
        "seconds": {
            "parse": round(parse_s, 2),
            "chunk": round(chunk_s, 2),
            "warmup": round(warm_s, 2),
            "embed": round(embed_s, 2),
            "faiss_save": round(faiss_s, 2),
            "bm25": round(bm25_s, 2),
            "total": round(total, 2),
        },
    }

    if verbose:
        print(f"  页数    {info['pages']}")
        print(f"  字符    {info['chars']}")
        print(f"  块数    {info['chunks']}  dim {info['dim']}")
        print(f"  faiss   {info['faiss_mb']} MB")
        print(f"  耗时    解析 {info['seconds']['parse']}s  切块 {info['seconds']['chunk']}s  "
              f"向量化 {info['seconds']['embed']}s  落盘 {info['seconds']['faiss_save']}s  "
              f"BM25 {info['seconds']['bm25']}s")
        print(f"  合计    {info['seconds']['total']}s")

    return {"info": info, "pages": pages, "chunks": chunks, "index": index, "bm25": bm}


# ---------------------------------------------------------------- 判分

def _norm(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("\n", "")


def hit(text: str, keys: list[str]) -> bool:
    """答案关键词是否出现。大小写、空白都不敏感。"""
    if not keys:
        return False
    t = _norm(text)
    return any(_norm(k) in t for k in keys)


def score_one(row: dict, result: dict) -> dict:
    """对单题打分。四项判据互相独立，谁也不代替谁。

    判分素材就是生产链路自己产出的 citations（重排后的 top-k，带全文），
    不再额外跑一遍检索——演示要展示的是真实链路，不是另一条。
    """
    answer = result["answer"]
    keys = row.get("answer_keys") or []
    top = result["citations"]

    # 1) 检索：top-k 里有没有一块真的含答案词（不看页码，页码对单页文档没意义）
    retrieval_ok = hit("\n".join(c["text"] for c in top), keys) if keys else None

    # 2) 生成：答案里有没有答案词
    answer_ok = hit(answer, keys) if keys else None
    # 宽松口径：模型答的是主题的**描述**（"Bold and modern tech aesthetic"）而不是主题名
    # （"Tech Innovation"）时，严格口径算错，但信息其实没错。两列都报，不藏。
    loose = keys + (row.get("alt_keys") or [])
    answer_ok_loose = hit(answer, loose) if keys else None

    # 3) 引用诚实：引用了 [n] 的那些块里，有没有一块含答案词
    cited_ns = result["citation_check"]["valid"]
    cited_chunks = [top[n - 1] for n in cited_ns if 1 <= n <= len(top)]
    cite_ok = hit("\n".join(c["text"] for c in cited_chunks), keys) if (keys and cited_chunks) else None

    # 4) 拒答：该拒的拒了没 / 不该拒的误拒了没
    refused = is_refusal(answer)
    if row.get("should_refuse"):
        refusal_ok = refused
    elif keys:
        refusal_ok = not refused
    else:
        refusal_ok = None

    return {
        "qid": row["qid"],
        "question": row["question"],
        "answer": answer,
        "answer_keys": keys,
        "retrieval_hit": retrieval_ok,
        "answer_hit": answer_ok,
        "answer_hit_loose": answer_ok_loose,
        "cite_hit": cite_ok,
        "refusal_ok": refusal_ok,
        "refused": refused,
        "should_refuse": bool(row.get("should_refuse")),
        "n_fake_cite": result["citation_check"]["n_fake"],
        "cited_pages": [c["page"] for c in result["citations"]],
        "cited_chunk_ids": [c["chunk_id"] for c in result["citations"]],
        "top_pages": [c["page"] for c in result["citations"]],
        "latency": result["latency"],
        "usage": result.get("usage", {}),
    }


def rate(rows: list[dict], field: str) -> dict:
    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return {"n": 0, "acc": None}
    return {"n": len(vals), "hits": sum(1 for v in vals if v), "acc": round(sum(1 for v in vals if v) / len(vals), 4)}


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="要迁移进去的 PDF")
    ap.add_argument("--name", required=True, help="本次演示的名字，也是输出目录名")
    ap.add_argument("--questions", default="", help="问题集 jsonl；留空则只建库")
    ap.add_argument("--out-dir", default="", help="输出目录，默认 runs/transfer/<name>")
    ap.add_argument("--top-k", type=int, default=config.TOP_K_RERANK)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题，调试用")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pdf = Path(args.pdf)
    if not pdf.is_absolute():
        # 先按当前工作目录解析，再退回项目根，这样从哪调用都对
        for base in (Path.cwd(), root):
            if (base / pdf).exists():
                pdf = (base / pdf).resolve()
                break
        else:
            pdf = (root / pdf).resolve()
    if not pdf.exists():
        print(f"[X] PDF 不存在：{pdf}")
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else root / "runs" / "transfer" / args.name
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    print("=" * 68)
    print(f"迁移演示 · {args.name}")
    print(f"PDF  {pdf.name}  ({pdf.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"输出 {out_dir}")
    print("=" * 68)

    print("\n[1/4] 建库（独立目录，不碰 runs/index）")
    built = build_transfer_index(pdf, out_dir)
    info = built["info"]

    rows: list[dict] = []
    qpath = None
    if args.questions:
        qpath = Path(args.questions)
        if not qpath.is_absolute():
            qpath = (root / qpath).resolve()
        if qpath.exists():
            rows = [json.loads(l) for l in qpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            print(f"  ! 问题集不存在，跳过提问：{qpath}")
    if args.limit:
        rows = rows[: args.limit]

    report: dict = {"name": args.name, "build": info, "questions_file": str(qpath) if qpath else None}

    if not rows:
        report["scored"] = 0
        _write(report, out_dir / "report.json", None, None)
        print("\n只建库，未提问。")
        return 0

    print(f"\n[2/4] 装载重排 + 生成模型（一次性，所有题共用）")
    t0 = time.perf_counter()
    rr = Reranker()
    rr.warmup()
    llm = LLM(config.LLM_GGUF)
    warm_s = llm.warmup()
    print(f"  完成，用时 {time.perf_counter() - t0:.1f}s（其中 LLM warm-up {warm_s:.1f}s）")

    # 关键：把新语料装成一个 state，直接喂给生产函数 rag.ask()。
    # 生产代码一行没改 —— 这就是「迁移」的证据。
    state = {
        "chunks": built["chunks"],
        "hybrid_state": {"chunks": built["chunks"], "index": built["index"], "bm25": built["bm25"]},
        "reranker": rr,
        "llm": llm,
        "ready": True,
    }

    print(f"\n[3/4] 逐题问答（{len(rows)} 题，top_k={args.top_k}）")
    results: list[dict] = []
    for i, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        res = rag.ask(row["question"], top_k=args.top_k, state=state)
        s = score_one(row, res)
        s["wall_s"] = round(time.perf_counter() - t0, 2)
        s["wall_s"] = round(time.perf_counter() - t0, 2)
        results.append(s)
        mark = lambda v: "-" if v is None else ("OK" if v else "NG")
        print(f"  {i:2d}. [{row['qid']}] 检索{mark(s['retrieval_hit'])} "
              f"生成{mark(s['answer_hit'])} 引用{mark(s['cite_hit'])} "
              f"拒答{mark(s['refusal_ok'])}  {s['latency']['total_ms']:.0f}ms")
        print(f"      Q: {row['question']}")
        print(f"      A: {s['answer'][:110].replace(chr(10), ' ')}")

    print(f"\n[4/4] 汇总")

    ret = rate(results, "retrieval_hit")
    ans = rate(results, "answer_hit")
    ans_loose = rate(results, "answer_hit_loose")
    cite = rate(results, "cite_hit")
    ref = rate(results, "refusal_ok")
    fake = sum(r["n_fake_cite"] for r in results)
    lat = [r["latency"]["total_ms"] for r in results]

    summary = {
        "n_questions": len(results),
        "retrieval_hit": ret,
        "answer_hit": ans,
        "answer_hit_loose": ans_loose,
        "cite_hit": cite,
        "refusal_ok": ref,
        "fake_citations": fake,
        "latency_ms": {
            "mean": round(statistics.mean(lat), 1) if lat else None,
            "p50": round(statistics.median(lat), 1) if lat else None,
            "max": round(max(lat), 1) if lat else None,
        },
    }
    report["summary"] = summary
    report["per_question"] = results

    print(f"  检索命中   {ret['hits']}/{ret['n']}   {ret['acc']}")
    print(f"  生成命中   {ans['hits']}/{ans['n']}   {ans['acc']}（宽松口径 {ans_loose['hits']}/{ans_loose['n']}）")
    print(f"  引用诚实   {cite['hits']}/{cite['n']}   {cite['acc']}")
    print(f"  拒答正确   {ref['hits']}/{ref['n']}   {ref['acc']}")
    print(f"  假引用     {fake}")
    if lat:
        print(f"  延迟       mean {summary['latency_ms']['mean']}ms  "
              f"p50 {summary['latency_ms']['p50']}ms  max {summary['latency_ms']['max']}ms")

    _write(report, out_dir / "report.json", info, summary)
    print(f"\n  报告 {out_dir / 'report.json'}")
    print(f"        {out_dir / 'report.md'}")
    return 0


def _write(report: dict, json_path: Path, info, summary) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    info = info or report["build"]
    md = [f"# 迁移演示 · {report['name']}", ""]
    md += ["## 建库成本", "",
           "| 项 | 值 |", "|---|---|",
           f"| PDF | {Path(info['pdf']).name}（{info['pdf_mb']} MB）|",
           f"| 页数 | {info['pages']} |",
           f"| 字符数 | {info['chars']} |",
           f"| 块数 | {info['chunks']} |",
           f"| 向量维度 | {info['dim']} |",
           f"| FAISS 体积 | {info['faiss_mb']} MB |",
           f"| 解析 | {info['seconds']['parse']} s |",
           f"| 切块 | {info['seconds']['chunk']} s |",
           f"| 向量化 | {info['seconds']['embed']} s |",
           f"| 落盘 | {info['seconds']['faiss_save']} s |",
           f"| BM25 | {info['seconds']['bm25']} s |",
           f"| **合计** | **{info['seconds']['total']} s** |", ""]

    if summary:
        md += ["## 问答结果", "",
               "| 指标 | 命中 | 比例 |", "|---|---|---|",
               f"| 检索命中（top-k 含答案词）| {summary['retrieval_hit']['hits']}/{summary['retrieval_hit']['n']} | {summary['retrieval_hit']['acc']} |",
               f"| 生成命中（答案含答案词）| {summary['answer_hit']['hits']}/{summary['answer_hit']['n']} | {summary['answer_hit']['acc']} |",
               f"| 生成命中·宽松（也认主题描述）| {summary.get('answer_hit_loose', {}).get('hits')}/{summary.get('answer_hit_loose', {}).get('n')} | {summary.get('answer_hit_loose', {}).get('acc')} |",
               f"| 引用诚实（引用块含答案词）| {summary['cite_hit']['hits']}/{summary['cite_hit']['n']} | {summary['cite_hit']['acc']} |",
               f"| 拒答正确 | {summary['refusal_ok']['hits']}/{summary['refusal_ok']['n']} | {summary['refusal_ok']['acc']} |",
               f"| 假引用 | {summary['fake_citations']} | — |",
               f"| 延迟 p50 / max | {summary['latency_ms']['p50']} / {summary['latency_ms']['max']} ms | — |", "",
               "## 逐题", "", "| # | 问题 | 答案 | 检索 | 生成 | 引用 |", "|---|---|---|---|---|---|"]
        for r in report["per_question"]:
            m = lambda v: "—" if v is None else ("✓" if v else "✗")
            a = r["answer"].replace("\n", " ").replace("|", "\\|")[:90]
            q = r["question"].replace("|", "\\|")[:60]
            md.append(f"| {r['qid']} | {q} | {a} | {m(r['retrieval_hit'])} | {m(r['answer_hit'])} | {m(r['cite_hit'])} |")
        md.append("")

    (json_path.parent / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    raise SystemExit(main())
