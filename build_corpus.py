"""构建主语料：下载 RAG 领域核心论文并按固定顺序合并成一本 PDF。

用法: .venv/Scripts/python.exe -X utf8 build_corpus.py
产出: data/corpus_rag_papers.pdf + data/papers_manifest.json
"""
import json
import os
import ssl
import sys
import time
import urllib.request

import certifi
import pymupdf

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "papers")
OUT = os.path.join(ROOT, "data", "corpus_rag_papers.pdf")
MANIFEST = os.path.join(ROOT, "data", "papers_manifest.json")

# (arxiv_id, 短标题, 主题分类) —— 顺序即合并后的章节顺序
PAPERS = [
    ("1706.03762", "Attention Is All You Need", "transformer"),
    ("1908.10084", "Sentence-BERT", "embedding"),
    ("2004.04906", "Dense Passage Retrieval", "retrieval"),
    ("2005.11401", "RAG for Knowledge-Intensive NLP", "rag"),
    ("2104.08663", "BEIR Benchmark", "evaluation"),
    ("2212.10496", "HyDE: Precise Zero-Shot Dense Retrieval", "retrieval"),
    ("2305.14283", "Query Rewriting for Retrieval", "retrieval"),
    ("2307.03172", "Lost in the Middle", "long_context"),
    ("2309.15217", "RAGAS Evaluation", "evaluation"),
    ("2310.11511", "Self-RAG", "rag"),
    ("2312.10997", "RAG for LLMs: A Survey", "survey"),
    ("2401.15884", "Corrective RAG", "rag"),
    ("2401.18059", "RAPTOR", "long_context"),
    ("2404.16130", "GraphRAG: From Local to Global", "rag"),
]

CTX = ssl.create_default_context(cafile=certifi.where())
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch(aid: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        print(f"  [skip] 已存在 {os.path.basename(dest)}")
        return True
    for url in (f"https://arxiv.org/pdf/{aid}", f"https://export.arxiv.org/pdf/{aid}"):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                continue
            with open(dest, "wb") as f:
                f.write(data)
            print(f"  [ok]   {aid} -> {len(data) / 1e6:.1f} MB")
            return True
        except Exception as e:
            print(f"  [fail] {url} -> {type(e).__name__}: {str(e)[:70]}")
            time.sleep(2)
    return False


def main():
    os.makedirs(RAW, exist_ok=True)
    print(f"下载 {len(PAPERS)} 篇论文到 {RAW}")
    ok, failed = [], []
    for aid, title, topic in PAPERS:
        print(f"[{title}]")
        dest = os.path.join(RAW, f"{aid}.pdf")
        (ok if fetch(aid, dest) else failed).append((aid, title, topic))
        time.sleep(1)

    if failed:
        print(f"\n警告: {len(failed)} 篇下载失败: {[f[0] for f in failed]}")

    # 合并
    print("\n合并")
    out = pymupdf.open()
    manifest = []
    for aid, title, topic in ok:
        src = pymupdf.open(os.path.join(RAW, f"{aid}.pdf"))
        start = out.page_count
        out.insert_pdf(src)
        n = out.page_count - start
        manifest.append(
            {"arxiv_id": aid, "title": title, "topic": topic,
             "page_start": start, "page_end": out.page_count - 1, "pages": n}
        )
        print(f"  {title}: p{start}-{out.page_count - 1} ({n} 页)")
        src.close()

    # 元数据 + 书签，便于前端/评测按章节定位
    out.set_metadata({"title": "RAG 技术论文集", "author": "arXiv", "subject": "RAG corpus"})
    toc = [[1, m["title"], m["page_start"] + 1] for m in manifest]
    out.set_toc(toc)
    out.save(OUT, garbage=4, deflate=True)
    total = out.page_count
    out.close()

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"total_pages": total, "papers": manifest}, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {OUT}")
    print(f"总页数: {total}  大小: {os.path.getsize(OUT) / 1e6:.1f} MB")
    print(f"清单: {MANIFEST}")
    return 0 if total >= 100 else 1


if __name__ == "__main__":
    sys.exit(main())
