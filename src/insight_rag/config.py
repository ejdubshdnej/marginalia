"""全局超参集中地。任何模块不许硬编码数字，一律从这里取。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CORPUS_PDF   = ROOT / "data" / "corpus_rag_papers.pdf"
MANIFEST     = ROOT / "data" / "papers_manifest.json"

EMBED_MODEL  = ROOT / "models" / "bge-base-zh-v1.5"
RERANK_MODEL = ROOT / "models" / "bge-reranker-base"
LLM_GGUF     = ROOT / "models" / "Qwen3-4B-Instruct-2507-GGUF" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

CHUNK_SIZE   = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 50          # 过滤 <50 字符碎片。本实现下从未触发：循环上界 n-overlap
                            # 保证每块 >= overlap+1 = 101 字符
COLUMN_MID   = 306.0        # A4 半宽，分栏判据

TOP_K_VECTOR = 20
TOP_K_BM25   = 20
TOP_K_FUSED  = 20
TOP_K_RERANK = 5
RRF_K        = 60

MAX_TOKENS   = 512
TEMPERATURE  = 0.1
N_GPU_LAYERS = -1           # -1 = 全部放 GPU

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 产物路径
# 唯一一处环境变量：换语料演示时把索引指到别处，避免覆盖 runs/index。
# 不设这个变量时行为和以前逐字节一致（默认仍是 runs/index）。
INDEX_DIR    = Path(os.environ.get("INSIGHT_RAG_INDEX_DIR") or (ROOT / "runs" / "index"))
FAISS_PATH   = INDEX_DIR / "faiss.index"
CHUNKS_PATH  = INDEX_DIR / "chunks.jsonl"
TOKENS_PATH  = INDEX_DIR / "tokens.json"
RUNS_DIR     = ROOT / "runs"
