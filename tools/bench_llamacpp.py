"""同一 prompt、同一 128 token 输出，量 llama.cpp 的 tok/s。

与 `tools/bench_transformers.py` 配对使用：两者 prompt 与 max_new_tokens 完全一致，
所以「llama.cpp vs transformers+bnb」的倍数才是可比的，而不是拿不同基准的数硬除。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, "src")
from insight_rag import config
from insight_rag.generate.llm import LLM

PROMPT = (
    "Explain in about 150 words why reciprocal rank fusion is used instead of "
    "weighted score summation in hybrid retrieval."
)

llm = LLM(config.LLM_GGUF)
t0 = time.time()
llm("warmup", max_tokens=8, temperature=0.0)
print(f"warmup {time.time() - t0:.2f}s")

speeds = []
for i in range(3):
    t0 = time.time()
    r = llm.chat(
        [{"role": "user", "content": PROMPT}], max_tokens=128, temperature=0.0
    )
    dt = time.time() - t0
    n_tok = r["usage"]["completion_tokens"]
    speeds.append(n_tok / dt)
    print(f"run{i} {n_tok} tok {dt:.2f}s -> {n_tok/dt:.1f} tok/s")

print(f"best {max(speeds):.1f} tok/s  mean {sum(speeds)/len(speeds):.1f} tok/s")
print("preview:", r["choices"][0]["message"]["content"][:100].replace("\n", " "))
