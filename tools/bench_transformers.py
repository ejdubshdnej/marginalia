"""对照测：同一模型走 transformers + bitsandbytes 4bit 的 tok/s。

用途：README 的生成速度表需要两条后端在同一 prompt、同一输出长度下的
可比数字（llama.cpp 88.8 tok/s vs transformers+bnb 18.4 tok/s）。
本脚本给出 transformers+bnb 路径的实测值。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

M = "models/Qwen3-4B-Instruct-2507"
PROMPT = (
    "Explain in about 150 words why reciprocal rank fusion is used instead of "
    "weighted score summation in hybrid retrieval."
)

tok = AutoTokenizer.from_pretrained(M)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(M, quantization_config=bnb, device_map="cuda")
model.eval()
print(f"load {time.time() - t0:.1f}s  vram {torch.cuda.memory_allocated()/1024**3:.2f} GB")

msgs = [
    {"role": "system", "content": "你是文档问答助手。"},
    {"role": "user", "content": PROMPT},
]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")

t0 = time.time()
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
print(f"warmup {time.time() - t0:.2f}s")

speeds = []
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = out.shape[1] - ids.shape[1]
    speeds.append(n / dt)
    print(f"run{i} {n} tok {dt:.2f}s -> {n/dt:.1f} tok/s")

print(f"best {max(speeds):.1f} tok/s  mean {sum(speeds)/len(speeds):.1f} tok/s")
