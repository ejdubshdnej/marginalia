"""llama.cpp 生成封装。warmup() 不是可选项：首次推理编译 CUDA kernel 实测 34 秒。"""
from __future__ import annotations

import time

from insight_rag import config


class LLM:
    def __init__(self, gguf_path=None, n_gpu_layers: int = config.N_GPU_LAYERS, n_ctx: int = 4096,
                 verbose: bool = False):
        from llama_cpp import Llama
        self.path = str(gguf_path or config.LLM_GGUF)
        self.n_gpu_layers = n_gpu_layers
        self.llm = Llama(model_path=self.path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
                         n_batch=512, verbose=verbose)
        self.warmup_seconds: float | None = None

    def warmup(self) -> float:
        """丢掉结果。不调用会让首个真实请求慢到 33 秒，P95 全错。"""
        t0 = time.time()
        self("warmup", max_tokens=8)
        self.warmup_seconds = time.time() - t0
        return self.warmup_seconds

    def __call__(self, prompt: str, max_tokens: int = config.MAX_TOKENS,
                 temperature: float = config.TEMPERATURE, system: str | None = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        r = self.chat(msgs, max_tokens=max_tokens, temperature=temperature)
        return r["choices"][0]["message"]["content"]

    def chat(self, messages: list[dict], max_tokens: int = config.MAX_TOKENS,
             temperature: float = config.TEMPERATURE) -> dict:
        """返回完整响应（含 usage），供 bench 统计 token 数。

        注意：本模型是 Qwen3-4B-Instruct-2507（非 thinking 版），其 GGUF 内的
        chat template 不含 enable_thinking 分支，传该参数会 TypeError。
        llama_cpp 0.3.36 的 create_chat_completion 也没有 chat_template_kwargs。
        """
        return self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature,
        )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    llm = LLM(config.LLM_GGUF)
    t0 = time.time()
    llm.warmup()
    print("warmup", round(time.time() - t0, 1), "s")
    t0 = time.time()
    out = llm("用一句话说明什么是 RRF。", max_tokens=64)
    dt = time.time() - t0
    print("out:", out[:120])
    print("elapsed", round(dt, 3), "s")
