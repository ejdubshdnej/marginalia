# 环境搭建

> 参考硬件：Windows 11 / RTX 5070 12 GB（Blackwell, sm_120）/ NVIDIA 驱动 596.49（CUDA 13.2）/ 32 GB RAM
> 文中数字均为实测，不是估算。

---

## 1. 跑通之后是什么状态

全链路可用：PDF 解析 → 切块 → BGE 向量化 → FAISS + BM25 双路召回 → RRF 融合 → bge-reranker 重排 → Qwen3-4B 生成（带引用编号）。

| 指标 | 实测 |
|---|---|
| 端到端延迟 | p50 762 ms / p95 1065 ms |
| 检索（含重排） | mean 95 ms / p95 114 ms |
| 生成速度 | 88.8 tok/s（llama.cpp CUDA，warm） |
| 建库耗时（248 页 → 2421 块） | 13.4 s |
| 显存峰值（三个模型同时驻留） | 4.40 GB |
| 首次调用延迟 | 31.4 s（CUDA kernel 编译，只发生一次） |

---

## 2. 安装

### 2.1 解释器

Python 3.12（本项目在 3.12.14 上验证）。用 `venv` 建环境：

```bash
python -m venv .venv
```

### 2.2 依赖

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**torch 必须单独从 PyTorch 官方源装**，PyPI 上的 torch 默认是 CPU 版：

```bash
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
```

驱动为 CUDA 12.x 时把 `cu130` 换成对应的 `cu121` / `cu124`。

### 2.3 三个容易卡住的地方

**(a) llama-cpp-python 的 CUDA wheel 不在 PyPI**

必须从它的 release 页装预编译 wheel，并且**版本要和你的 GPU 架构匹配**：cu124 / cu125 的轮子只编译到 sm_90，Blackwell（sm_120）跑不了，需要 cu130。

```bash
.venv/Scripts/python.exe -m pip install \
  "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36-cu130/llama_cpp_python-0.3.36-py3-none-win_amd64.whl"
```

前置依赖：`pip install diskcache jinja2 numpy typing-extensions`。
国内网络访问 GitHub 不畅时可用镜像前缀（例如 `https://ghproxy.net/` + 原地址），直连环境去掉即可。

**(b) `import llama_cpp` 报 `Failed to load shared library 'llama.dll'`**

`llama.dll` 依赖 torch 自带的 CUDA runtime DLL，但 Windows 的 DLL 搜索路径默认不覆盖 `site-packages`。在 `.venv/Lib/site-packages/sitecustomize.py` 里手动加进去：

```python
import os, sys
if sys.platform == "win32":
    for p in (os.path.join(os.path.dirname(__file__), "llama_cpp", "lib"),
              os.path.join(os.path.dirname(__file__), "torch", "lib")):
        if os.path.isdir(p):
            os.add_dll_directory(p)
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
```

**(c) jieba 只有 sdist，构建隔离可能失败**

`pip install jieba` 在部分受限环境下会报 `[WinError 5] 拒绝访问 while executing command installing build dependencies`——pip 的构建隔离要建一个 `mode=0o700` 的临时目录，在权限收紧的环境里这个目录建完自己都写不进去。绕开构建隔离直接用 sdist：

```python
import tarfile, shutil, os, sysconfig
with tarfile.open("jieba-0.42.1.tar.gz") as t:
    t.extractall("/tmp/jieba", filter="data")
sp = sysconfig.get_paths()["purelib"]
shutil.copytree("/tmp/jieba/jieba-0.42.1/jieba", os.path.join(sp, "jieba"))
```

---

## 3. 模型

三个模型都不随仓库分发，需要自己下载。默认从 HuggingFace 取；国内网络可设 `HF_ENDPOINT=https://hf-mirror.com`。

| 用途 | 模型 | 期望本地路径 | 大小 |
|---|---|---|---|
| 向量化 | `BAAI/bge-base-zh-v1.5` | `models/bge-base-zh-v1.5` | 391 MB |
| 重排 | `BAAI/bge-reranker-base` | `models/bge-reranker-base` | 3.2 GB（含 fp32 冗余） |
| 生成（GGUF） | `Qwen3-4B-Instruct-2507` Q4_K_M | `models/Qwen3-4B-Instruct-2507-GGUF/` | 2.4 GB |
| 生成（safetensors，备选） | 同上 fp16 | `models/Qwen3-4B-Instruct-2507/` | 7.6 GB |

路径写死在 `src/insight_rag/config.py`，想换位置改那几行即可。

**为什么生成用 GGUF 而不是 transformers 4-bit**：

| 后端 | 生成速度 | 显存 | 备注 |
|---|---|---|---|
| llama.cpp CUDA (Q4_K_M) | **88.8 tok/s** | 3.5 GB | prompt 处理也快一个量级 |
| transformers + bitsandbytes nf4 | 18.4 tok/s | 2.8 GB | Windows 上 bnb 缺优化内核，GPU 利用率只有 30% 左右 |
| transformers bf16（未量化） | 15.4 tok/s | 10.0 GB | 显存吃不消 |

> 这三个数字来自 `tools/bench_llamacpp.py` 与 `tools/bench_transformers.py`，口径见 README 的性能表。

---

## 4. 建库与自测

```bash
# 1. 下载 14 篇论文并合并成语料 PDF（约 18.8 MB）
.venv/Scripts/python.exe -X utf8 build_corpus.py

# 2. 建索引，输出到 runs/index/
PYTHONPATH=src .venv/Scripts/python.exe -X utf8 -m insight_rag.index.build

# 3. 起服务
.venv/Scripts/python.exe -X utf8 tools/launch.py
```

或者直接双击 `start.bat`——它会依次做体检、建索引（缺的话）、起服务、打开浏览器。

延迟基准：

```bash
PYTHONPATH=src .venv/Scripts/python.exe -X utf8 bench.py
```

```
stage          mean      p50      p95     (ms)
retrieve       95.4     92.1    113.5
generate      666.9    652.0    951.4
E2E           762.3    744.1   1064.8
gen avg  88.8 tok/s
```

> **首次调用有 30 秒以上的 CUDA kernel 编译开销**，必须预热，否则 p95 会被拖到 30 秒以上。服务在 `lifespan` 里调用 `rag.init()` 完成预热；`bench.py` 自己也会先跑一次空转。

---

## 5. 本项目的两个环境假设

1. **控制台编码**：Windows 默认控制台是 GBK 代码页。所有带中文输出的脚本都做了 `sys.stdout.reconfigure(encoding="utf-8")`，运行时建议加 `-X utf8`。`start.bat` 里已经带了。
2. **子进程不接管管道**：`tools/launch.py` 用 `subprocess.Popen` 启动 uvicorn 时**不**加 `capture_output=True`，让子进程直接继承控制台句柄。这样服务日志会实时出现在同一个窗口里，出问题时不用去翻日志文件。
