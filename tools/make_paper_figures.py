"""论文配图。SimSun（宋体）配 Times New Roman，与正文一致；300 dpi。

单独成脚本而不是塞进 ablation.py：论文图与消融图的重绘频率不同，分开更好维护。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.dpi"] = 300

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig_ablation():
    """图：四组消融的 Recall@5 与 nDCG@10。"""
    d = json.loads(Path("runs/ablation.json").read_text(encoding="utf-8"))
    names = [g["name"] for g in d["groups"]]
    recall = [g["recall_at_5"] for g in d["groups"]]
    ndcg = [g["ndcg_at_10"] for g in d["groups"]]
    short = ["A\n纯向量", "B\n纯BM25", "C\nRRF", "D\nC+重排"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    x = np.arange(len(names))
    w = 0.55

    ax = axes[0]
    bars = ax.bar(x, recall, w, color=["#8FB8DE", "#E8A87C", "#94B88F", "#C4767F"])
    ax.axhline(0.60, ls="--", lw=0.9, color="#666666")
    ax.text(3.42, 0.615, "门槛 0.60", fontsize=7, color="#666666", ha="right")
    for b, v in zip(bars, recall):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, short)
    ax.set_ylabel("页级 Recall@5")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a) 页级召回率")
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar(x, ndcg, w, color=["#8FB8DE", "#E8A87C", "#94B88F", "#C4767F"])
    ax.axhline(0.55, ls="--", lw=0.9, color="#666666")
    ax.text(3.42, 0.565, "门槛 0.55", fontsize=7, color="#666666", ha="right")
    for b, v in zip(bars, ndcg):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, short)
    ax.set_ylabel("块级 nDCG@10")
    ax.set_ylim(0, 0.66)
    ax.set_title("(b) 归一化折损累计增益")
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_ablation.png")


def fig_chunk_scan():
    """图：块大小 / 融合权重两个旋钮的敏感性（实测点连成线）。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))

    # (a) 块大小
    sizes = ["300/60", "500/100", "800/160"]
    blocks = [3996, 2421, 1536]
    r = [0.6000, 0.7333, 0.6667]
    n = [0.2003, 0.3740, 0.3078]
    xx = np.arange(3)
    ax = axes[0]
    ax.plot(xx, r, "o-", color="#C4767F", ms=4, lw=1.2, label="Recall@5")
    ax.plot(xx, n, "s--", color="#4A6FA5", ms=4, lw=1.2, label="nDCG@10")
    ax.set_xticks(xx, [f"{s}\n({b} 块)" for s, b in zip(sizes, blocks)])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("指标值")
    ax.set_title("(a) 切块粒度")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for i, (a, b) in enumerate(zip(r, n)):
        ax.annotate(f"{a:.4f}", (i, a), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color="#C4767F")
        ax.annotate(f"{b:.4f}", (i, b), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=7, color="#4A6FA5")

    # (b) 加权和 alpha
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    r2 = [0.9333, 0.8667, 0.7333, 0.6667, 0.6000]
    n2 = [0.3738, 0.3969, 0.3753, 0.3268, 0.2849]
    ax = axes[1]
    ax.plot(alphas, r2, "o-", color="#C4767F", ms=4, lw=1.2, label="Recall@5")
    ax.plot(alphas, n2, "s--", color="#4A6FA5", ms=4, lw=1.2, label="nDCG@10")
    ax.axhline(0.7333, ls=":", lw=0.9, color="#94B88F")
    ax.text(0.90, 0.745, "RRF\n0.7333", fontsize=6.5, color="#5E7A5B", ha="right", va="bottom")
    ax.set_xlabel("向量路权重 α")
    ax.set_ylabel("指标值")
    ax.set_ylim(0.2, 1.0)
    ax.set_title("(b) 加权和融合（α→0 即纯 BM25）")
    ax.legend(frameon=False, loc="center left")
    ax.grid(ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_sensitivity.png")


def fig_latency():
    """图：端到端延迟分布（两次 benchmark 的 CDF 对比）。"""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    styles = [
        ("runs/bench_20261113_050028.json", "限制 120 字（默认）", "#4A6FA5", "-"),
        ("runs/bench_20261113_045734.json", "不限长度", "#E8A87C", "--"),
    ]
    for path, label, color, ls in styles:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        vals = np.sort([r["total_ms"] for r in d["rows"]])
        ys = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, ys, ls, color=color, lw=1.3, label=label)
    ax.axvline(2500, ls=":", lw=1.0, color="#999999")
    ax.text(2500, 0.05, " 门槛 2500 ms", fontsize=6.5, color="#666666", rotation=90, va="bottom")
    ax.set_xlabel("端到端延迟（ms）")
    ax.set_ylabel("累计占比")
    ax.set_xlim(0, 3200)
    ax.set_ylim(0, 1.02)
    ax.set_title("E2E 延迟 CDF")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.grid(ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_latency.png")


def fig_pipeline():
    """图：系统架构（手绘方块）。坐标按数据单位手算，避免文字溢出色块。"""
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF0F7", ec="#4A6FA5", fs=7.0):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#555555"), zorder=1)

    # ---------- ① 离线 ----------
    ax.add_patch(plt.Rectangle((0.02, 2.80), 6.92, 1.72, fill=False, ec="#AAAAAA",
                               ls=(0, (4, 3)), lw=0.8, zorder=0))
    ax.text(0.12, 4.62, "① 离线索引构建（一次性，21.1 s）", fontsize=7.6, weight="bold", color="#333333")

    box(0.14, 3.18, 1.30, 0.80, "248 页 PDF\n93.9 万字符")
    box(1.62, 3.18, 1.30, 0.80, "分栏解析\n2 421 块")
    box(3.10, 3.62, 1.75, 0.60, "bge-base-zh 编码", fs=6.8)
    box(3.10, 3.00, 1.75, 0.60, "jieba 分词 + BM25", fs=6.8)
    box(5.03, 3.62, 1.75, 0.60, "faiss IndexFlatIP\n7.09 MB", fs=6.8)
    box(5.03, 3.00, 1.75, 0.60, "tokens.json\n重建 1.5 s", fs=6.8)

    arrow(1.44, 3.58, 1.62, 3.58)
    arrow(2.92, 3.66, 3.10, 3.90)
    arrow(2.92, 3.50, 3.10, 3.30)
    arrow(4.85, 3.92, 5.03, 3.92)
    arrow(4.85, 3.30, 5.03, 3.30)

    # 右侧说明（不占方块）
    ax.text(7.10, 4.16, "编码吞吐\n225.2 块/秒", fontsize=6.8, color="#444444",
            va="top", linespacing=1.4)
    ax.text(7.10, 3.44, "索引体积\n7.09 MB", fontsize=6.8, color="#444444",
            va="top", linespacing=1.4)
    ax.text(7.10, 2.92, "全库 21.1 s", fontsize=6.8, color="#444444", va="top")

    # ---------- ② 在线 ----------
    ax.text(0.12, 2.58, "② 在线查询（单次）", fontsize=7.6, weight="bold", color="#333333")

    ONLINE_Y, ONLINE_H = 1.24, 0.80
    xs = [(0.14, 1.28, "用户提问\n（中文界面）", "#EAF0F7", "#4A6FA5"),
          (1.62, 1.42, "双路召回\n各 top-20", "#F7EFE8", "#C4767F"),
          (3.24, 1.22, "RRF 融合\nk=60", "#F7EFE8", "#C4767F"),
          (4.66, 1.46, "CrossEncoder\n重排 → top-5", "#F7EFE8", "#C4767F"),
          (6.32, 1.52, "Qwen3-4B\nllama.cpp 生成", "#F7EFE8", "#C4767F"),
          (8.04, 1.55, "答案 + [n]\n引用可点击", "#EFF5EC", "#94B88F")]
    for x, w, t, fc, ec in xs:
        box(x, ONLINE_Y, w, ONLINE_H, t, fc=fc, ec=ec)

    for i in range(len(xs) - 1):
        x1 = xs[i][0] + xs[i][1]
        arrow(x1 + 0.02, ONLINE_Y + ONLINE_H / 2, xs[i + 1][0] - 0.02, ONLINE_Y + ONLINE_H / 2)

    # 索引产物 -> 召回
    arrow(1.90, 2.78, 1.90, 2.06)
    ax.text(2.00, 2.40, "索引产物", fontsize=6.6, color="#555555")

    ax.text(0.14, 0.52,
            "实测：faiss 单查 0.17 ms · 重排 80.1 ms · 生成 691.9 ms · E2E p95 1 064.8 ms · 自造引用 0/190",
            fontsize=6.6, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "fig_arch.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_arch.png")


if __name__ == "__main__":
    fig_ablation()
    fig_chunk_scan()
    fig_latency()
    fig_pipeline()
    print("->", OUT)
