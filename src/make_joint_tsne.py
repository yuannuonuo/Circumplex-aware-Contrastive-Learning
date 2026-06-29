# -*- coding: utf-8 -*-
"""
make_joint_tsne.py —— 联合损失训练的 t-SNE 图（2×3：行=模型，列=Raw/中期/终态）。

数据取自联合训练就地捕获的 joint_embs.npz（键 raw/mid/final/y），不重训、不加载模型。
样式与 make_best_tsne 一致：行标=模型名、列标=阶段、每格 silhouette 角标、右侧类别图例、无主标题。
"""

import numpy as np                              # 数值
from sklearn.metrics import silhouette_score     # 量化聚类紧致度
import matplotlib                                # 绘图
matplotlib.use("Agg")                            # 无界面后端
import matplotlib.pyplot as plt                  # 画图

from tsne_plot import scatter, tsne_2d, GROUP_NAMES, GROUP_COLORS  # 复用着色/降维/组名/配色
from values import ID2LABEL, NUM_CLASSES          # 类名与类别数（图例用）

# 行 = 模型；值 = (joint_embs.npz 路径, 行标签=模型名)
ROWS = [
    ("outputs/joint_base/joint_embs.npz",  "DeBERTa-v3-base"),
    ("outputs/joint_large/joint_embs.npz", "DeBERTa-v3-large"),
]
COL_TITLES = ["Raw", "Mid-training", "Final"]    # 列标题：原始 / 中期 / 终态


def load_rows():
    """从两个 joint npz 取 raw/mid/final 表征并各做 t-SNE。"""
    out = []
    for npz, label in ROWS:                       # 逐模型
        d = np.load(npz)                          # 加载表征
        y = d["y"]                                # 标签
        embs = [d["raw"], d["mid"], d["final"]]   # 原始 / 中期 / 终态
        sils = [silhouette_score(e, y, metric="cosine") for e in embs]  # 轮廓系数
        coords = [tsne_2d(e) for e in embs]       # 各降到 2 维
        out.append((label, coords, sils, y))
        print(f"{label}: sil raw={sils[0]:.3f} mid={sils[1]:.3f} final={sils[2]:.3f}", flush=True)
    return out


def build(rows, color_by, out_path):
    """画一张 2×3 图：行=模型，列标 Raw/Mid/Final，带 silhouette 角标与类别图例。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))   # 2 行 3 列
    for r, (label, coords, sils, y) in enumerate(rows):  # 逐行(模型)
        for c in range(3):                        # 逐列(时点)
            ax = axes[r][c]
            scatter(ax, coords[c], y, color_by,    # 着色散点
                    title=(COL_TITLES[c] if r == 0 else None),       # 仅首行标时点
                    ylabel=(label if c == 0 else None))              # 仅首列标模型名
            ax.text(0.02, 0.97, f"silhouette={sils[c]:.2f}", transform=ax.transAxes,  # silhouette 角标
                    fontsize=15, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.75))

    if color_by == "class":                       # 19 类图例
        cmap = plt.get_cmap("tab20")
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(c % 20),
                              label=ID2LABEL[c], markersize=10) for c in range(NUM_CLASSES)]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5),
                   fontsize=13, ncol=1, frameon=False)
    else:                                         # 4 大组图例（Figure 1 配色）
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=GROUP_COLORS[i],
                              label=g, markersize=15) for i, g in enumerate(GROUP_NAMES)]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5),
                   fontsize=19, ncol=1, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved figure -> {out_path}", flush=True)


if __name__ == "__main__":
    rows = load_rows()                            # 取表征 + 降维
    build(rows, "class", "outputs/joint_tsne_19class.png")  # 19 类着色
    build(rows, "group", "outputs/joint_tsne_4group.png")   # 4 大组着色
    print("JOINT_TSNE_DONE")
