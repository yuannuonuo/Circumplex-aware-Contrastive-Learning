# -*- coding: utf-8 -*-
"""
make_best_tsne.py —— 用保留下来的"最佳配置"表征，画两张 2×3 t-SNE 对比图。

数据取自各两阶段模型的 outputs/twostage_*/embs.npz（训练时就地捕获的表征），不重训、不加载 .pt。
布局（每张图 2 行 × 3 列）：
  行：两个模型（用模型名作行标签）
  列：原始预训练 / stage1 后 / stage2 后（顺序固定，不在图上标注）
两张图分别按 ①19 类 与 ②4 个高阶大组 着色，并各带类别图例。
无标题、无 s1/s2/阶段标注。
"""

import numpy as np                              # 数值
from sklearn.metrics import silhouette_score     # 量化聚类紧致度
import matplotlib                                # 绘图
matplotlib.use("Agg")                            # 无界面后端
import matplotlib.pyplot as plt                  # 画图

from tsne_plot import scatter, tsne_2d, GROUP_NAMES, GROUP_COLORS  # 复用着色/降维/组名/配色
from values import ID2LABEL, NUM_CLASSES          # 类名与类别数（图例用）

# 行 = 模型；值 = (npz路径, stage1键, stage2键, 行标签=模型名)
ROWS = [
    ("outputs/twostage_base/embs.npz",  "stage1", "stage2", "DeBERTa-v3-base"),
    ("outputs/twostage_large/embs.npz", "stage1", "stage2", "DeBERTa-v3-large"),
]


def load_rows():
    """从两个 npz 取 raw/stage1/stage2 三组表征并各做 t-SNE。"""
    out = []
    for npz, k1, k2, label in ROWS:              # 逐模型
        d = np.load(npz)                          # 加载表征
        y = d["y"]                                # 标签
        embs = [d["raw"], d[k1], d[k2]]           # 原始 / stage1 / stage2
        sils = [silhouette_score(e, y, metric="cosine") for e in embs]  # 轮廓系数
        coords = [tsne_2d(e) for e in embs]       # 各降到 2 维
        out.append((label, coords, sils, y))
        print(f"{label}: sil raw={sils[0]:.3f} stage1={sils[1]:.3f} stage2={sils[2]:.3f}", flush=True)
    return out


COL_TITLES = ["Raw", "After stage1", "After stage2"]  # 列标题：标识阶段(不含 s1/s2 超参)


def build(rows, color_by, out_path):
    """画一张 2×3 图：行=模型(行标签为模型名)，列标阶段(Raw/stage1/stage2)；带类别图例。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))   # 2 行 3 列
    for r, (label, coords, sils, y) in enumerate(rows):  # 逐行(模型)
        for c in range(3):                        # 逐列(阶段)
            ax = axes[r][c]
            scatter(ax, coords[c], y, color_by,    # 着色散点
                    title=(COL_TITLES[c] if r == 0 else None),       # 仅首行标阶段名
                    ylabel=(label if c == 0 else None))              # 仅首列标模型名
            ax.text(0.02, 0.97, f"silhouette={sils[c]:.2f}", transform=ax.transAxes,  # 角标轮廓系数
                    fontsize=15, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.75))

    # 类别图例（放在整图右侧）
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
    build(rows, "class", "outputs/best_tsne_19class.png")  # 19 类着色
    build(rows, "group", "outputs/best_tsne_4group.png")   # 4 大组着色
    print("BEST_TSNE_DONE")
