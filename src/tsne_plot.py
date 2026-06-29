# -*- coding: utf-8 -*-
"""
tsne_plot.py —— t-SNE 出图的通用工具（降维、按 19 类 / 4 大组着色）。

被 make_best_tsne.py / make_joint_tsne.py / make_lowbudget_tsne.py 复用。
"""

import numpy as np                              # 数值
from sklearn.manifold import TSNE                # 降维
import matplotlib                                # 绘图
matplotlib.use("Agg")                            # 无界面后端
import matplotlib.pyplot as plt                  # 画图
plt.rcParams["font.family"] = "Times New Roman"   # 论文字体
plt.rcParams["axes.unicode_minus"] = False        # 负号用 ASCII，避免缺字形

from values import NUM_CLASSES, ID2GROUP, HIGHER_ORDER  # 标签工具

GROUP_NAMES = list(HIGHER_ORDER.keys())            # 4 大组名（固定顺序）
GROUP2ID = {g: i for i, g in enumerate(GROUP_NAMES)}  # 组名 -> 0..3
# 与论文 Figure 1（Schwartz 圆环）高阶维度配色一致，按 GROUP_NAMES 顺序：
#   Openness-to-change / Self-enhancement / Conservation / Self-transcendence
GROUP_COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD"]


def tsne_2d(X, seed=42):
    """高维向量 -> 2 维（t-SNE，PCA 初始化）。"""
    return TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(X)


def scatter(ax, xy, y, color_by, title=None, ylabel=None):
    """在 ax 上按 19 类或 4 大组着色散点。"""
    if color_by == "class":                       # 19 类：tab20 色板
        cmap = plt.get_cmap("tab20")
        for c in range(NUM_CLASSES):
            p = xy[y == c]
            if len(p): ax.scatter(p[:, 0], p[:, 1], s=9, color=cmap(c % 20), alpha=0.85, linewidths=0)
    else:                                         # 4 大组：用 Figure 1 配色
        g = np.array([GROUP2ID[ID2GROUP[int(c)]] for c in y])  # 类 id -> 组 id
        for gi in range(4):
            p = xy[g == gi]
            if len(p): ax.scatter(p[:, 0], p[:, 1], s=9, color=GROUP_COLORS[gi], alpha=0.85, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])           # 去刻度
    if title: ax.set_title(title, fontsize=20)      # 列标题
    if ylabel: ax.set_ylabel(ylabel, fontsize=19)   # 行/阶段标注
