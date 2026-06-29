# -*- coding: utf-8 -*-
"""
make_method_figs.py —— 绘制两个核心机制的原理示意图：
  1) outputs/circular_softlabel_demo.png —— 圆形软标签（circular soft-label）
     左：Schwartz 19 类在环上的排列 + 4 个高阶大组；右：以某目标类为例，
     软目标概率随"环上距离"的高斯衰减，对比传统 one-hot。
  2) outputs/supcon_demo.png —— 监督对比学习（supervised contrastive learning）
     嵌入空间里"同价值拉近、异价值推远"，并标出仅作负例的对比难负例。
图内文字用英文，避免中文字体缺失；中文讲解见 REPORT。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge

from values import (circular_soft_labels, circular_distance_matrix,
                    HIGHER_ORDER, LABEL2ID, NUM_CLASSES)

GROUPS = list(HIGHER_ORDER.keys())
GCOL = {g: plt.get_cmap("tab10")(i) for i, g in enumerate(GROUPS)}
ID2GROUP = {LABEL2ID[v]: g for g, vs in HIGHER_ORDER.items() for v in vs}
# 紧凑短名（图上标注用）
SHORT = {0:"SD-thought",1:"SD-action",2:"Stimulation",3:"Hedonism",4:"Achievement",
         5:"Pow-dom",6:"Pow-res",7:"Face",8:"Sec-pers",9:"Sec-soc",10:"Tradition",
         11:"Conf-rules",12:"Conf-interp",13:"Humility",14:"Ben-depend",15:"Ben-caring",
         16:"Univ-concern",17:"Univ-nature",18:"Univ-toler"}


def fig_softlabel():
    C = NUM_CLASSES
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6))

    # ---- 左：19 类圆环 + 4 大组弧段 ----
    ang = np.pi/2 - 2*np.pi*np.arange(C)/C          # 顶部开始顺时针均匀分布
    x, y = np.cos(ang), np.sin(ang)
    # 画 4 大组的彩色外弧
    for g in GROUPS:
        ids = [LABEL2ID[v] for v in HIGHER_ORDER[g]]
        a0 = np.rad2deg(ang[ids[0]]) + 360/C/2
        a1 = np.rad2deg(ang[ids[-1]]) - 360/C/2
        ax1.add_patch(Wedge((0, 0), 1.18, a1, a0, width=0.07, fc=GCOL[g], alpha=0.85))
        amid = np.deg2rad((a0+a1)/2)
        ax1.text(1.34*np.cos(amid), 1.34*np.sin(amid), g, ha="center", va="center",
                 fontsize=11, fontweight="bold", color=GCOL[g],
                 rotation=0)
    # 类点 + 短名
    for i in range(C):
        ax1.scatter(x[i], y[i], s=90, color=GCOL[ID2GROUP[i]], zorder=3, edgecolor="white")
        r = 1.06
        ha = "left" if x[i] > 0.05 else ("right" if x[i] < -0.05 else "center")
        ax1.text(r*x[i], r*y[i], SHORT[i], fontsize=8,
                 ha=ha, va="center", color="black")
    ax1.plot(np.cos(np.linspace(0, 2*np.pi, 200)), np.sin(np.linspace(0, 2*np.pi, 200)),
             color="0.7", lw=1, zorder=0)
    ax1.set_xlim(-1.7, 1.7); ax1.set_ylim(-1.6, 1.6); ax1.set_aspect("equal"); ax1.axis("off")
    ax1.set_title("(a) Schwartz circular structure: 19 values on a ring, 4 higher-order groups",
                  fontsize=11)

    # ---- 右：软目标概率 vs 环距离（高斯衰减） vs one-hot ----
    M = circular_soft_labels()                       # [C,C] 软标签矩阵
    D = circular_distance_matrix()                   # [C,C] 环距离
    t = 2                                            # 目标类示例: Stimulation
    dists = D[t]                                     # 该类到所有类的环距离
    # 按环距离聚合（同一距离权重相同），画到 d=0..9
    dmax = dists.max()
    dd = np.arange(dmax+1)
    probs = [M[t][dists == d][0] for d in dd]        # 每个距离的软标签概率
    ax2.bar(dd-0.18, probs, width=0.36, color="#d62728", label="circular soft-label", zorder=3)
    onehot = [1.0 if d == 0 else 0.0 for d in dd]
    ax2.bar(dd+0.18, onehot, width=0.36, color="0.6", label="one-hot (baseline)", zorder=3)
    for d, p in zip(dd, probs):
        if p > 0.005:
            ax2.text(d-0.18, p+0.012, f"{p:.3f}", ha="center", fontsize=7.5, color="#d62728")
    ax2.set_xlabel("ring distance from true class  (0 = true, 9 = opposite side)", fontsize=10)
    ax2.set_ylabel("target probability mass", fontsize=10)
    ax2.set_xticks(dd)
    ax2.set_ylim(0, 0.95)
    ax2.set_title(f"(b) Soft target for class '{SHORT[t]}':\n"
                  f"peak=0.85 on true class, the rest spread by Gaussian over ring distance",
                  fontsize=11)
    ax2.legend(fontsize=10, frameon=False)
    ax2.grid(axis="y", ls=":", alpha=0.5)

    plt.tight_layout()
    out = "outputs/circular_softlabel_demo.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved ->", out)


def fig_supcon():
    rng = np.random.default_rng(7)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.6))

    def cluster(center, n, s=0.32):
        return rng.normal(center, s, size=(n, 2))

    # ---- (a) 训练前：混在一起 ----
    cA0 = cluster([0.2, 0.3], 8, 0.9); cB0 = cluster([-0.1, -0.2], 8, 0.9); cC0 = cluster([0.0, 0.1], 8, 0.9)
    for c, col, lab in [(cA0, "#d62728", "value A"), (cB0, "#1f77b4", "value B"), (cC0, "#2ca02c", "value C")]:
        ax1.scatter(c[:, 0], c[:, 1], s=55, color=col, alpha=0.8, edgecolor="white", label=lab)
    hn0 = cluster([0.0, 0.0], 4, 0.9)
    ax1.scatter(hn0[:, 0], hn0[:, 1], s=80, marker="X", color="black", label="contrastive\nhard negative")
    ax1.set_title("(a) Before: pretrained embeddings — values entangled", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.legend(fontsize=8.5, loc="upper right", frameon=True)
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)

    # ---- (b) 训练后：同类成簇 + pull/push 箭头 ----
    A = cluster([-1.7, 1.2], 7, 0.28); B = cluster([1.8, 1.0], 7, 0.28); Cc = cluster([0.1, -1.8], 7, 0.28)
    for c, col, lab in [(A, "#d62728", "value A"), (B, "#1f77b4", "value B"), (Cc, "#2ca02c", "value C")]:
        ax2.scatter(c[:, 0], c[:, 1], s=60, color=col, alpha=0.85, edgecolor="white", label=lab)
    # 难负例：靠近 anchor 但永远只作负例（不成簇、不作锚点/正例）
    hn = np.array([[-0.6, 0.7], [-0.3, 0.2]])
    ax2.scatter(hn[:, 0], hn[:, 1], s=110, marker="X", color="black",
                label="contrastive hard negative\n(negative-only, label=-1)", zorder=5)
    anchor = A[0]
    ax2.scatter(*anchor, s=180, facecolor="none", edgecolor="#d62728", linewidth=2.4, zorder=6)
    ax2.annotate("anchor", anchor, textcoords="offset points", xytext=(-6, 12), fontsize=9, color="#d62728")

    def arrow(p, q, color, style="-", lw=2):
        ax2.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=16,
                                      color=color, lw=lw, ls=style, zorder=4, alpha=0.9))
    # pull: anchor -> 同类正例（绿色实线）
    for q in A[1:4]:
        arrow(anchor, q, "#2e7d32", "-", 2)
    # push: anchor -> 异类负例 + 难负例（红色虚线，方向示意"推远"）
    for q in list(B[:2]) + list(Cc[:2]) + list(hn):
        arrow(anchor, q, "#c62828", (0, (4, 3)), 1.6)
    ax2.plot([], [], color="#2e7d32", lw=2, label="pull together (same value)")
    ax2.plot([], [], color="#c62828", lw=1.6, ls="--", label="push apart (different value / hard neg.)")
    ax2.set_title("(b) After supervised contrastive learning —\nsame value pulled together, others pushed apart",
                  fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.legend(fontsize=8, loc="lower right", frameon=True)
    ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3)

    plt.tight_layout()
    out = "outputs/supcon_demo.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    fig_softlabel()
    fig_supcon()
    print("METHOD_FIGS_DONE")
