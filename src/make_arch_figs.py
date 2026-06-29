# -*- coding: utf-8 -*-
"""
make_arch_figs.py —— 绘制两阶段 / 联合损失两种训练方案的模型结构图。
输出 outputs/arch_twostage.png 与 outputs/arch_joint.png。
共享底座：DeBERTa 编码器 → 均值池化 → {分类头 / 投影头}（见 src/model.py: ValueClassifier）。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#1F3A5F"; BLUE = "#2E5E8C"; ENC = "#3A6FB0"; ORANGE = "#E87D0D"
GREEN = "#2CA02C"; RED = "#C0392B"; GRAY = "#555555"; LBLUE = "#DCE6F2"; LOSSBG = "#FBE6CC"


def box(ax, x, y, w, h, text, fc, tc="white", fs=12, bold=True, ec=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                       fc=fc, ec=ec or fc, lw=1.5, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, text=None, color=GRAY, style="-|>", lw=2.0, dy=0.12, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, zorder=1, linestyle=ls))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + dy, text, ha="center", va="bottom",
                fontsize=9.5, color=color)


def base(figsize=(13, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    return fig, ax


# ============================ 两阶段 ============================
fig, ax = base()
# 阶段标签
ax.text(0.1, 5.55, "阶段一：监督对比塑造表征", fontsize=14, fontweight="bold", color=BLUE)
ax.text(0.1, 2.75, "阶段二：分类微调", fontsize=14, fontweight="bold", color=ORANGE)

# --- 阶段一 (y ~ 3.9) ---
y1 = 3.85
box(ax, 0.2, y1, 1.5, 0.95, "回答\n文本", GRAY, fs=12)
box(ax, 2.2, y1-0.12, 2.0, 1.2, "DeBERTa\n编码器", ENC, fs=13)
box(ax, 4.7, y1, 1.5, 0.95, "均值\n池化", BLUE, fs=12)
box(ax, 6.8, y1, 1.7, 0.95, "投影头\nproj", ORANGE, fs=12)
box(ax, 9.1, y1, 1.5, 0.95, "z\n(L2 归一)", GREEN, fs=11)
box(ax, 11.0, y1-0.05, 1.8, 1.05, "SupCon\n监督对比损失", LOSSBG, tc=RED, fs=12)
arrow(ax, 1.7, y1+0.48, 2.2, y1+0.48)
arrow(ax, 4.2, y1+0.48, 4.7, y1+0.48)
arrow(ax, 6.2, y1+0.48, 6.8, y1+0.48)
arrow(ax, 8.5, y1+0.48, 9.1, y1+0.48)
arrow(ax, 10.6, y1+0.48, 11.0, y1+0.48)
ax.text(11.9, y1-0.45, "Contrastive Response\n作对比难负例 (label=-1)", ha="center",
        fontsize=8.5, color=RED)

# --- 阶段二 (y ~ 1.1) ---
y2 = 1.05
box(ax, 0.2, y2, 1.5, 0.95, "回答\n文本", GRAY, fs=12)
box(ax, 2.2, y2-0.12, 2.0, 1.2, "DeBERTa\n编码器", ENC, fs=13)
box(ax, 4.7, y2, 1.5, 0.95, "均值\n池化", BLUE, fs=12)
box(ax, 6.8, y2, 1.7, 0.95, "分类头\nclassifier", ORANGE, fs=12)
box(ax, 9.1, y2, 1.5, 0.95, "logits\n(19)", GREEN, fs=11)
box(ax, 11.0, y2-0.05, 1.8, 1.05, "圆形软标签CE\n+ 类别平衡", LOSSBG, tc=RED, fs=11)
arrow(ax, 1.7, y2+0.48, 2.2, y2+0.48)
arrow(ax, 4.2, y2+0.48, 4.7, y2+0.48)
arrow(ax, 6.2, y2+0.48, 6.8, y2+0.48)
arrow(ax, 8.5, y2+0.48, 9.1, y2+0.48)
arrow(ax, 10.6, y2+0.48, 11.0, y2+0.48)
# 编码器权重初始化箭头（阶段一 -> 阶段二）
arrow(ax, 3.2, y1-0.12, 3.2, y2+1.08, text="  权重初始化", color=NAVY, lw=2.2, dy=0.0, ls=(0,(4,3)))
ax.text(3.35, 2.55, "编码器权重\n传递", ha="left", va="center", fontsize=9, color=NAVY)

plt.tight_layout()
out1 = "outputs/arch_twostage.png"
plt.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved ->", out1)


# ============================ 联合损失 ============================
fig, ax = base()
ax.text(0.1, 5.5, "联合损失：单阶段端到端联合优化", fontsize=14, fontweight="bold", color=NAVY)

ym = 2.9  # 主干中线
box(ax, 0.2, ym-0.5, 1.5, 1.0, "回答\n文本", GRAY, fs=12)
box(ax, 2.2, ym-0.62, 2.0, 1.24, "DeBERTa\n编码器", ENC, fs=13)
box(ax, 4.7, ym-0.5, 1.6, 1.0, "均值\n池化", BLUE, fs=12)
arrow(ax, 1.7, ym, 2.2, ym)
arrow(ax, 4.2, ym, 4.7, ym)

# 分支点
bx = 6.3
# 上分支：分类头
yu = 4.25
box(ax, 7.2, yu-0.45, 1.7, 0.9, "分类头\nclassifier", ORANGE, fs=12)
box(ax, 9.3, yu-0.45, 1.4, 0.9, "logits\n(19)", GREEN, fs=11)
box(ax, 11.0, yu-0.5, 1.85, 1.0, "圆形软标签CE\n+ 类别平衡", LOSSBG, tc=RED, fs=11)
arrow(ax, bx, ym+0.35, 7.2, yu, color=ORANGE)
arrow(ax, 8.9, yu, 9.3, yu)
arrow(ax, 10.7, yu, 11.0, yu)

# 下分支：投影头
yd = 1.55
box(ax, 7.2, yd-0.45, 1.7, 0.9, "投影头\nproj", ORANGE, fs=12)
box(ax, 9.3, yd-0.45, 1.4, 0.9, "z\n(L2 归一)", GREEN, fs=11)
box(ax, 11.0, yd-0.5, 1.85, 1.0, "SupCon\n监督对比", LOSSBG, tc=RED, fs=12)
arrow(ax, bx, ym-0.35, 7.2, yd, color=ORANGE)
arrow(ax, 8.9, yd, 9.3, yd)
arrow(ax, 10.7, yd, 11.0, yd)
ax.text(11.9, yd-0.9, "Contrastive Response\n作对比难负例", ha="center", fontsize=8.5, color=RED)

# 总损失
box(ax, 4.0, 0.15, 6.5, 0.62,
    "总损失  L = 圆形软标签CE + 类别平衡 + λ·SupCon   (λ=0.3)", NAVY, fs=12)

plt.tight_layout()
out2 = "outputs/arch_joint.png"
plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved ->", out2)
print("ARCH_FIGS_DONE")
