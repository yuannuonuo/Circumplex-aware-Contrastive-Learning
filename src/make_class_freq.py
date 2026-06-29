# -*- coding: utf-8 -*-
"""
make_class_freq.py —— 画 train 上 19 类的样本频次图（水平条形，按频次降序，按 4 大组着色）。
直观展示长尾（约 6:1），呼应「类别平衡」的必要性。输出 outputs/class_freq.png。
"""
import os, json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"   # 论文字体
plt.rcParams["axes.unicode_minus"] = False

from values import ID2LABEL, NUM_CLASSES, HIGHER_ORDER, LABEL2ID

GROUPS = list(HIGHER_ORDER.keys())
GCOL = {g: plt.get_cmap("tab10")(i) for i, g in enumerate(GROUPS)}
ID2GROUP = {LABEL2ID[v]: g for g, vs in HIGHER_ORDER.items() for v in vs}

recs = [json.loads(l) for l in open("data/train.jsonl")]
cnt = Counter(r["Value"] for r in recs)
ids = sorted(range(NUM_CLASSES), key=lambda i: cnt[ID2LABEL[i]])  # 升序：最长尾在下/上

names = [ID2LABEL[i] for i in ids]
vals = [cnt[ID2LABEL[i]] for i in ids]
cols = [GCOL[ID2GROUP[i]] for i in ids]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(range(len(ids)), vals, color=cols, edgecolor="white")
ax.set_yticks(range(len(ids))); ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("train sample count", fontsize=12)
ax.set_xlim(0, max(vals) * 1.12)
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 4, b.get_y() + b.get_height() / 2, str(v),
            va="center", fontsize=9, color="#333333")
handles = [plt.Line2D([0], [0], marker="s", ls="", color=GCOL[g], label=g, markersize=10) for g in GROUPS]
ax.legend(handles=handles, fontsize=10, loc="lower right", frameon=True)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = "outputs/class_freq_en.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved ->", out)
