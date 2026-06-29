# -*- coding: utf-8 -*-
"""
run_nlpcc.py —— 真正运行 su0315 原仓库的 BaselineModel，适配到 NLPCC2026 Track1。

与 ../run_baselines.py（我自己重写的方法）不同：本脚本**直接 import 并运行原仓库的
model/model_baseline.py 里的 BaselineModel（BERT + dropout + linear）**，只把任务改成
本任务——单标签 19 类、输入=单段 response、损失改 softmax 交叉熵（原仓库是多标签 BCE）。

即：模型代码=原仓库的；数据/标签/损失=本任务的适配。
"""

import os, sys, json                            # 标准库
import numpy as np                              # 数值
import torch                                     # 训练
import torch.nn as nn                            # 损失
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))  # 原仓库 model/ 无 __init__，直接加目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))  # 复用本项目数据/标签工具

from model_baseline import BaselineModel, device  # ← 原仓库的模型与设备
from data import read_jsonl, CONSISTENT_KEY              # 本项目数据读取
from values import LABEL2ID, NUM_CLASSES                  # 19 类标签映射
from sklearn.metrics import f1_score, accuracy_score      # 指标


def load(split):
    """读本任务数据，返回 (response 文本列表, 单标签 id 数组)。"""
    recs = read_jsonl(f"data/{split}.jsonl")
    X = [r[CONSISTENT_KEY] for r in recs]
    y = np.array([LABEL2ID[r["Value"]] for r in recs])
    return X, y


@torch.no_grad()
def eval_dev(model, X, y, bs=32):
    """用原仓库模型在 dev 上预测，算 macro-F1 / acc。"""
    model.eval(); preds = []
    for i in range(0, len(X), bs):
        logits = model(X[i:i+bs])                # ← 原仓库 forward（内部自带分词）
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return f1_score(y, preds, average="macro", zero_division=0), accuracy_score(y, preds)


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4    # 默认 4 epoch
    bs = 32
    Xtr, ytr = load("train"); Xdv, ydv = load("dev")
    print(f"device={device}  train={len(Xtr)} dev={len(Xdv)}  epochs={epochs}", flush=True)

    # 原仓库 BaselineModel(output_size, l1_labels, l1_to_l2_map, l1_exs)；后三个在 baseline 里未用
    model = BaselineModel(NUM_CLASSES, None, None, None).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()                    # 单标签 softmax CE（原仓库是多标签 BCE）
    idx = np.arange(len(Xtr))
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); np.random.shuffle(idx); run = 0.0; nb = 0
        for s in range(0, len(idx), bs):
            b = idx[s:s+bs]
            texts = [Xtr[i] for i in b]
            labels = torch.tensor(ytr[b], dtype=torch.long, device=device)
            logits = model(texts)                 # ← 原仓库 forward
            loss = ce(logits, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += loss.item(); nb += 1
        f1, acc = eval_dev(model, Xdv, ydv)
        best = max(best, f1)
        print(f"[ep{ep}/{epochs}] loss={run/nb:.4f}  dev macroF1={f1:.4f} acc={acc:.4f} (best={best:.4f})", flush=True)

    os.makedirs("baselines/su0315_hvd/nlpcc_out", exist_ok=True)
    torch.save(model.state_dict(), "baselines/su0315_hvd/nlpcc_out/model.pt")  # 保存模型参数
    json.dump({"repo": "su0315_hvd (原仓库 BaselineModel: bert-base-uncased + linear)",
               "task": "NLPCC2026 Track1 单标签19类", "epochs": epochs,
               "dev_macro_f1_best": best},
              open("baselines/su0315_hvd/nlpcc_out/result.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n>>> su0315 原仓库 BaselineModel 在本任务 dev macroF1(best)={best:.4f}")
    print("SU0315_NLPCC_DONE")


if __name__ == "__main__":
    main()
