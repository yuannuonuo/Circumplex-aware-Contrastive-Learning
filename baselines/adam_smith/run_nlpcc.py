# -*- coding: utf-8 -*-
"""
run_nlpcc.py —— 真正运行 adam_smith 原仓库的 BertFineTunerPl（SemEval-2023 最佳系统
Adam-Smith 的模型代码），适配到 NLPCC2026 Track1。

直接 import 原仓库 models/BertFineTunerPl.py 的 BertFineTunerPl（AutoModel + CLS/MEAN 池化
+ 分类头），用它**本身的 forward 与 BCEWithLogitsLoss**。把本任务的单标签 19 类编码成
"单正例的 one-hot 多标签目标"（这正是用它的多标签系统处理单标签问题的标准方式），
预测取 argmax。即：模型与损失=原仓库的；数据=本任务的。

backbone 用 deberta-v3-base（原最佳系统用 DeBERTa；large 太慢，这里用 base 快速忠实复现其架构）。
"""

import os, sys, json                            # 标准库
import numpy as np                              # 数值
import torch                                     # 训练
import torch.nn as nn                            # 损失
ADAM = os.path.dirname(__file__)
sys.path.insert(0, ADAM)                          # 使 models / toolbox 命名空间包可导入
sys.path.insert(0, os.path.join(ADAM, "..", "..", "src"))  # 复用本项目数据/标签

from models.BertFineTunerPl import BertFineTunerPl  # ← 原仓库模型（Adam-Smith）
from transformers import AutoTokenizer
from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID, ID2LABEL, NUM_CLASSES
from sklearn.metrics import f1_score, accuracy_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_PATH = "microsoft/deberta-v3-base"


def load(split):
    recs = read_jsonl(f"data/{split}.jsonl")
    return [r[CONSISTENT_KEY] for r in recs], np.array([LABEL2ID[r["Value"]] for r in recs])


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    bs, max_len = 32, 96
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    Xtr, ytr = load("train"); Xdv, ydv = load("dev")
    print(f"device={DEVICE} train={len(Xtr)} dev={len(Xdv)} epochs={epochs}", flush=True)

    # 原仓库 params 字典（用它最佳系统风格：DeBERTa + CLS+MEAN 池化 + BCEWithLogitsLoss）
    params = {"MODEL_PATH": MODEL_PATH, "EMBEDDING": "CLS + MEAN", "HIDDEN_LAYERS": None,
              "DROPOUT": 0.1, "CRITERION": [nn.BCEWithLogitsLoss()]}
    label_cols = [ID2LABEL[i] for i in range(NUM_CLASSES)]
    # ← 实例化原仓库模型
    model = BertFineTunerPl(n_classes=NUM_CLASSES, params=params, label_columns=label_cols,
                            n_training_steps=10, n_warmup_steps=1).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    def batches(X, y, shuffle):
        idx = np.arange(len(X));
        if shuffle: np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s+bs]
            enc = tok([X[i] for i in b], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            yield (enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), y[b])

    @torch.no_grad()
    def eval_dev():
        model.eval(); preds = []
        for ids, mask, _ in batches(Xdv, ydv, False):
            _, out = model(ids, mask)             # ← 原仓库 forward（返回 sigmoid 概率）
            preds.append(out.argmax(-1).cpu().numpy())  # 单标签预测 = argmax
        p = np.concatenate(preds)
        return f1_score(ydv, p, average="macro", zero_division=0), accuracy_score(ydv, p)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0; nb = 0
        for ids, mask, yb in batches(Xtr, ytr, True):
            onehot = torch.zeros(len(yb), NUM_CLASSES, device=DEVICE)  # 单标签 -> 单正例多标签目标
            onehot[torch.arange(len(yb)), yb] = 1.0
            loss, _ = model(ids, mask, onehot)    # ← 原仓库 forward + BCEWithLogitsLoss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += loss.item(); nb += 1
        f1, acc = eval_dev(); best = max(best, f1)
        print(f"[ep{ep}/{epochs}] loss={run/nb:.4f} dev macroF1={f1:.4f} acc={acc:.4f} (best={best:.4f})", flush=True)

    os.makedirs("baselines/adam_smith/nlpcc_out", exist_ok=True)
    torch.save(model.state_dict(), "baselines/adam_smith/nlpcc_out/model.pt")  # 保存模型参数
    json.dump({"repo": "adam_smith (原仓库 BertFineTunerPl: DeBERTa-v3-base + CLS+MEAN + BCEWithLogitsLoss)",
               "task": "NLPCC2026 Track1 单标签19类", "epochs": epochs, "dev_macro_f1_best": best},
              open("baselines/adam_smith/nlpcc_out/result.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n>>> adam_smith 原仓库 BertFineTunerPl 在本任务 dev macroF1(best)={best:.4f}")
    print("ADAM_SMITH_NLPCC_DONE")


if __name__ == "__main__":
    main()
