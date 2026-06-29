# -*- coding: utf-8 -*-
"""
run_nlpcc.py —— 运行 cocchieri 原仓库（BERT.ipynb）抽取出的 BERTClass + loss_fn，
适配到 NLPCC2026 Track1（单标签 19 类、输入=response）。backbone 用 roberta-base。

模型与损失=原仓库的（extracted_model.py，逐字抽取，仅分类头 20→19）；
本任务单标签编码为单正例 one-hot 多标签目标（配合其 BCEWithLogitsLoss），预测取 argmax。
"""

import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from extracted_model import BERTClass, loss_fn          # ← 原仓库抽取的模型与损失
from transformers import AutoTokenizer
from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID, NUM_CLASSES
from sklearn.metrics import f1_score, accuracy_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_NAME = "roberta-base"
MAX_LEN, BS = 96, 16          # 原 notebook 用 MAX_LEN=75/BATCH_SIZE=16；这里 96 适配本任务文本


def load(split):
    recs = read_jsonl(f"data/{split}.jsonl")
    return [r[CONSISTENT_KEY] for r in recs], np.array([LABEL2ID[r["Value"]] for r in recs])


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    Xtr, ytr = load("train"); Xdv, ydv = load("dev")
    print(f"device={DEVICE} model={MODEL_NAME} train={len(Xtr)} dev={len(Xdv)} epochs={epochs}", flush=True)

    model = BERTClass(MODEL_NAME, NUM_CLASSES).to(DEVICE)   # ← 原仓库模型
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    def batches(X, y, shuffle):
        idx = np.arange(len(X))
        if shuffle: np.random.shuffle(idx)
        for s in range(0, len(idx), BS):
            b = idx[s:s+BS]
            enc = tok([X[i] for i in b], truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].to(DEVICE); mask = enc["attention_mask"].to(DEVICE)
            ttids = torch.zeros_like(ids)        # RoBERTa 只用 token_type 0（原 forward 需要该参数）
            yield ids, mask, ttids, y[b]

    @torch.no_grad()
    def eval_dev():
        model.eval(); preds = []
        for ids, mask, ttids, _ in batches(Xdv, ydv, False):
            out = model(ids, mask, ttids)         # ← 原仓库 forward
            preds.append(out.argmax(-1).cpu().numpy())
        p = np.concatenate(preds)
        return f1_score(ydv, p, average="macro", zero_division=0), accuracy_score(ydv, p)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0; nb = 0
        for ids, mask, ttids, yb in batches(Xtr, ytr, True):
            onehot = torch.zeros(len(yb), NUM_CLASSES, device=DEVICE)  # 单标签→单正例多标签目标
            onehot[torch.arange(len(yb)), yb] = 1.0
            out = model(ids, mask, ttids)         # ← 原仓库 forward
            loss = loss_fn(out, onehot)           # ← 原仓库 BCEWithLogitsLoss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += loss.item(); nb += 1
        f1, acc = eval_dev(); best = max(best, f1)
        print(f"[ep{ep}/{epochs}] loss={run/nb:.4f} dev macroF1={f1:.4f} acc={acc:.4f} (best={best:.4f})", flush=True)

    os.makedirs("baselines/cocchieri_bert_roberta/nlpcc_out", exist_ok=True)
    torch.save(model.state_dict(), "baselines/cocchieri_bert_roberta/nlpcc_out/model.pt")  # 保存模型参数
    json.dump({"repo": "cocchieri (BERT.ipynb 抽取的 BERTClass + BCEWithLogitsLoss, roberta-base)",
               "task": "NLPCC2026 Track1 单标签19类", "epochs": epochs, "dev_macro_f1_best": best},
              open("baselines/cocchieri_bert_roberta/nlpcc_out/result.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n>>> cocchieri 原仓库 BERTClass(roberta-base) 在本任务 dev macroF1(best)={best:.4f}")
    print("COCCHIERI_NLPCC_DONE")


if __name__ == "__main__":
    main()
