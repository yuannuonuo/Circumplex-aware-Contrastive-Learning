# -*- coding: utf-8 -*-
"""
neural_baseline.py —— 神经基线：标准 Transformer 纯微调（单标签 19 类，one-hot 交叉熵）。

与经典基线（classical_baseline.py，TF-IDF）对照。这里是"朴素神经基线"——直接微调一个
预训练 transformer（bert-base / roberta-base），不加圆形软标签 / 类别平衡 / SupCon，
用来衬托本项目结构化方法相对于"普通微调"的增益。
"""

import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from data import read_jsonl, ResponseDataset, make_collate
from model import ValueClassifier, soft_label_ce
from values import NUM_CLASSES
from train import get_device, evaluate


def run(model_name, epochs, device, train_recs, dev_recs):
    """朴素微调一个 backbone：单标签 19 类，one-hot CE，无任何结构化技巧。"""
    tok = AutoTokenizer.from_pretrained(model_name)
    bs = 16 if "large" in model_name else 32
    tr = DataLoader(ResponseDataset(train_recs, tok, 96, False), batch_size=bs, shuffle=True,
                    collate_fn=make_collate(tok, 96, False))
    dv = DataLoader(ResponseDataset(dev_recs, tok, 96, False), batch_size=64,
                    collate_fn=make_collate(tok, 96, False))
    model = ValueClassifier(model_name).to(device)
    onehot = torch.eye(NUM_CLASSES, device=device)            # one-hot 目标（无软标签）
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total = len(tr) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tr:
            ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
            labels = b["labels"].to(device)
            loss = soft_label_ce(model(ids, mask), onehot[labels], None, labels)  # 普通 CE，无类别平衡
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        m, _, _ = evaluate(model, dv, device); best = max(best, m["macro_f1"])
        print(f"   {model_name} ep{ep}/{epochs} macroF1={m['macro_f1']:.4f} (best={best:.4f})", flush=True)
    os.makedirs("baselines/neural_models", exist_ok=True)
    torch.save(model.state_dict(), f"baselines/neural_models/{model_name.split('/')[-1]}.pt")  # 保存模型参数
    return best


def main():
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["bert-base-uncased", "roberta-base"]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    device = get_device()
    train_recs = read_jsonl("data/train.jsonl"); dev_recs = read_jsonl("data/test.jsonl")
    res = {}
    for mn in models:
        print(f"\n#### 神经基线 {mn} (纯微调, {epochs} epoch) ####", flush=True)
        best = run(mn, epochs, device, train_recs, dev_recs)
        res[mn] = {"dev_macro_f1": float(best)}
        print(f"  [{mn}] best dev macroF1={best:.4f}", flush=True)
    json.dump(res, open("baselines/neural_results.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== 神经基线汇总 ==="); [print(f"  {k}: {v['dev_macro_f1']:.4f}") for k, v in res.items()]
    print("NEURAL_DONE")


if __name__ == "__main__":
    main()
