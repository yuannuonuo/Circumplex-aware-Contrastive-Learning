# -*- coding: utf-8 -*-
"""
predict.py —— 单模型推理，生成提交文件 preds/track1.pred.jsonl。

加载一个 ValueClassifier checkpoint（如 joint_large/joint_model.pt），对 test 的每段
response 预测主导价值（19 选 1），输出与测试输入逐行对齐。**单模型，非集成。**

用法：
  python src/predict.py --model microsoft/deberta-v3-large \
      --ckpt outputs/joint_large/joint_model.pt --out preds/track1.pred.jsonl
"""

import argparse, json, os
import torch
from transformers import AutoTokenizer

from values import ID2LABEL
from data import read_jsonl, CONSISTENT_KEY
from model import ValueClassifier
from train import get_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-large")  # backbone
    ap.add_argument("--ckpt", default="outputs/joint_large/joint_model.pt")  # 单个模型权重
    ap.add_argument("--test", default="data/test/track1.jsonl")             # 测试输入
    ap.add_argument("--out", default="preds/track1.pred.jsonl")            # 输出
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    device = get_device()

    tok = AutoTokenizer.from_pretrained(args.model)            # 分词器
    model = ValueClassifier(args.model).to(device)            # 模型
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()  # 载权重
    recs = read_jsonl(args.test)                              # 读测试（只含 response）
    texts = [r[CONSISTENT_KEY] for r in recs]
    print(f"{len(texts)} test responses; model={args.model} ckpt={args.ckpt}", flush=True)

    preds = []
    for i in range(0, len(texts), args.bs):                   # 分批前向
        enc = tok(texts[i:i+args.bs], truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())        # argmax 取主导价值

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps({"Value": ID2LABEL[int(p)]}, ensure_ascii=False) + "\n")
    n = sum(1 for _ in open(args.out, encoding="utf-8"))      # 行数校验：必须与测试输入逐行一致
    assert n == len(recs), f"line mismatch: {n} preds vs {len(recs)} inputs"
    print(f"wrote {n} predictions -> {args.out}  (line count matches input)")


if __name__ == "__main__":
    main()
