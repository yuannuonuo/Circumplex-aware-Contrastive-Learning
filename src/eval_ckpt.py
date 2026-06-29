# -*- coding: utf-8 -*-
"""
eval_ckpt.py —— 加载一个 ValueClassifier checkpoint，在 dev 上复算 macro-F1 / acc。

适用于本项目所有 ValueClassifier 模型：联合损失（joint）、两阶段（twostage）、
消融（ablation）、标准神经基线（neural_baseline）。加载即确定性复现。

用法：
  python src/eval_ckpt.py --model microsoft/deberta-v3-large --ckpt outputs/joint_large/joint_model.pt
"""

import argparse
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model import ValueClassifier
from data import read_jsonl, ResponseDataset, make_collate
from train import get_device, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="backbone 名，如 microsoft/deberta-v3-large")
    ap.add_argument("--ckpt", required=True, help="ValueClassifier state_dict 路径")
    ap.add_argument("--dev", default="data/test.jsonl")
    args = ap.parse_args()
    device = get_device()

    tok = AutoTokenizer.from_pretrained(args.model)
    m = ValueClassifier(args.model).to(device)
    m.load_state_dict(torch.load(args.ckpt, map_location=device))  # 加载参数
    dev = read_jsonl(args.dev)
    loader = DataLoader(ResponseDataset(dev, tok, 96, False), batch_size=64,
                        collate_fn=make_collate(tok, 96, False))
    metrics, _, _ = evaluate(m, loader, device)
    print(f"{args.ckpt}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
