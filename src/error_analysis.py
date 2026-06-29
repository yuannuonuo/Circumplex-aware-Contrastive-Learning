# -*- coding: utf-8 -*-
"""
error_analysis.py —— 在 dev 上做逐类 F1 + 混淆诊断。

重点关注"组内（圆形相邻）混淆"——这正是整套设计要解决的失败模式：
统计有多少错误落在同一高阶组内、错误按圆形距离如何分布。
"""

import argparse                                # 命令行参数
import numpy as np                              # 数值
import torch                                    # 推理
from collections import Counter                 # 计数
from transformers import AutoTokenizer          # 分词器
from sklearn.metrics import f1_score            # 逐类 F1

from values import ID2LABEL, ID2GROUP, circular_distance_matrix, NUM_CLASSES  # 标签工具
from data import read_jsonl, ResponseDataset, make_collate                    # 数据工具
from model import ValueClassifier               # 模型
from train import get_device, evaluate          # 设备 + 评测函数复用
from torch.utils.data import DataLoader         # 数据迭代器


def main():
    """加载某 checkpoint，在 dev 上算整体/逐类指标与圆形距离误差剖面。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")  # 模型名
    ap.add_argument("--ckpt", required=True)                         # 权重路径
    ap.add_argument("--dev", default="data/test.jsonl")               # 验证集
    args = ap.parse_args()
    device = get_device()                        # 选设备

    tok = AutoTokenizer.from_pretrained(args.model)  # 分词器
    recs = read_jsonl(args.dev)                      # 读 dev
    ds = ResponseDataset(recs, tok, 96, False)        # 构造数据集
    loader = DataLoader(ds, batch_size=64, collate_fn=make_collate(tok, 96, False))  # loader
    model = ValueClassifier(args.model).to(device)    # 构造模型
    model.load_state_dict(torch.load(args.ckpt, map_location=device))  # 加载权重

    metrics, preds, golds = evaluate(model, loader, device)  # 跑评测，拿预测与金标签
    print("OVERALL:", {k: round(v, 4) for k, v in metrics.items()})  # 打印整体指标

    print("\nPER-CLASS F1 (sorted):")            # 逐类 F1（从低到高排序，便于看最差的类）
    names = [ID2LABEL[i] for i in range(NUM_CLASSES)]  # 类别名列表
    f1s = f1_score(golds, preds, labels=list(range(NUM_CLASSES)),     # 各类 F1
                   average=None, zero_division=0)
    for i in np.argsort(f1s):                     # 按 F1 升序遍历
        print(f"  {f1s[i]:.3f}  {names[i]}")       # 打印每类 F1 与名字

    # 错误的圆形距离剖面：看错误是不是集中在"相邻类别"
    cd = circular_distance_matrix()               # 圆形距离矩阵
    errs = [(g, p) for g, p in zip(golds, preds) if g != p]  # 收集所有错分对 (真值, 预测)
    dist_hist = Counter(int(cd[g, p]) for g, p in errs)       # 按圆形距离统计错误个数
    in_group = sum(ID2GROUP[g] == ID2GROUP[p] for g, p in errs)  # 落在同一高阶组内的错误数
    print(f"\n{len(errs)} errors total; {in_group} ({in_group/max(len(errs),1):.0%}) "
          f"are within the same higher-order group")            # 组内错误占比
    print("error count by circular distance (1=adjacent neighbour):")
    for d in sorted(dist_hist):                   # 按距离从小到大打印
        print(f"  dist {d}: {dist_hist[d]}")       # 该距离上的错误个数


if __name__ == "__main__":
    main()
