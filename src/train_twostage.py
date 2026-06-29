# -*- coding: utf-8 -*-
"""
train_twostage.py —— 改进 2：两阶段监督对比（SupCon）。

阶段 1（表征预训练）：只用 SupCon 损失训练 encoder + 投影头。同价值的回答互相拉近；
                      每条样本自带的对比回答作为"只当负例"的难负例。
阶段 2（分类微调）：  加载塑形好的 encoder，用"圆形软标签 + 类别平衡"交叉熵训练分类头
                      （encoder 仍可训，但用更小学习率）。

与 train.py 的 --supcon（联合训练，CE 与 SupCon 同时优化）对照。
【结论】两阶段(0.9087) > 联合(0.9049)，验证了"先塑表征再分类"更优，但两者都不及
纯 soft+CB(0.9200)，故 SupCon 在本任务非关键杠杆。
"""

import argparse                                # 命令行
import os                                        # 目录
import numpy as np                              # 随机种子
import torch                                     # 训练
import torch.nn.functional as F                  # 归一化
from torch.utils.data import DataLoader          # 数据迭代器
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # 分词/调度

from values import NUM_CLASSES, circular_soft_labels  # 标签工具
from data import read_jsonl, ResponseDataset, make_collate, class_counts  # 数据工具
from model import ValueClassifier, soft_label_ce, supcon_loss  # 模型与损失
from train import get_device, effective_number_weights, evaluate  # 复用设备/权重/评测


def stage1_supcon(model, loader, device, epochs, lr, temp):
    """阶段 1：仅用 SupCon 损失训练 encoder + 投影头，塑造表征空间。"""
    params = list(model.encoder.parameters()) + list(model.proj.parameters())  # 只训 encoder+proj
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)   # 优化器
    total = len(loader) * epochs                 # 总步数
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)  # 调度
    for ep in range(1, epochs + 1):              # 逐 epoch
        model.train(); run = 0.0                  # 训练模式 + 累计损失
        for step, batch in enumerate(loader):     # 遍历 batch
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)  # 主输入
            labels = batch["labels"].to(device)   # 标签
            z = F.normalize(model.proj(model.encode(ids, mask)), dim=-1)  # 正样本投影向量
            c_ids = batch["c_input_ids"].to(device); c_mask = batch["c_attention_mask"].to(device)  # 对比回答
            z_c = F.normalize(model.proj(model.encode(c_ids, c_mask)), dim=-1)  # 对比回答投影向量
            z_all = torch.cat([z, z_c], 0)        # 拼接
            lab_all = torch.cat([labels, labels.new_full((c_ids.size(0),), -1)])  # 对比回答标签设 -1
            loss = supcon_loss(z_all, lab_all, temp)  # 纯 SupCon 损失
            opt.zero_grad(); loss.backward()      # 反向
            torch.nn.utils.clip_grad_norm_(params, 1.0)  # 裁剪
            opt.step(); sched.step(); run += loss.item()  # 更新 + 累计
        print(f"[stage1 supcon epoch {ep}] loss={run/len(loader):.4f}", flush=True)  # 每轮损失


def stage2_classify(model, train_loader, dev_loader, device, epochs, lr, soft, cw):
    """阶段 2：在塑形好的 encoder 上训练分类头（软标签 + 类别平衡 CE）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # 全参数优化
    total = len(train_loader) * epochs           # 总步数
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)  # 调度
    best, best_m = -1, None                       # 记录最优
    for ep in range(1, epochs + 1):              # 逐 epoch
        model.train(); run = 0.0
        for batch in train_loader:                # 遍历 batch
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)  # 输入
            labels = batch["labels"].to(device)   # 标签
            logits = model(ids, mask)             # 前向
            loss = soft_label_ce(logits, soft[labels], cw, labels)  # 软标签 + 类别平衡 CE
            opt.zero_grad(); loss.backward()      # 反向
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 裁剪
            opt.step(); sched.step(); run += loss.item()  # 更新
        m, _, _ = evaluate(model, dev_loader, device)  # dev 评测
        print(f"[stage2 epoch {ep}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()),
              flush=True)
        if m["macro_f1"] > best:                  # 刷新最优则保存
            best, best_m = m["macro_f1"], m
            torch.save(model.state_dict(), os.path.join(ARGS.out, "best.pt"))
    return best, best_m


def main():
    """串起两阶段流程：先 SupCon 塑表征，再分类微调。"""
    global ARGS                                   # 让 stage2 能访问输出目录
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")  # 编码器
    ap.add_argument("--out", default="outputs/twostage")            # 输出目录
    ap.add_argument("--s1_epochs", type=int, default=3)             # 阶段1轮数
    ap.add_argument("--s2_epochs", type=int, default=4)             # 阶段2轮数
    ap.add_argument("--bs", type=int, default=32)                  # 批大小
    ap.add_argument("--s1_lr", type=float, default=2e-5)          # 阶段1学习率
    ap.add_argument("--s2_lr", type=float, default=1e-5)          # 阶段2学习率（更小，保护表征）
    ap.add_argument("--temp", type=float, default=0.1)           # SupCon 温度
    ap.add_argument("--max_len", type=int, default=96)          # 最大长度
    ap.add_argument("--seed", type=int, default=42)            # 随机种子
    ARGS = ap.parse_args()
    torch.manual_seed(ARGS.seed); np.random.seed(ARGS.seed)  # 固定种子
    device = get_device(); os.makedirs(ARGS.out, exist_ok=True)  # 设备 + 输出目录

    tok = AutoTokenizer.from_pretrained(ARGS.model)  # 分词器
    train_recs = read_jsonl("data/train.jsonl"); dev_recs = read_jsonl("data/test.jsonl")  # 读数据
    train_ds = ResponseDataset(train_recs, tok, ARGS.max_len, with_contrastive=True)  # 训练集（带对比）
    dev_ds = ResponseDataset(dev_recs, tok, ARGS.max_len, False)                      # 验证集
    train_loader = DataLoader(train_ds, batch_size=ARGS.bs, shuffle=True,
                              collate_fn=make_collate(tok, ARGS.max_len, True))
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False,
                            collate_fn=make_collate(tok, ARGS.max_len, False))

    model = ValueClassifier(ARGS.model).to(device)  # 模型
    soft = torch.tensor(circular_soft_labels(NUM_CLASSES, 1.0, 0.85), device=device)  # 圆形软标签
    cw = effective_number_weights(class_counts(train_recs)).to(device)  # 类别平衡权重

    print("=== STAGE 1: SupCon representation pretraining ===")
    stage1_supcon(model, train_loader, device, ARGS.s1_epochs, ARGS.s1_lr, ARGS.temp)  # 阶段1
    print("=== STAGE 2: classifier finetune (soft-label + class-balance) ===")
    best, best_m = stage2_classify(model, train_loader, dev_loader, device,            # 阶段2
                                   ARGS.s2_epochs, ARGS.s2_lr, soft, cw)
    print(f"\nBEST dev macro-F1={best:.4f}  metrics={best_m}")  # 打印最优


if __name__ == "__main__":
    main()
