# -*- coding: utf-8 -*-
"""
train_joint_ablation.py —— 联合训练的消融实验（单独的 ablation 代码文件）。

在"完整联合损失"（圆形软标签CE + 类别平衡 + λ·SupCon(含 Contrastive 难负例)）基础上，
用 4 个开关做留一法消融，每次只关掉一个组件：
  --no_soft_label   ：用 one-hot 硬标签替代圆形软标签
  --no_class_balance：用均匀权重替代有效样本数加权
  --supcon 0        ：去掉 SupCon（纯分类）
  --no_contrastive  ：保留 SupCon 但不使用 Contrastive Response 难负例（只用 consistent）

单跑（固定 seed），保存最佳 .pt 与分数。本批用于"低预算(≤5 epoch)"消融，故数值偏低且
**须标注欠训设置**——所有变体同预算，是公平的留一对比。
"""

import argparse, os, json                       # 命令行/文件
import numpy as np                              # 数值
import torch                                     # 训练
import torch.nn.functional as F                  # 归一化
from torch.utils.data import DataLoader          # 数据迭代器
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # 分词/调度

from values import NUM_CLASSES, circular_soft_labels  # 标签工具
from data import read_jsonl, ResponseDataset, make_collate, class_counts  # 数据
from model import ValueClassifier, soft_label_ce, supcon_loss  # 模型与损失
from train import get_device, effective_number_weights, evaluate  # 复用


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")  # backbone
    ap.add_argument("--out", default="outputs/ablation/full")       # 输出目录
    ap.add_argument("--epochs", type=int, default=5)               # 低预算 epoch
    ap.add_argument("--bs", type=int, default=32)                # 批大小
    ap.add_argument("--lr", type=float, default=2e-5)           # 学习率
    ap.add_argument("--supcon", type=float, default=0.3)       # λ（0 = 去 SupCon）
    ap.add_argument("--temp", type=float, default=0.10)       # SupCon 温度
    ap.add_argument("--no_soft_label", action="store_true")  # 消融①：用 one-hot
    ap.add_argument("--no_class_balance", action="store_true")  # 消融②：均匀权重
    ap.add_argument("--no_contrastive", action="store_true")  # 消融④：SupCon 不用对比难负例
    ap.add_argument("--tag", default="full")                 # 变体名（写进记录）
    ap.add_argument("--seed", type=int, default=42)          # 随机种子（单跑）
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)  # 固定种子
    device = get_device(); os.makedirs(args.out, exist_ok=True)
    print(f"[{args.tag}] config={vars(args)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_recs = read_jsonl("data/train.jsonl"); dev_recs = read_jsonl("data/test.jsonl")
    use_sup = args.supcon > 0                       # 是否用 SupCon
    need_contrastive = use_sup and not args.no_contrastive  # 是否需要加载对比负例数据
    train_ds = ResponseDataset(train_recs, tok, 96, with_contrastive=need_contrastive)
    dev_ds = ResponseDataset(dev_recs, tok, 96, False)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              collate_fn=make_collate(tok, 96, need_contrastive))
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=make_collate(tok, 96, False))

    # 消融①：软标签 or one-hot
    if args.no_soft_label:
        soft = torch.eye(NUM_CLASSES, device=device)            # one-hot
    else:
        soft = torch.tensor(circular_soft_labels(NUM_CLASSES, 1.0, 0.85), device=device)  # 圆形软标签
    # 消融②：类别平衡 or 均匀
    cw = None if args.no_class_balance else effective_number_weights(class_counts(train_recs)).to(device)

    model = ValueClassifier(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); run_ce = run_sc = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if use_sup:                              # 需要对比向量
                logits, z = model(ids, mask, return_proj=True)
                if need_contrastive:                 # 含 Contrastive 难负例
                    c_ids = batch["c_input_ids"].to(device); c_mask = batch["c_attention_mask"].to(device)
                    z_c = F.normalize(model.proj(model.encode(c_ids, c_mask)), dim=-1)
                    z_all = torch.cat([z, z_c], 0)
                    lab_all = torch.cat([labels, labels.new_full((c_ids.size(0),), -1)])
                    sc = supcon_loss(z_all, lab_all, args.temp)
                else:                                # 消融④：只用 consistent，无对比负例
                    sc = supcon_loss(z, labels, args.temp)
            else:                                    # 消融③：无 SupCon
                logits = model(ids, mask); sc = torch.zeros((), device=device)
            ce = soft_label_ce(logits, soft[labels], cw, labels)  # 分类损失
            loss = ce + args.supcon * sc             # 联合损失
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run_ce += ce.item(); run_sc += float(sc)
        m, _, _ = evaluate(model, dev_loader, device)
        if m["macro_f1"] > best:                     # 保存最佳
            best = m["macro_f1"]; torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        print(f"[{args.tag} ep{ep}/{args.epochs}] CE={run_ce/len(train_loader):.4f} "
              f"SupCon={run_sc/len(train_loader):.4f} macroF1={m['macro_f1']:.4f} (best={best:.4f})", flush=True)

    json.dump({"tag": args.tag, "dev_macro_f1": best, "config": vars(args),
               "note": "low-budget ablation (<=5 epoch, undertrained)"},
              open(os.path.join(args.out, "result.json"), "w"), indent=2, ensure_ascii=False)
    print(f">>> ABLATION {args.tag}: dev macroF1={best:.4f}  saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
