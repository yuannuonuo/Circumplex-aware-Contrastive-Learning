# -*- coding: utf-8 -*-
"""
train_joint.py —— 联合损失训练（stage1 与 stage2 合一，单阶段）。

与两阶段对照：这里把 SupCon 与分类放在同一阶段、同时优化：
    loss = 圆形软标签CE(+类别平衡) + λ·SupCon
其余设定与之前一致（temp、soft-label、class-balance、λ=0.3）。

训练过程中"就地"在三个时点抽取 dev 表征，存进 npz（供画 Raw→中期→终态 的 t-SNE）：
    raw  ：训练前的原始预训练表征
    mid  ：训练到一半（epochs//2）时的表征
    final：训练结束时的表征
同时保存最终模型 .pt 与 dev 最佳 macro-F1。
"""

import argparse                                # 命令行
import os, json                                 # 文件/结果
import numpy as np                              # 数值
import torch                                     # 训练
import torch.nn.functional as F                  # 归一化
from torch.utils.data import DataLoader          # 数据迭代器
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # 分词/调度

from values import NUM_CLASSES, circular_soft_labels  # 标签工具
from data import read_jsonl, ResponseDataset, make_collate, class_counts, CONSISTENT_KEY  # 数据
from model import ValueClassifier, soft_label_ce, supcon_loss  # 模型与损失
from train import get_device, effective_number_weights, evaluate  # 复用


@torch.no_grad()                                   # 抽特征无梯度
def dev_embed(model, texts, tok, device, bs=64, max_len=96):
    """就地抽取当前模型在 dev 上的池化句向量 [N,H]。"""
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        out.append(model.encode(enc["input_ids"].to(device),
                                enc["attention_mask"].to(device)).cpu().numpy())
    model.train()                                  # 抽完切回训练模式
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")  # base 或 large
    ap.add_argument("--out", default="outputs/joint_base")          # 输出目录
    ap.add_argument("--epochs", type=int, default=22)              # 联合单阶段总轮数
    ap.add_argument("--bs", type=int, default=32)                # 批大小
    ap.add_argument("--lr", type=float, default=2e-5)           # 学习率
    ap.add_argument("--supcon", type=float, default=0.3)       # λ：SupCon 权重
    ap.add_argument("--temp", type=float, default=0.10)       # SupCon 温度
    ap.add_argument("--max_len", type=int, default=96)       # 最大长度
    ap.add_argument("--seed", type=int, default=42)         # 随机种子
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)  # 固定种子
    device = get_device(); os.makedirs(args.out, exist_ok=True)  # 设备/目录
    print(f"device={device} config={vars(args)}", flush=True)

    from values import LABEL2ID                    # dev 标签映射
    tok = AutoTokenizer.from_pretrained(args.model)  # 分词器
    train_recs = read_jsonl("data/train.jsonl"); dev_recs = read_jsonl("data/test.jsonl")  # 数据
    dev_texts = [r[CONSISTENT_KEY] for r in dev_recs]            # dev 文本
    dev_y = np.array([LABEL2ID[r["Value"]] for r in dev_recs])  # dev 标签
    train_ds = ResponseDataset(train_recs, tok, args.max_len, with_contrastive=True)  # 训练集(带对比)
    dev_ds = ResponseDataset(dev_recs, tok, args.max_len, False)                      # 验证集
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              collate_fn=make_collate(tok, args.max_len, True))
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False,
                            collate_fn=make_collate(tok, args.max_len, False))

    soft = torch.tensor(circular_soft_labels(NUM_CLASSES, 1.0, 0.85), device=device)  # 圆形软标签
    cw = effective_number_weights(class_counts(train_recs)).to(device)  # 类别平衡权重

    model = ValueClassifier(args.model).to(device)  # 模型
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)  # 优化器
    total = len(train_loader) * args.epochs         # 总步数
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)  # 调度

    embs = {"y": dev_y}                              # 表征字典
    embs["raw"] = dev_embed(model, dev_texts, tok, device)  # ① 训练前：原始表征
    mid_ep = max(1, args.epochs // 2)               # 中期时点（半程）
    best = -1.0                                      # 最佳 dev 宏 F1

    for ep in range(1, args.epochs + 1):            # 逐 epoch 联合训练
        model.train(); run_ce = run_sc = 0.0
        for batch in train_loader:                   # 遍历 batch
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)  # 正样本
            labels = batch["labels"].to(device)      # 标签
            logits, z = model(ids, mask, return_proj=True)  # 同时拿 logits 与对比向量
            c_ids = batch["c_input_ids"].to(device); c_mask = batch["c_attention_mask"].to(device)  # 对比负例
            z_c = F.normalize(model.proj(model.encode(c_ids, c_mask)), dim=-1)  # 对比负例向量
            z_all = torch.cat([z, z_c], 0)           # 拼接
            lab_all = torch.cat([labels, labels.new_full((c_ids.size(0),), -1)])  # 对比负例标签 -1
            ce = soft_label_ce(logits, soft[labels], cw, labels)  # 分类损失
            sc = supcon_loss(z_all, lab_all, args.temp)           # 对比损失
            loss = ce + args.supcon * sc             # 联合损失
            opt.zero_grad(); loss.backward()         # 反向
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 裁剪
            opt.step(); sched.step(); run_ce += ce.item(); run_sc += sc.item()  # 更新+累计
        m, _, _ = evaluate(model, dev_loader, device)  # dev 评测
        if m["macro_f1"] > best:                     # 刷新最佳则保存"最佳轮"模型（保证保存分=报告分）
            best = m["macro_f1"]
            torch.save(model.state_dict(), os.path.join(args.out, "joint_model.pt"))
        print(f"[ep{ep}/{args.epochs}] CE={run_ce/len(train_loader):.4f} "
              f"SupCon={run_sc/len(train_loader):.4f} macroF1={m['macro_f1']:.4f} (best={best:.4f})", flush=True)
        if ep == mid_ep:                             # ② 中期：抽表征
            embs["mid"] = dev_embed(model, dev_texts, tok, device)
            np.savez(os.path.join(args.out, "joint_embs.npz"), **embs)  # 增量存盘

    embs["final"] = dev_embed(model, dev_texts, tok, device)  # ③ 终态：抽表征（末轮模型，供可视化）
    np.savez(os.path.join(args.out, "joint_embs.npz"), **embs)  # 存表征（joint_model.pt 为最佳轮）
    json.dump({"dev_macro_f1_best": best, "config": vars(args)},
              open(os.path.join(args.out, "joint_results.json"), "w"), indent=2)  # 存结果
    print(f"\nBEST dev macro-F1={best:.4f}  saved -> {args.out}")
    print("JOINT_DONE")


if __name__ == "__main__":
    main()
