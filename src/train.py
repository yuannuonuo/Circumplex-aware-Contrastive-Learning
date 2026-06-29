# -*- coding: utf-8 -*-
"""
train.py —— 训练一个 Track 1 价值分类器（核心训练脚本）。

支持三种可叠加的配置，便于做消融实验：
  --soft_label   ：用"圆形软标签"替代 one-hot 硬标签
  --class_balance：用"有效样本数"做类别平衡加权，应对长尾
  --supcon LAMBDA：叠加监督对比辅助损失，并用对比回答作难负例

示例：
  python src/train.py --model microsoft/deberta-v3-large \
      --soft_label --class_balance --supcon 0.3 --epochs 4
"""

import argparse                                # 命令行参数解析
import os                                       # 文件路径与目录操作
import numpy as np                              # 随机种子/数值
import torch                                    # 训练主框架
from torch.utils.data import DataLoader         # 批数据迭代器
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # 分词器 + 学习率调度
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score  # 评测指标

from values import NUM_CLASSES, circular_soft_labels, ID2LABEL, ID2GROUP  # 标签体系工具
from data import (read_jsonl, ResponseDataset, make_collate, class_counts)  # 数据工具
from model import ValueClassifier, soft_label_ce, supcon_loss              # 模型与损失


def get_device():
    """自动选择计算设备：优先 Apple MPS，其次 CUDA，最后 CPU。"""
    if torch.backends.mps.is_available():       # Apple 芯片的 GPU 后端
        return torch.device("mps")
    if torch.cuda.is_available():               # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")                  # 兜底用 CPU


def effective_number_weights(counts, beta=0.999):
    """按 Cui 等人的"有效样本数"公式，由类别频次算出每类权重（类别平衡）。"""
    eff = 1.0 - torch.pow(beta, counts)         # 有效样本数 = 1 - beta^n
    w = (1.0 - beta) / eff                       # 权重与有效样本数成反比
    return w / w.mean()                          # 归一化，使平均权重约为 1


@torch.no_grad()                                 # 评测不需要梯度
def evaluate(model, loader, device):
    """在给定 DataLoader 上评测，返回指标字典 + 预测 + 金标签。"""
    model.eval()                                 # 切到评测模式（关 dropout）
    preds, golds = [], []                        # 收集预测与真值
    for batch in loader:                         # 遍历每个 batch
        logits = model(batch["input_ids"].to(device),       # 前向得到 logits
                       batch["attention_mask"].to(device))
        preds.append(logits.argmax(-1).cpu())    # 取 argmax 作为预测类别
        golds.append(batch["labels"])            # 收集金标签
    preds = torch.cat(preds).numpy()             # 拼接成 numpy 数组
    golds = torch.cat(golds).numpy()
    macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)  # 主指标：宏 F1
    acc = accuracy_score(golds, preds)           # 准确率
    p = precision_score(golds, preds, average="macro", zero_division=0)  # 宏精确率
    r = recall_score(golds, preds, average="macro", zero_division=0)     # 宏召回率
    # 高阶组准确率：预测是否至少落在正确的"大组"里（用于诊断组内/跨组错误）
    grp = np.mean([ID2GROUP[int(a)] == ID2GROUP[int(b)]
                   for a, b in zip(preds, golds)])
    return {"macro_f1": macro_f1, "acc": acc, "macro_p": p, "macro_r": r,
            "group_acc": grp}, preds, golds


def main():
    """解析参数 -> 准备数据/模型 -> 训练若干 epoch -> 按 dev 宏 F1 保存最优。"""
    ap = argparse.ArgumentParser()              # 构造参数解析器
    ap.add_argument("--model", default="microsoft/deberta-v3-base")  # 编码器名
    ap.add_argument("--train", default="data/train.jsonl")           # 训练集路径
    ap.add_argument("--dev", default="data/test.jsonl")               # 验证集路径
    ap.add_argument("--out", default="outputs/model")                # 输出目录
    ap.add_argument("--epochs", type=int, default=4)                 # 训练轮数
    ap.add_argument("--bs", type=int, default=32)                    # 批大小
    ap.add_argument("--lr", type=float, default=2e-5)                # 学习率
    ap.add_argument("--max_len", type=int, default=96)               # 最大序列长度
    ap.add_argument("--warmup", type=float, default=0.06)            # warmup 步数占比
    ap.add_argument("--soft_label", action="store_true")            # 是否用圆形软标签
    ap.add_argument("--sigma", type=float, default=1.0)             # 软标签高斯带宽
    ap.add_argument("--peak", type=float, default=0.85)            # 软标签真实类别质量
    ap.add_argument("--class_balance", action="store_true")        # 是否类别平衡
    ap.add_argument("--supcon", type=float, default=0.0)           # SupCon 权重（0 表示不用）
    ap.add_argument("--temp", type=float, default=0.1)            # SupCon 温度
    ap.add_argument("--seed", type=int, default=42)              # 随机种子
    ap.add_argument("--init_encoder", default=None,
                    help="可选：预训练好的 encoder 权重路径（如 FULCRA 预训练产物）")
    args = ap.parse_args()                       # 解析命令行

    torch.manual_seed(args.seed); np.random.seed(args.seed)  # 固定随机种子，保证可复现
    device = get_device()                         # 选设备
    print(f"device={device}  config={vars(args)}")  # 打印设备与全部配置

    tok = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
    train_recs = read_jsonl(args.train)              # 读训练样本
    dev_recs = read_jsonl(args.dev)                  # 读验证样本

    use_contrastive = args.supcon > 0                # 是否需要对比负例（SupCon 开启时为 True）
    train_ds = ResponseDataset(train_recs, tok, args.max_len, use_contrastive)  # 训练集
    dev_ds = ResponseDataset(dev_recs, tok, args.max_len, False)                # 验证集（永不带对比）
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,        # 训练用 loader（打乱）
                              collate_fn=make_collate(tok, args.max_len, use_contrastive))
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False,                # 验证用 loader（不打乱）
                            collate_fn=make_collate(tok, args.max_len, False))

    model = ValueClassifier(args.model).to(device)   # 构造模型并搬到设备
    if args.init_encoder:                            # 若指定了预训练 encoder（FULCRA 实验用）
        sd = torch.load(args.init_encoder, map_location=device)        # 读权重
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)  # 非严格加载
        print(f"loaded pretrained encoder from {args.init_encoder} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # 软目标矩阵：开启 soft_label 用圆形软标签，否则退化为单位阵（等价 one-hot）
    if args.soft_label:
        soft = torch.tensor(circular_soft_labels(NUM_CLASSES, args.sigma, args.peak),
                            device=device)
    else:
        soft = torch.eye(NUM_CLASSES, device=device)

    cw = None                                        # 类别权重，默认无
    if args.class_balance:                           # 若启用类别平衡
        cw = effective_number_weights(class_counts(train_recs)).to(device)  # 由训练集频次算权重

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)  # AdamW 优化器
    total_steps = len(train_loader) * args.epochs    # 总训练步数
    sched = get_linear_schedule_with_warmup(         # 线性 warmup + 线性衰减调度
        opt, int(total_steps * args.warmup), total_steps)

    best_f1, best_metrics = -1.0, None               # 记录最优 dev 宏 F1 及其指标
    os.makedirs(args.out, exist_ok=True)             # 创建输出目录
    for ep in range(1, args.epochs + 1):             # 逐 epoch 训练
        model.train()                                # 切到训练模式
        running = 0.0                                # 累计损失
        for step, batch in enumerate(train_loader):  # 遍历每个 batch
            ids = batch["input_ids"].to(device)       # token id
            mask = batch["attention_mask"].to(device)  # 注意力掩码
            labels = batch["labels"].to(device)       # 标签
            target_dist = soft[labels]                # 按标签查软目标分布 [B, C]

            if use_contrastive:                       # === SupCon 联合训练分支 ===
                logits, z = model(ids, mask, return_proj=True)  # 同时拿 logits 与对比向量
                c_ids = batch["c_input_ids"].to(device)         # 对比回答 token id
                c_mask = batch["c_attention_mask"].to(device)   # 对比回答掩码
                import torch.nn.functional as F                 # 局部导入做归一化
                z_c = F.normalize(model.proj(model.encode(c_ids, c_mask)), dim=-1)  # 对比回答的投影向量
                z_all = torch.cat([z, z_c], 0)                  # 拼接正样本与难负例
                lab_all = torch.cat([labels, labels.new_full((c_ids.size(0),), -1)])  # 对比回答标签设 -1
                cl = soft_label_ce(logits, target_dist, cw, labels)  # 主分类损失
                sc = supcon_loss(z_all, lab_all, args.temp)          # 对比损失
                loss = cl + args.supcon * sc                         # 总损失 = CE + λ*SupCon
            else:                                     # === 普通分类分支 ===
                logits = model(ids, mask)             # 只前向得到 logits
                loss = soft_label_ce(logits, target_dist, cw, labels)  # 软标签交叉熵

            opt.zero_grad()                           # 梯度清零
            loss.backward()                           # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防爆炸
            opt.step(); sched.step()                  # 更新参数与学习率
            running += loss.item()                    # 累加损失
            if (step + 1) % 20 == 0:                  # 每 20 步打印一次平均损失
                print(f"  ep{ep} step{step+1}/{len(train_loader)} "
                      f"loss={running/(step+1):.4f}", flush=True)

        metrics, _, _ = evaluate(model, dev_loader, device)  # 每个 epoch 末在 dev 上评测
        print(f"[epoch {ep}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
              flush=True)
        if metrics["macro_f1"] > best_f1:             # 若刷新了最优宏 F1
            best_f1 = metrics["macro_f1"]; best_metrics = metrics  # 更新记录
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))  # 保存最优权重
            with open(os.path.join(args.out, "config.txt"), "w") as f:        # 同时存配置与指标
                f.write(str(vars(args)) + "\n" + str(metrics) + "\n")

    print(f"\nBEST dev macro-F1={best_f1:.4f}  metrics={best_metrics}")  # 训练结束打印最优结果


if __name__ == "__main__":
    main()                                            # 作为脚本运行时进入主流程
