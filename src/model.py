# -*- coding: utf-8 -*-
"""
model.py —— 编码器分类模型 + 两类损失函数。

模型结构：预训练编码器（DeBERTa 等）做特征提取，接一个 19 路分类头；另外挂一个
投影头（projection head），专门给"监督对比学习 SupCon"用。
两类损失：
  - soft_label_ce：对"软目标分布"的交叉熵，支持类别平衡加权（圆形软标签在此处用）。
  - supcon_loss  ：监督对比损失，把同类拉近、异类推远，支持"只当负例"的难负例。
"""

import torch                                  # 张量
import torch.nn as nn                         # 网络层
import torch.nn.functional as F               # 函数式算子（softmax/normalize 等）
from transformers import AutoModel            # 自动加载任意 HuggingFace 编码器

from values import NUM_CLASSES                 # 类别数 = 19


class ValueClassifier(nn.Module):
    """价值分类器：共享编码器 + 分类头 + 对比投影头。"""

    def __init__(self, model_name: str, num_classes: int = NUM_CLASSES,
                 proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()                              # 初始化父类 nn.Module
        self.encoder = AutoModel.from_pretrained(model_name)  # 加载预训练编码器
        h = self.encoder.config.hidden_size             # 编码器隐藏维度（如 768/1024）
        self.dropout = nn.Dropout(dropout)              # 分类头前的 dropout
        self.classifier = nn.Linear(h, num_classes)     # 线性分类头：h -> 19
        # 投影头：仅用于对比目标，把池化表征映射到低维并配 ReLU
        self.proj = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, proj_dim)
        )

    def encode(self, input_ids, attention_mask):
        """把一批 token 编码成"句向量"（对非 padding token 做 mean pooling）。"""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  # 过编码器
        mask = attention_mask.unsqueeze(-1).float()     # 掩码扩一维 [B, L, 1] 便于广播
        summed = (out.last_hidden_state * mask).sum(1)   # 仅对有效 token 的隐藏态求和 [B, H]
        counts = mask.sum(1).clamp(min=1e-9)             # 有效 token 数（防止除零）
        return summed / counts                           # 求平均 = mean pooling 句向量

    def forward(self, input_ids, attention_mask, return_proj: bool = False):
        """前向：返回分类 logits；若 return_proj=True 再额外返回归一化后的对比向量。"""
        pooled = self.encode(input_ids, attention_mask)  # 得到句向量
        logits = self.classifier(self.dropout(pooled))   # 过 dropout 与分类头得到 19 路 logits
        if return_proj:                                  # 对比学习路径
            z = F.normalize(self.proj(pooled), dim=-1)    # 投影后 L2 归一化（落到单位球面）
            return logits, z                              # 同时返回 logits 与对比向量
        return logits                                     # 普通分类只返回 logits


def soft_label_ce(logits, target_dist, class_weight=None, true_labels=None):
    """对"软目标分布"的交叉熵；可按样本真实类别施加类别平衡权重。

    参数：
        logits      ：模型输出 [B, C]。
        target_dist ：软目标分布 [B, C]（one-hot 或圆形软标签查表得到）。
        class_weight：每类权重 [C]，按样本真实类别取出后逐样本相乘 = 类别平衡。
        true_labels ：每个样本的真实类别 id [B]，用于查 class_weight。
    """
    logp = F.log_softmax(logits, dim=-1)             # 对 logits 取 log-softmax
    loss = -(target_dist * logp).sum(dim=-1)          # 交叉熵 = -Σ 目标分布 * log 预测，得每样本损失 [B]
    if class_weight is not None and true_labels is not None:  # 若启用类别平衡
        loss = loss * class_weight[true_labels]       # 按真实类别给该样本损失加权
    return loss.mean()                                # 返回批均值


def supcon_loss(z, labels, temperature: float = 0.1):
    """监督对比损失（SupCon）。labels == -1 的样本"只当负例"
    （永不作为锚点、也永不作为别人的正例），用于注入对比难负例。

    参数：
        z          ：已 L2 归一化的对比向量 [B, D]。
        labels     ：每个样本的类别 id [B]；-1 表示"仅负例"。
        temperature：温度系数，越小则对比越"尖锐"。
    """
    device = z.device                                 # 当前设备（mps/cuda/cpu）
    sim = z @ z.t() / temperature                     # 两两余弦相似度 / 温度 -> 相似度矩阵 [B, B]
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # 每行减去最大值，数值稳定（不影响梯度）
    logits_mask = ~torch.eye(z.size(0), dtype=torch.bool, device=device)  # 对角线（自己和自己）置 False
    exp_sim = torch.exp(sim) * logits_mask            # 指数化并屏蔽自身项
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)  # log-softmax 的对比版本

    lab = labels.view(-1, 1)                           # 标签列向量 [B, 1] 便于广播比较
    pos_mask = (lab == lab.t()) & (lab != -1) & logits_mask  # 正例掩码：同类、非 -1、且不是自身
    pos_counts = pos_mask.sum(1)                       # 每个锚点的正例个数
    valid = pos_counts > 0                             # 至少有 1 个正例的锚点才有效
    if valid.sum() == 0:                               # 若整批都没有正例对
        return z.new_tensor(0.0)                       # 返回 0 损失（不产生梯度）
    # 对每个有效锚点：平均其所有正例的 log_prob，取负号并对锚点求均值
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()                   # SupCon 损失


def weighted_supcon_loss(z, labels, ring_D, temperature: float = 0.1, beta: float = 0.0):
    """环距加权的监督对比损失：按"环上距离"给负例加权。

    思路：误差集中在"组内/环上相邻"的细类之间，所以让对比阶段**更狠地推开近邻负例**、
    对"跨大组的远负例"只用轻力（已经好分）。距离近(d=1)的负例权重 = 1+beta，距离最远
    (d=dmax)的权重 = 1。beta=0 时本函数严格退化为标准 supcon_loss。

    参数：
        ring_D     ：[C,C] 类间环形距离矩阵（long/float 张量，与 z 同设备）。
        temperature：温度。
        beta       ：组内加权强度；越大越聚焦推开近邻负例。
    label==-1 的构造难负例视为"最难"，给最大权重 1+beta。
    """
    device = z.device
    B = z.size(0)
    sim = z @ z.t() / temperature                      # 相似度/温度 [B,B]
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # 数值稳定
    offdiag = ~torch.eye(B, dtype=torch.bool, device=device)  # 屏蔽自身
    lab = labels.view(-1, 1)                            # 标签列向量
    pos_mask = (lab == lab.t()) & (lab != -1) & offdiag  # 正例掩码（同类、非 -1、非自身）
    pos_counts = pos_mask.sum(1)                        # 每锚点正例数
    valid = pos_counts > 0                              # 有正例的锚点才有效
    if valid.sum() == 0:                                # 整批无正例对
        return z.new_tensor(0.0)

    dmax = float(ring_D.max().item())                   # 最大环距（19 类 = 9）
    labA = labels.clamp(min=0)                          # 把 -1 暂映射到 0 以便索引
    Dbatch = ring_D[labA][:, labA].float()              # [B,B] 两两类别的环距
    # 负例权重：近(d=1)->1+beta，远(d=dmax)->1，随距离线性递减
    W = 1.0 + beta * (dmax - Dbatch) / max(dmax - 1.0, 1.0)
    neg_minus1 = (labels == -1).view(1, -1).expand(B, B)  # 列为构造难负例(-1)的位置
    W = torch.where(neg_minus1, torch.full_like(W, 1.0 + beta), W)  # 难负例给最大权重
    W = torch.where(pos_mask, torch.ones_like(W), W)    # 正例权重置 1（分母里按 1 计）

    exp_sim = torch.exp(sim) * offdiag                  # 指数化并屏蔽自身
    denom = (W * exp_sim).sum(1, keepdim=True) + 1e-12  # 加权分母（近邻负例占更大份额）
    log_prob = sim - torch.log(denom)                   # 加权 log-softmax
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid] / pos_counts[valid]  # 每锚点正例均值
    return -mean_log_prob_pos.mean()                    # 加权 SupCon 损失
