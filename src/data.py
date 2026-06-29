# -*- coding: utf-8 -*-
"""
data.py —— Track 1 的数据加载。

关键约束：测试阶段只有"回答文本"可用（test/track1.jsonl 里只有
"Consistent Value Response" 字段）。所以分类器的输入永远是"单段回答字符串"。
Scenario / Question / Contrastive Response 只能在训练阶段用，绝不进推理主干，
否则会造成训练-测试输入分布不一致而掉分。
"""

import json                                   # 读取 jsonl 用
from typing import List, Dict                 # 类型注解

import torch                                  # 张量与 Dataset
from torch.utils.data import Dataset          # PyTorch 数据集基类

from values import LABEL2ID                    # 标签字符串 -> id 的映射

CONSISTENT_KEY = "Consistent Value Response"   # "对齐良好的回答"字段名（训练目标 & 测试输入）
CONTRASTIVE_KEY = "Contrastive Response"       # "对齐较差的回答"字段名（训练期当难负例用）


def read_jsonl(path: str) -> List[Dict]:
    """读取 jsonl 文件，每行解析为一个 dict，返回 dict 列表。"""
    with open(path, encoding="utf-8") as f:            # 以 UTF-8 打开文件
        return [json.loads(line) for line in f if line.strip()]  # 逐行解析，跳过空行


class ResponseDataset(Dataset):
    """单段回答 -> 19 类标签的数据集，输入形态与测试阶段完全一致。"""

    def __init__(self, records: List[Dict], tokenizer, max_len: int = 96,
                 with_contrastive: bool = False):
        self.tok = tokenizer                  # 分词器（仅在 collate 里用，这里存着备用）
        self.max_len = max_len                # 最大截断长度
        self.with_contrastive = with_contrastive  # 是否同时返回对比负例（对比学习时为 True）
        self.records = records                # 原始样本列表

    def __len__(self):
        return len(self.records)              # 样本数量

    def __getitem__(self, i):
        r = self.records[i]                   # 取第 i 条原始样本
        item = {"text": r[CONSISTENT_KEY]}    # 主输入：对齐良好的回答文本
        if "Value" in r:                      # 训练/验证集才有金标签
            item["label"] = LABEL2ID[r["Value"]]      # 把标签字符串转成 id
        # 对比回答：同情境、同目标价值但对齐更差，天然是对比学习的难负例
        if self.with_contrastive and CONTRASTIVE_KEY in r:
            item["contrastive_text"] = r[CONTRASTIVE_KEY]
        return item                           # 返回一个轻量 dict（真正的分词在 collate 里做）


def make_collate(tokenizer, max_len: int = 96, with_contrastive: bool = False):
    """返回一个 collate 函数：把一批样本动态 padding 并分词成张量。"""

    def collate(batch):                       # batch 是 __getitem__ 返回的 dict 列表
        texts = [b["text"] for b in batch]    # 取出该批所有主文本
        # 对主文本批量分词 + 动态 padding 到批内最长，返回 PyTorch 张量
        enc = tokenizer(texts, truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        out = {"input_ids": enc["input_ids"],          # token id 张量 [B, L]
               "attention_mask": enc["attention_mask"]}  # 注意力掩码 [B, L]
        if "label" in batch[0]:               # 若有标签则一并打包
            out["labels"] = torch.tensor([b["label"] for b in batch],
                                         dtype=torch.long)
        # 若需要对比负例，则对 contrastive_text 同样分词，键名加 c_ 前缀
        if with_contrastive and "contrastive_text" in batch[0]:
            cenc = tokenizer([b["contrastive_text"] for b in batch],
                             truncation=True, max_length=max_len,
                             padding=True, return_tensors="pt")
            out["c_input_ids"] = cenc["input_ids"]
            out["c_attention_mask"] = cenc["attention_mask"]
        return out                            # 返回可直接喂模型的张量字典

    return collate                            # 把闭包返回给 DataLoader 使用


def class_counts(records: List[Dict], num_classes: int = 19) -> torch.Tensor:
    """统计每个类别在样本里出现的次数，返回长度 19 的张量（用于类别平衡）。"""
    counts = torch.zeros(num_classes)         # 初始化全 0 计数向量
    for r in records:                         # 遍历每条样本
        if "Value" in r:                      # 有标签才统计
            counts[LABEL2ID[r["Value"]]] += 1  # 对应类别计数 +1
    return counts                             # 返回类别频次向量
