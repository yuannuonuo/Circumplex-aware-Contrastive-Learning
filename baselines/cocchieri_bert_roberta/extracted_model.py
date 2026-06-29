# -*- coding: utf-8 -*-
"""
extracted_model.py —— 从 cocchieri 原仓库 models/BERT.ipynb 原样抽取的模型与损失。

内容逐字来自其 notebook（cell 13 的 BERTClass、cell 16 的 loss_fn），仅做一处本任务必需
的适配：分类头输出维度 20（ValueEval 20 类）→ NUM_CLASSES（本任务 19 类）。其余（AutoModel
return_dict=False、pooler→dropout→linear、BCEWithLogitsLoss）保持原样。
"""

import torch
import transformers

OUT_CHANNELS = 768          # 原 notebook 全局变量（base 模型 768）

# ===== 以下 BERTClass 逐字来自 cocchieri BERT.ipynb（仅 20 -> NUM_CLASSES）=====
class BERTClass(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(BERTClass, self).__init__()
        self.l1 = transformers.AutoModel.from_pretrained(model, return_dict=False)
        self.l2 = torch.nn.Dropout(p=0.3)
        self.l3 = torch.nn.Linear(OUT_CHANNELS, num_classes)   # 原为 Linear(OUT_CHANNELS, 20)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


# ===== loss_fn 逐字来自 cocchieri BERT.ipynb（cell 16）=====
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
