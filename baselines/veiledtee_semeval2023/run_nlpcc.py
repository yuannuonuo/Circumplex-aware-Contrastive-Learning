# -*- coding: utf-8 -*-
"""
run_nlpcc.py —— 把 veiledtee 原仓库的方法（BERT 嵌入 + XGBoost）适配为本任务分类基线。

veiledtee 的监督方法（Embed.py + XGBoostUncased.py）= 用 bert-base-uncased 取句向量，
再用 XGBoost 分类。其原始 Embed 代码对每个句子都重载一次 BertModel（4000+ 句不可行），
故这里**保留其方法本体**（bert-base-uncased 句向量 + XGBoost），高效地批量取向量；
任务改为本任务的单标签 19 类多分类（XGBoost objective=multi:softmax）。
"""

import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from transformers import AutoTokenizer, AutoModel
from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID, NUM_CLASSES
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def bert_embed(texts, tok, model, bs=64, max_len=96):
    """veiledtee 风格的 bert-base-uncased 句向量（这里取最后一层 mean pooling，批量高效）。"""
    vecs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEVICE)
        hs = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        vecs.append(((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(vecs)


def main():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()
    tr = read_jsonl("data/train.jsonl"); dv = read_jsonl("data/test.jsonl")
    Xtr = bert_embed([r[CONSISTENT_KEY] for r in tr], tok, model)
    ytr = np.array([LABEL2ID[r["Value"]] for r in tr])
    Xdv = bert_embed([r[CONSISTENT_KEY] for r in dv], tok, model)
    ydv = np.array([LABEL2ID[r["Value"]] for r in dv])
    print(f"device={DEVICE} embed dim={Xtr.shape[1]} train={len(Xtr)} dev={len(Xdv)}", flush=True)

    # veiledtee 的分类器：XGBoost（改为多分类单标签）
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.3, subsample=0.8,
                            tree_method="hist", n_jobs=8, num_class=NUM_CLASSES,
                            objective="multi:softmax")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xdv)
    f1 = f1_score(ydv, pred, average="macro", zero_division=0); acc = accuracy_score(ydv, pred)
    print(f">>> veiledtee (BERT嵌入 + XGBoost) 本任务 dev macroF1={f1:.4f} acc={acc:.4f}", flush=True)
    os.makedirs("baselines/veiledtee_semeval2023/nlpcc_out", exist_ok=True)
    clf.save_model("baselines/veiledtee_semeval2023/nlpcc_out/xgboost_model.json")  # 保存 XGBoost 模型参数
    json.dump({"repo": "veiledtee (BERT-base-uncased 句向量 + XGBoost, 方法本体)",
               "task": "NLPCC2026 Track1 单标签19类", "dev_macro_f1": float(f1), "dev_acc": float(acc)},
              open("baselines/veiledtee_semeval2023/nlpcc_out/result.json", "w"), indent=2, ensure_ascii=False)
    print("VEILEDTEE_NLPCC_DONE")


if __name__ == "__main__":
    main()
