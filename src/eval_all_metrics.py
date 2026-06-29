# -*- coding: utf-8 -*-
"""
eval_all_metrics.py —— 加载所有实验的已存模型，在 dev 上统一输出
Accuracy / Macro-Precision / Macro-Recall / Macro-F1。结果存 outputs/all_metrics.{json,csv}。
覆盖：本项目(joint/两阶段)、消融、欠训对照、朴素 one-hot、经典基线、神经基线、5 个公开仓库、Kiesel 官方。
"""
import os, sys, json, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import joblib

from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID, NUM_CLASSES, ID2LABEL
from model import ValueClassifier
from train import get_device

DEV = get_device()
dv = read_jsonl("data/test.jsonl")
X = [r[CONSISTENT_KEY] for r in dv]
Y = np.array([LABEL2ID[r["Value"]] for r in dv])
ROWS = []


def metrics(pred):
    return (accuracy_score(Y, pred),
            precision_score(Y, pred, average="macro", zero_division=0),
            recall_score(Y, pred, average="macro", zero_division=0),
            f1_score(Y, pred, average="macro", zero_division=0))


def add(group, name, pred):
    a, p, r, f = metrics(pred)
    ROWS.append({"group": group, "model": name, "accuracy": round(a, 4),
                 "macro_precision": round(p, 4), "macro_recall": round(r, 4), "macro_f1": round(f, 4)})
    print(f"  [{name:30s}] Acc={a:.4f}  P={p:.4f}  R={r:.4f}  F1={f:.4f}", flush=True)


@torch.no_grad()
def vc_pred(model_name, ckpt):
    tok = AutoTokenizer.from_pretrained(model_name)
    m = ValueClassifier(model_name).to(DEV); m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.eval()
    out = []
    for i in range(0, len(X), 64):
        e = tok(X[i:i+64], truncation=True, max_length=96, padding=True, return_tensors="pt")
        out.append(m(e["input_ids"].to(DEV), e["attention_mask"].to(DEV)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def mean_embed(model_name, enc, max_len=96):
    tok = AutoTokenizer.from_pretrained(model_name); out = []
    for i in range(0, len(X), 64):
        e = tok(X[i:i+64], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEV)
        h = enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h*m).sum(1)/m.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(out)


print("=== 本项目 + 消融 + 对照 + 神经基线（ValueClassifier）===")
VC = [
    ("本项目方法", "joint_large", "microsoft/deberta-v3-large", "outputs/joint_large/joint_model.pt"),
    ("本项目方法", "joint_base", "microsoft/deberta-v3-base", "outputs/joint_base/joint_model.pt"),
    ("本项目方法", "twostage_large", "microsoft/deberta-v3-large", "outputs/twostage_large/model.pt"),
    ("本项目方法", "twostage_base", "microsoft/deberta-v3-base", "outputs/twostage_base/model.pt"),
    ("消融", "no_soft", "microsoft/deberta-v3-base", "outputs/ablation/no_soft/best.pt"),
    ("消融", "no_cb", "microsoft/deberta-v3-base", "outputs/ablation/no_cb/best.pt"),
    ("消融", "no_contrastive", "microsoft/deberta-v3-base", "outputs/ablation/no_contrastive/best.pt"),
    ("消融", "no_supcon", "microsoft/deberta-v3-base", "outputs/ablation/no_supcon/best.pt"),
    ("对照", "onehot_simple", "microsoft/deberta-v3-base", "outputs/onehot_simple/best.pt"),
    ("神经基线", "neural_bert-base", "bert-base-uncased", "baselines/neural_models/bert-base-uncased.pt"),
    ("神经基线", "neural_roberta-base", "roberta-base", "baselines/neural_models/roberta-base.pt"),
]
for g, n, mn, ck in VC:
    try: add(g, n, vc_pred(mn, ck))
    except Exception as e: print(f"  {n} 跳过: {e}")

print("=== 经典基线（TF-IDF + joblib）===")
try:
    vec = joblib.load("baselines/classical_models/tfidf_vectorizer.joblib"); Fdv = vec.transform(X)
    for nm in ["tfidf_linsvc", "tfidf_logreg", "tfidf_xgboost"]:
        add("经典基线", nm, joblib.load(f"baselines/classical_models/{nm}.joblib").predict(Fdv))
except Exception as e: print("  经典跳过:", e)

print("=== 公开仓库基线 ===")
try:  # su0315
    sys.path.insert(0, "baselines/su0315_hvd/model")
    from model_baseline import BaselineModel
    m = BaselineModel(19, None, None, None).to(DEV)
    m.load_state_dict(torch.load("baselines/su0315_hvd/nlpcc_out/model.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        p = np.concatenate([m(X[i:i+32]).argmax(-1).cpu().numpy() for i in range(0, len(X), 32)])
    add("公开仓库", "su0315 BaselineModel(BERT)", p)
except Exception as e: print("  su0315 跳过:", e)

try:  # cocchieri
    sys.path.insert(0, "baselines/cocchieri_bert_roberta")
    from extracted_model import BERTClass
    tok = AutoTokenizer.from_pretrained("roberta-base")
    m = BERTClass("roberta-base", 19).to(DEV)
    m.load_state_dict(torch.load("baselines/cocchieri_bert_roberta/nlpcc_out/model.pt", map_location=DEV)); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            e = tok(X[i:i+32], truncation=True, max_length=96, padding="max_length", return_tensors="pt")
            ids = e["input_ids"].to(DEV); mask = e["attention_mask"].to(DEV)
            out.append(m(ids, mask, torch.zeros_like(ids)).argmax(-1).cpu().numpy())
    add("公开仓库", "cocchieri BERTClass(roberta)", np.concatenate(out))
except Exception as e: print("  cocchieri 跳过:", e)

try:  # adam_smith
    import torch.nn as nn
    sys.path.insert(0, "baselines/adam_smith")
    from models.BertFineTunerPl import BertFineTunerPl
    from values import ID2LABEL as I2L, NUM_CLASSES as NC
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    params = {"MODEL_PATH": "microsoft/deberta-v3-base", "EMBEDDING": "CLS + MEAN",
              "HIDDEN_LAYERS": None, "DROPOUT": 0.1, "CRITERION": [nn.BCEWithLogitsLoss()]}
    m = BertFineTunerPl(NC, params, [I2L[i] for i in range(NC)], 10, 1).to(DEV)
    m.load_state_dict(torch.load("baselines/adam_smith/nlpcc_out/model.pt", map_location=DEV)); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            e = tok(X[i:i+32], truncation=True, max_length=96, padding=True, return_tensors="pt")
            _, o = m(e["input_ids"].to(DEV), e["attention_mask"].to(DEV)); out.append(o.argmax(-1).cpu().numpy())
    add("公开仓库", "adam_smith BertFineTunerPl(DeBERTa)", np.concatenate(out))
except Exception as e: print("  adam_smith 跳过:", e)

try:  # veiledtee
    import xgboost as xgb
    enc = AutoModel.from_pretrained("bert-base-uncased").to(DEV).eval()
    Fb = mean_embed("bert-base-uncased", enc)
    bst = xgb.XGBClassifier(); bst.load_model("baselines/veiledtee_semeval2023/nlpcc_out/xgboost_model.json")
    add("公开仓库", "veiledtee BERT-embed+XGBoost", bst.predict(Fb))
except Exception as e: print("  veiledtee 跳过:", e)

try:  # value_fulcra
    enc = AutoModel.from_pretrained("bert-base-uncased").to(DEV)
    enc.load_state_dict(torch.load("baselines/value_fulcra/nlpcc_out/fulcra_encoder.pt", map_location=DEV)); enc.eval()
    Ff = mean_embed("bert-base-uncased", enc)
    lr = joblib.load("baselines/value_fulcra/nlpcc_out/logreg.joblib")
    add("公开仓库", "value_fulcra FULCRA+probe", lr.predict(Ff))
except Exception as e: print("  value_fulcra 跳过:", e)

print("=== Kiesel'22 官方 baseline ===")
try:  # kiesel BERT
    tok = AutoTokenizer.from_pretrained("baselines/kiesel_acl22/nlpcc_out/bert")
    m = AutoModelForSequenceClassification.from_pretrained("baselines/kiesel_acl22/nlpcc_out/bert").to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            e = tok(X[i:i+64], truncation=True, max_length=96, padding=True, return_tensors="pt").to(DEV)
            out.append(m(**e).logits.argmax(-1).cpu().numpy())
    add("Kiesel官方", "kiesel BERT(BCE)", np.concatenate(out))
except Exception as e: print("  kiesel BERT 跳过:", e)

try:  # kiesel SVM
    vec = joblib.load("baselines/kiesel_acl22/nlpcc_out/svm_vectorizer.joblib")
    clf = joblib.load("baselines/kiesel_acl22/nlpcc_out/svm_model.joblib")
    add("Kiesel官方", "kiesel SVM(C=18)", clf.predict(vec.transform(X)))
except Exception as e: print("  kiesel SVM 跳过:", e)

# 存表
json.dump(ROWS, open("outputs/all_metrics.json", "w"), indent=2, ensure_ascii=False)
with open("outputs/all_metrics.csv", "w") as f:
    f.write("group,model,accuracy,macro_precision,macro_recall,macro_f1\n")
    for r in ROWS:
        f.write(f"{r['group']},{r['model']},{r['accuracy']},{r['macro_precision']},{r['macro_recall']},{r['macro_f1']}\n")
print(f"\n=== 汇总（{len(ROWS)} 个模型，按 macro-F1 降序）===")
print(f"{'模型':32s}{'Acc':>8s}{'Macro-P':>9s}{'Macro-R':>9s}{'Macro-F1':>10s}")
for r in sorted(ROWS, key=lambda x: -x["macro_f1"]):
    print(f"{r['model']:32s}{r['accuracy']:8.4f}{r['macro_precision']:9.4f}{r['macro_recall']:9.4f}{r['macro_f1']:10.4f}")
print("ALL_METRICS_DONE  saved -> outputs/all_metrics.{json,csv}")
