# -*- coding: utf-8 -*-
"""
eval_saved.py —— 加载所有已保存的基线模型参数，在 dev 上复算 macro-F1（不重训）。

覆盖：经典（TF-IDF+joblib）、标准神经（ValueClassifier .pt）、5 个公开仓库
（su0315/cocchieri/adam_smith 的原模型类 .pt、veiledtee 的 XGBoost、value_fulcra 的
FULCRA encoder + 线性探针）。每个基线独立 try，互不影响。
"""

import os, sys, numpy as np, torch, joblib
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID
from model import ValueClassifier
from train import get_device

DEV = get_device()
dv = read_jsonl("data/test.jsonl")
X = [r[CONSISTENT_KEY] for r in dv]
Y = np.array([LABEL2ID[r["Value"]] for r in dv])
def mf1(pred): return f1_score(Y, pred, average="macro", zero_division=0)
def show(name, pred): print(f"  [{name:34s}] dev macroF1 = {mf1(pred):.4f}", flush=True)


@torch.no_grad()
def vc_pred(model_name, ckpt):  # ValueClassifier 通用预测
    tok = AutoTokenizer.from_pretrained(model_name)
    m = ValueClassifier(model_name).to(DEV); m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.eval()
    out = []
    for i in range(0, len(X), 64):
        e = tok(X[i:i+64], truncation=True, max_length=96, padding=True, return_tensors="pt")
        out.append(m(e["input_ids"].to(DEV), e["attention_mask"].to(DEV)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def mean_embed(model_name, encoder, max_len=96):  # 句向量(mean pooling)
    tok = AutoTokenizer.from_pretrained(model_name); out = []
    for i in range(0, len(X), 64):
        e = tok(X[i:i+64], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEV)
        h = encoder(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h*m).sum(1)/m.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(out)


print("=== 经典基线（TF-IDF + joblib）===")
try:
    vec = joblib.load("baselines/classical_models/tfidf_vectorizer.joblib"); Fdv = vec.transform(X)
    for name in ["tfidf_linsvc", "tfidf_logreg", "tfidf_xgboost"]:
        show(name, joblib.load(f"baselines/classical_models/{name}.joblib").predict(Fdv))
except Exception as e: print("  跳过:", e)

print("=== 标准神经基线（ValueClassifier .pt）===")
for mn in ["bert-base-uncased", "roberta-base"]:
    try: show("neural_"+mn, vc_pred(mn, f"baselines/neural_models/{mn}.pt"))
    except Exception as e: print(f"  {mn} 跳过:", e)

print("=== 公开仓库基线（原模型类）===")
# su0315: BaselineModel(BERT+linear), forward 接收文本列表
try:
    sys.path.insert(0, "baselines/su0315_hvd/model")
    from model_baseline import BaselineModel
    m = BaselineModel(19, None, None, None).to(DEV)
    m.load_state_dict(torch.load("baselines/su0315_hvd/nlpcc_out/model.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        p = np.concatenate([m(X[i:i+32]).argmax(-1).cpu().numpy() for i in range(0, len(X), 32)])
    show("su0315 BaselineModel(BERT)", p)
except Exception as e: print("  su0315 跳过:", e)

# cocchieri: BERTClass(roberta), forward(ids,mask,token_type)
try:
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
    show("cocchieri BERTClass(roberta)", np.concatenate(out))
except Exception as e: print("  cocchieri 跳过:", e)

# adam_smith: BertFineTunerPl(DeBERTa), forward(ids,mask)->(loss,sigmoid)
try:
    import torch.nn as nn
    sys.path.insert(0, "baselines/adam_smith")
    from models.BertFineTunerPl import BertFineTunerPl
    from values import ID2LABEL, NUM_CLASSES
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    params = {"MODEL_PATH": "microsoft/deberta-v3-base", "EMBEDDING": "CLS + MEAN",
              "HIDDEN_LAYERS": None, "DROPOUT": 0.1, "CRITERION": [nn.BCEWithLogitsLoss()]}
    m = BertFineTunerPl(NUM_CLASSES, params, [ID2LABEL[i] for i in range(NUM_CLASSES)], 10, 1).to(DEV)
    m.load_state_dict(torch.load("baselines/adam_smith/nlpcc_out/model.pt", map_location=DEV)); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            e = tok(X[i:i+32], truncation=True, max_length=96, padding=True, return_tensors="pt")
            _, o = m(e["input_ids"].to(DEV), e["attention_mask"].to(DEV)); out.append(o.argmax(-1).cpu().numpy())
    show("adam_smith BertFineTunerPl(DeBERTa)", np.concatenate(out))
except Exception as e: print("  adam_smith 跳过:", e)

# veiledtee: bert-base-uncased mean 句向量 + XGBoost
try:
    import xgboost as xgb
    enc = AutoModel.from_pretrained("bert-base-uncased").to(DEV).eval()
    Fb = mean_embed("bert-base-uncased", enc)
    bst = xgb.XGBClassifier(); bst.load_model("baselines/veiledtee_semeval2023/nlpcc_out/xgboost_model.json")
    show("veiledtee BERT-embed + XGBoost", bst.predict(Fb))
except Exception as e: print("  veiledtee 跳过:", e)

# value_fulcra: FULCRA 预训练 encoder + 线性探针
try:
    enc = AutoModel.from_pretrained("bert-base-uncased").to(DEV)
    enc.load_state_dict(torch.load("baselines/value_fulcra/nlpcc_out/fulcra_encoder.pt", map_location=DEV)); enc.eval()
    Ff = mean_embed("bert-base-uncased", enc)
    lr = joblib.load("baselines/value_fulcra/nlpcc_out/logreg.joblib")
    show("value_fulcra FULCRA-encoder + probe", lr.predict(Ff))
except Exception as e: print("  value_fulcra 跳过:", e)

print("EVAL_SAVED_DONE")
