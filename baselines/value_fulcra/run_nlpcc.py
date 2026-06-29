# -*- coding: utf-8 -*-
"""
run_nlpcc.py —— value_fulcra (microsoft/ValueCompass) 适配为本任务分类基线。

CLAVE 评估器代码"待发布"（仓库无本地分类器），value_fulcra 唯一能落地的分类基线是
其核心思想：用 **FULCRA 数据**（本任务规则允许的额外监督数据）预训练一个价值表征 encoder，
再在本任务上做线性探针。即：
  ① 用 FULCRA(data_hybrid.jsonl) 的"粗粒度价值多标签"预训练 bert-base-uncased（复用
     项目 src/pretrain_fulcra.py 的 load_fulcra）；
  ② 冻结该 encoder，在我们 19 类上用逻辑回归做线性探针，报 dev macro-F1。
FULCRA 数据用的是本仓库 clone 的 Value_FULCRA/data/data_hybrid.jsonl。
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID, NUM_CLASSES

# ===== FULCRA 加载（自包含，原在 src/pretrain_fulcra.py，已内联以免依赖）=====
BROAD_VOCAB = [   # FULCRA 出现的粗粒度价值词表（10 经典 Schwartz + humility）
    "self-direction", "stimulation", "hedonism", "achievement", "power",
    "security", "conformity", "tradition", "benevolence", "universalism", "humility",
]
V2I = {v: i for i, v in enumerate(BROAD_VOCAB)}


def parse_response(dialogue: str) -> str:
    """从 FULCRA 的 dialogue 抽取模型回答（最后一个 "Bob:" 之后）。"""
    idx = dialogue.rfind("Bob:")
    return (dialogue[idx + 4:] if idx >= 0 else dialogue).strip()


def load_fulcra(path: str):
    """读 FULCRA，返回 [(回答文本, 多热标签向量), ...]；跳过全负极性样本。"""
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        labels = torch.zeros(len(BROAD_VOCAB))
        for v in r.get("value_types", []):
            name, _, pol = v.rpartition(":")
            name = name.strip().lower(); pol = pol.strip()
            if name in V2I and pol in ("+1", "1", "1+"):
                labels[V2I[name]] = 1.0
        if labels.sum() == 0:
            continue
        recs.append((parse_response(r["dialogue"]), labels))
    return recs

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL = "bert-base-uncased"
FULCRA = "baselines/value_fulcra/Value_FULCRA/data/data_hybrid.jsonl"  # clone 的 FULCRA 数据


@torch.no_grad()
def embed(texts, tok, encoder, bs=64, max_len=96):
    encoder.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEVICE)
        hs = encoder(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(out)


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tok = AutoTokenizer.from_pretrained(MODEL)
    encoder = AutoModel.from_pretrained(MODEL).to(DEVICE)
    head = nn.Linear(encoder.config.hidden_size, len(BROAD_VOCAB)).to(DEVICE)

    # ① FULCRA 粗粒度价值多标签预训练
    recs = load_fulcra(FULCRA)
    print(f"device={DEVICE} FULCRA usable={len(recs)} 预训练 {epochs} epoch", flush=True)
    class DS(Dataset):
        def __init__(s, r): s.r = r
        def __len__(s): return len(s.r)
        def __getitem__(s, i): return s.r[i]
    def coll(b):
        enc = tok([x[0] for x in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.stack([x[1] for x in b])
    loader = DataLoader(DS(recs), batch_size=32, shuffle=True, collate_fn=coll)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06*len(loader)*epochs), len(loader)*epochs)
    for ep in range(1, epochs+1):
        encoder.train(); head.train(); run = 0.0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            hs = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float(); pooled = (hs*m).sum(1)/m.sum(1).clamp(min=1e-9)
            loss = F.binary_cross_entropy_with_logits(head(pooled), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); run += loss.item()
        print(f"  [FULCRA pretrain ep{ep}] bce={run/len(loader):.4f}", flush=True)

    # ② 本任务线性探针（冻结 FULCRA 预训练 encoder + 逻辑回归）
    tr = read_jsonl("data/train.jsonl"); dv = read_jsonl("data/test.jsonl")
    Xtr = embed([r[CONSISTENT_KEY] for r in tr], tok, encoder); ytr = np.array([LABEL2ID[r["Value"]] for r in tr])
    Xdv = embed([r[CONSISTENT_KEY] for r in dv], tok, encoder); ydv = np.array([LABEL2ID[r["Value"]] for r in dv])
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced").fit(Xtr, ytr)
    pred = clf.predict(Xdv)
    f1 = f1_score(ydv, pred, average="macro", zero_division=0); acc = accuracy_score(ydv, pred)
    print(f">>> value_fulcra (FULCRA预训练 encoder + 线性探针) 本任务 dev macroF1={f1:.4f} acc={acc:.4f}", flush=True)
    os.makedirs("baselines/value_fulcra/nlpcc_out", exist_ok=True)
    import joblib
    torch.save(encoder.state_dict(), "baselines/value_fulcra/nlpcc_out/fulcra_encoder.pt")  # FULCRA 预训练 encoder
    joblib.dump(clf, "baselines/value_fulcra/nlpcc_out/logreg.joblib")                       # 线性探针分类器
    json.dump({"repo": "value_fulcra (FULCRA数据预训练 bert encoder + 19类线性探针)",
               "task": "NLPCC2026 Track1 单标签19类", "fulcra_pretrain_epochs": epochs,
               "dev_macro_f1": float(f1), "dev_acc": float(acc)},
              open("baselines/value_fulcra/nlpcc_out/result.json", "w"), indent=2, ensure_ascii=False)
    print("VALUE_FULCRA_NLPCC_DONE")


if __name__ == "__main__":
    main()
