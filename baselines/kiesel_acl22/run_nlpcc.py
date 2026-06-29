# -*- coding: utf-8 -*-
"""
run_nlpcc.py —— 把 Kiesel et al. (ACL'22) *Identifying the Human Values behind Arguments*
官方 baseline 适配到本任务（NLPCC2026 Track1，19 类单标签，输入 = Consistent Value Response）。

忠实复用该论文三个官方 baseline 的方法/超参（见 src/python/components/models/）：
  - BERT     : bert-base-uncased + 序列分类头 + BCEWithLogitsLoss（多标签式），lr=2e-5/wd=0.01。
               单标签适配：目标用 one-hot（每样本仅 1 正类），预测取 argmax → 单标签。
  - SVM      : TfidfVectorizer(stop_words='english') + LinearSVC(C=18, class_weight='balanced',
               max_iter=10000)。单标签适配：直接多分类。
  - 1-Baseline: 原论文恒预测全 1（多标签 trivial 下限）；单标签里对应"多数类"恒预测下限。

数据规则：只在官方 data/ 上训练，不引入论文自带的 webis-argvalues-22 标注（符合"仅 FULCRA 可作额外监督"）。
用法（项目根运行）：
  python baselines/kiesel_acl22/run_nlpcc.py svm
  python baselines/kiesel_acl22/run_nlpcc.py majority
  python baselines/kiesel_acl22/run_nlpcc.py bert 6
  python baselines/kiesel_acl22/run_nlpcc.py all 6
结果存 baselines/kiesel_acl22/nlpcc_out/result.json，模型存同目录。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import joblib

from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID, NUM_CLASSES

OUT = os.path.join(os.path.dirname(__file__), "nlpcc_out")
os.makedirs(OUT, exist_ok=True)


def load():
    tr = read_jsonl("data/train.jsonl"); dv = read_jsonl("data/test.jsonl")
    Xtr = [r[CONSISTENT_KEY] for r in tr]; ytr = np.array([LABEL2ID[r["Value"]] for r in tr])
    Xdv = [r[CONSISTENT_KEY] for r in dv]; ydv = np.array([LABEL2ID[r["Value"]] for r in dv])
    return Xtr, ytr, Xdv, ydv


def mf1(y, p): return f1_score(y, p, average="macro", zero_division=0)


def run_svm(Xtr, ytr, Xdv, ydv):
    """论文 SVM baseline：TF-IDF(stop_words='english') + LinearSVC(C=18, class_weight='balanced')。"""
    vec = TfidfVectorizer(stop_words="english")        # 与论文一致
    Ftr = vec.fit_transform(Xtr); Fdv = vec.transform(Xdv)
    clf = LinearSVC(C=18, class_weight="balanced", max_iter=10000)  # 论文超参
    clf.fit(Ftr, ytr)
    s = mf1(ydv, clf.predict(Fdv))
    joblib.dump(vec, os.path.join(OUT, "svm_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(OUT, "svm_model.joblib"))
    print(f"  [kiesel_svm]      dev macroF1 = {s:.4f}", flush=True)
    return s


def run_majority(ytr, ydv):
    """论文 1-Baseline 的单标签对应：恒预测训练集多数类（trivial 下限）。"""
    maj = Counter(ytr.tolist()).most_common(1)[0][0]
    s = mf1(ydv, np.full_like(ydv, maj))
    json.dump({"majority_class_id": int(maj)}, open(os.path.join(OUT, "majority.json"), "w"))
    print(f"  [kiesel_majority] dev macroF1 = {s:.4f}", flush=True)
    return s


def run_bert(Xtr, ytr, Xdv, ydv, epochs=6):
    """论文 BERT baseline：bert-base-uncased + 序列分类头 + BCE（多标签式），单标签取 argmax。"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    def enc(texts):
        e = tok(texts, truncation=True, max_length=96, padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"]
    itr, mtr = enc(Xtr); idv, mdv = enc(Xdv)
    ytr_t = torch.tensor(ytr); ydv_t = torch.tensor(ydv)
    onehot = torch.eye(NUM_CLASSES)
    tl = DataLoader(TensorDataset(itr, mtr, ytr_t), batch_size=16, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=NUM_CLASSES).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # 论文 lr/wd
    total = len(tl) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    lossf = torch.nn.BCEWithLogitsLoss()                # 论文多标签 BCE

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for ids, mask, y in tl:
            ids, mask = ids.to(dev), mask.to(dev)
            tgt = onehot[y].to(dev)                       # one-hot 目标（单标签特例）
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = lossf(logits, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        # eval: argmax 取单标签
        model.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(idv), 64):
                lo = model(input_ids=idv[i:i+64].to(dev),
                           attention_mask=mdv[i:i+64].to(dev)).logits
                preds.append(lo.argmax(-1).cpu().numpy())
        s = mf1(ydv, np.concatenate(preds)); best = max(best, s)
        print(f"   kiesel_bert ep{ep}/{epochs} dev macroF1={s:.4f} (best={best:.4f})", flush=True)
        if s >= best:
            model.save_pretrained(os.path.join(OUT, "bert")); tok.save_pretrained(os.path.join(OUT, "bert"))
    print(f"  [kiesel_bert]     dev macroF1 = {best:.4f}", flush=True)
    return best


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    Xtr, ytr, Xdv, ydv = load()
    res = {}
    if which in ("svm", "all"):      res["kiesel_svm"] = run_svm(Xtr, ytr, Xdv, ydv)
    if which in ("majority", "all"): res["kiesel_majority"] = run_majority(ytr, ydv)
    if which in ("bert", "all"):     res["kiesel_bert"] = run_bert(Xtr, ytr, Xdv, ydv, epochs)
    # 合并写入（保留已有键）
    p = os.path.join(OUT, "result.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update({k: round(float(v), 4) for k, v in res.items()})
    json.dump(old, open(p, "w"), indent=2, ensure_ascii=False)
    print("KIESEL_DONE ->", old, flush=True)


if __name__ == "__main__":
    main()
