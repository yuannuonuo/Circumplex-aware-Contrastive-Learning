# -*- coding: utf-8 -*-
"""
classical_baseline.py —— 经典（非神经）基线：TF-IDF 特征 + 经典分类器。

不使用任何神经网络：用 1-2gram TF-IDF 把 response 向量化，再用 LinearSVC / LogReg /
XGBoost 分类到 19 类。作为"经典基线"与"神经基线（transformer）"对照。
"""

import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from data import read_jsonl, CONSISTENT_KEY
from values import LABEL2ID, NUM_CLASSES


def load(split):
    recs = read_jsonl(f"data/{split}.jsonl")
    return [r[CONSISTENT_KEY] for r in recs], np.array([LABEL2ID[r["Value"]] for r in recs])


def main():
    Xtr, ytr = load("train"); Xdv, ydv = load("dev")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)  # 1-2gram TF-IDF
    Ftr = vec.fit_transform(Xtr); Fdv = vec.transform(Xdv)
    print(f"TF-IDF dim={Ftr.shape[1]} train={len(Xtr)} dev={len(Xdv)}", flush=True)

    import joblib
    os.makedirs("baselines/classical_models", exist_ok=True)
    joblib.dump(vec, "baselines/classical_models/tfidf_vectorizer.joblib")  # 保存 TF-IDF 向量器
    res = {}
    def ev(name, clf):
        clf.fit(Ftr, ytr); pred = clf.predict(Fdv)
        joblib.dump(clf, f"baselines/classical_models/{name}.joblib")        # 保存分类器参数
        f1 = f1_score(ydv, pred, average="macro", zero_division=0); acc = accuracy_score(ydv, pred)
        res[name] = {"dev_macro_f1": float(f1), "dev_acc": float(acc)}
        print(f"  [{name:18s}] dev macroF1={f1:.4f} acc={acc:.4f}", flush=True)

    ev("tfidf_linsvc", LinearSVC(C=1.0, class_weight="balanced"))
    ev("tfidf_logreg", LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"))
    try:
        import xgboost as xgb
        ev("tfidf_xgboost", xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3, subsample=0.8,
                                              tree_method="hist", n_jobs=8, num_class=NUM_CLASSES, objective="multi:softmax"))
    except Exception as e:
        print(f"  [tfidf_xgboost] 跳过: {e}", flush=True)

    json.dump(res, open("baselines/classical_results.json", "w"), indent=2, ensure_ascii=False)
    print("CLASSICAL_DONE")


if __name__ == "__main__":
    main()
