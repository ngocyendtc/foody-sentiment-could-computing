"""Huấn luyện mô hình cảm xúc theo khía cạnh trên dữ liệu đã gán nhãn (data/aspect/foody_aspect.csv).

Mỗi khía cạnh (food, price, service, ambience, hygiene, location) là một bài toán phân lớp 4 nhãn:
none (không nhắc tới) / pos / neu / neg. Dùng lại đúng hai mô hình và đặc trưng của báo cáo:
  Naive Bayes : MultinomialNB trên BoW đếm 1–2 gram
  SVM         : LinearSVC cân bằng lớp trên TF-IDF từ 1–2 gram + n-gram ký tự 2–5
                (bọc CalibratedClassifierCV để có xác suất dùng cho thang điểm 10)
Tham số alpha / C chọn bằng kiểm định chéo 5-fold trên tập huấn luyện (macro-F1).
Đánh giá trên tập kiểm tra và so sánh với cách cũ dùng từ khoá (aspect_keyword.py).

  python src/train_aspect.py
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aspect_model import ASPECT_KEYS, ASPECT_NAMES, CLASSES, MODEL_DIR, class_proba, decide  # noqa: E402
from data_utils import ROOT  # noqa: E402
from text_preprocess import Preprocessor  # noqa: E402

DATA = Path(os.environ.get("ASPECT_DATA", ROOT / "data" / "aspect" / "foody_aspect.csv"))
RESULTS = ROOT / "results" / "aspect_results.json"
GRID = {"nb": [0.05, 0.1, 0.2, 0.5, 1.0], "svm": [0.03, 0.1, 0.3, 1.0]}
TITLES = {"nb": "Naive Bayes", "svm": "SVM tuyến tính", "keyword": "Từ khoá (cách cũ)"}


def make_vec(kind):
    word = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, token_pattern=r"\S+", lowercase=False)
    if kind == "nb":
        return FeatureUnion([("word", TfidfVectorizer(use_idf=False, norm=None, **word))])
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=300_000,
                           sublinear_tf=True, lowercase=False)
    return FeatureUnion([("word", TfidfVectorizer(sublinear_tf=True, **word)), ("char", char)])


def make_clf(kind, p, y=None, calibrate=False):
    if kind == "nb":
        return MultinomialNB(alpha=p)
    svc = LinearSVC(C=p, class_weight="balanced")
    if not calibrate:
        return svc
    minc = min(Counter(y).values())
    if minc >= 2:
        return CalibratedClassifierCV(svc, cv=min(3, minc), method="sigmoid")
    return svc


def tune(kind, texts, Y):
    """Chọn tham số cho từng khía cạnh bằng 5-fold (vectơ hoá học lại trên mỗi fold)."""
    folds = []
    for tr, va in KFold(5, shuffle=True, random_state=42).split(texts):
        vec = make_vec(kind)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xva = vec.transform([texts[i] for i in va])
        folds.append((tr, va, Xtr, Xva))
    best = {}
    for k in ASPECT_KEYS:
        y = np.array(Y[k])
        curve = []
        for p in GRID[kind]:
            scores = []
            for tr, va, Xtr, Xva in folds:
                if len(set(y[tr])) < 2:
                    continue
                pred = make_clf(kind, p).fit(Xtr, y[tr]).predict(Xva)
                scores.append(f1_score(y[va], pred, labels=CLASSES, average="macro", zero_division=0))
            curve.append((p, float(np.mean(scores)) if scores else 0.0))
        best[k] = max(curve, key=lambda r: r[1])
    return best


def tuple_f1(gold, pred):
    """F1 theo cặp (khía cạnh, cảm xúc) kiểu VLSP: chỉ tính các ô khác none."""
    tp = sum(g == p != "none" for g, p in zip(gold, pred))
    npred = sum(p != "none" for p in pred)
    ngold = sum(g != "none" for g in gold)
    prec = tp / npred if npred else 0.0
    rec = tp / ngold if ngold else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def evaluate(gold, pred):
    """gold, pred: dict khía cạnh -> list nhãn."""
    per = {}
    G, P = [], []
    for k in ASPECT_KEYS:
        g, p = gold[k], pred[k]
        G += g
        P += p
        gm = [x != "none" for x in g]
        pm = [x != "none" for x in p]
        dp, dr, df, _ = precision_recall_fscore_support(gm, pm, average="binary", zero_division=0)
        idx = [i for i, m in enumerate(gm) if m]
        pol_acc = accuracy_score([g[i] for i in idx], [p[i] for i in idx]) if idx else None
        per[k] = {"support_mention": int(sum(gm)), "detect_p": dp, "detect_r": dr, "detect_f1": df,
                  "polarity_acc": pol_acc, "macro_f1_4": f1_score(g, p, labels=CLASSES, average="macro", zero_division=0),
                  "tuple_f1": tuple_f1(g, p)[2]}
    gm = [x != "none" for x in G]
    pm = [x != "none" for x in P]
    dp, dr, df, _ = precision_recall_fscore_support(gm, pm, average="binary", zero_division=0)
    tp_, tr_, tf_ = tuple_f1(G, P)
    return {"per_aspect": per, "overall": {"detect_p": dp, "detect_r": dr, "detect_f1": df,
                                            "tuple_p": tp_, "tuple_r": tr_, "tuple_f1": tf_}}


THRESHOLDS = [round(x, 2) for x in np.arange(0.2, 0.81, 0.05)]


def oof_proba(kind, best, texts, Y):
    """Xác suất out-of-fold (5-fold) trên tập huấn luyện, dùng để chọn ngưỡng 'có nhắc'."""
    n = len(texts)
    out = {k: np.zeros((n, len(CLASSES))) for k in ASPECT_KEYS}
    for tr, va in KFold(5, shuffle=True, random_state=7).split(texts):
        vec = make_vec(kind)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xva = vec.transform([texts[i] for i in va])
        for k in ASPECT_KEYS:
            y = [Y[k][i] for i in tr]
            clf = make_clf(kind, best[k][0], y, calibrate=True).fit(Xtr, y)
            out[k][va] = class_proba(clf, Xva)
    return out


def apply_threshold(P, t):
    return [decide(dict(zip(CLASSES, row)), t) for row in P]


def pick_thresholds(oof, Y):
    th = {}
    for k in ASPECT_KEYS:
        th[k] = max(THRESHOLDS, key=lambda t: (tuple_f1(Y[k], apply_threshold(oof[k], t))[2], -abs(t - 0.5)))
    return th


def keyword_baseline(texts):
    """Cách cũ: từ khoá + từ điển cảm xúc + mô hình câu, quy về 6 khía cạnh."""
    import aspect_keyword as kw
    from predict import SentimentModel
    sent = SentimentModel("svm")
    mapping = {"Đồ ăn": "food", "Giá cả": "price", "Thái độ phục vụ": "service", "Tốc độ phục vụ": "service",
               "Không gian": "ambience", "Vệ sinh": "hygiene", "Vị trí": "location"}
    out = {k: [] for k in ASPECT_KEYS}
    lab = {"Tích cực": "pos", "Trung tính": "neu", "Tiêu cực": "neg"}
    for t in texts:
        res, _ = kw.analyze(t, sent)
        scores = {k: [] for k in ASPECT_KEYS}
        for name, r in res.items():
            if r is not None:
                scores[mapping[name]].append(r["score"])
        for k in ASPECT_KEYS:
            out[k].append(lab[kw.label_of(float(np.mean(scores[k])))] if scores[k] else "none")
    return out


def main():
    t0 = time.time()
    df = pd.read_csv(DATA)
    pre = Preprocessor(negation=True, segment=False)
    df["clean"] = df["review"].map(pre)
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    Xtr_text, Xte_text = tr["clean"].tolist(), te["clean"].tolist()
    Ytr = {k: tr[k].tolist() for k in ASPECT_KEYS}
    Yte = {k: te[k].tolist() for k in ASPECT_KEYS}
    print(f"train {len(tr)}, test {len(te)}")

    results = {"n_train": len(tr), "n_test": len(te), "models": {}}
    MODEL_DIR.mkdir(exist_ok=True)
    for kind in ["nb", "svm"]:
        t = time.time()
        best = tune(kind, Xtr_text, Ytr)
        vec = make_vec(kind)
        Xtr = vec.fit_transform(Xtr_text)
        Xte = vec.transform(Xte_text)
        th = pick_thresholds(oof_proba(kind, best, Xtr_text, Ytr), Ytr)   # chỉ dùng tập huấn luyện
        clfs, pred, pred_argmax = {}, {}, {}
        for k in ASPECT_KEYS:
            clf = make_clf(kind, best[k][0], Ytr[k], calibrate=True).fit(Xtr, Ytr[k])
            clfs[k] = clf
            P = class_proba(clf, Xte)
            pred[k] = apply_threshold(P, th[k])
            pred_argmax[k] = [CLASSES[i] for i in P.argmax(axis=1)]
        ev = evaluate(Yte, pred)
        ev["argmax"] = evaluate(Yte, pred_argmax)["overall"]
        ev["params"] = {k: {"param": best[k][0], "cv_macro_f1": best[k][1], "mention_threshold": th[k]}
                        for k in ASPECT_KEYS}
        ev["train_time_s"] = time.time() - t
        results["models"][kind] = ev
        joblib.dump({"vectorizer": vec, "clfs": clfs, "kind": kind, "preprocess": pre.config(),
                     "thresholds": th, "sklearn_version": sklearn.__version__, "test": ev},
                    MODEL_DIR / f"aspect_{kind}.joblib", compress=3)
        o, a = ev["overall"], ev["argmax"]
        print(f"{TITLES[kind]}: phát hiện F1 {o['detect_f1']:.3f} | (khía cạnh, cảm xúc) F1 {o['tuple_f1']:.3f} "
              f"(argmax: {a['detect_f1']:.3f} / {a['tuple_f1']:.3f}) | ngưỡng {th} | {time.time() - t:.1f}s")

    t = time.time()
    results["models"]["keyword"] = evaluate(Yte, keyword_baseline(te["review"].tolist()))
    o = results["models"]["keyword"]["overall"]
    print(f"{TITLES['keyword']}: phát hiện F1 {o['detect_f1']:.3f} | (khía cạnh, cảm xúc) F1 {o['tuple_f1']:.3f} "
          f"| {time.time() - t:.1f}s")
    results["titles"] = TITLES
    results["aspect_names"] = ASPECT_NAMES
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tổng thời gian {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
