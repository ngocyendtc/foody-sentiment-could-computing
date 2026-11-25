"""Huấn luyện nhanh hai mô hình với cấu hình cuối cùng ở Bảng 2.4 (không dò lại tham số).

  Naive Bayes : MultinomialNB, BoW đếm 1–2 gram, alpha = 0,2
  SVM         : LinearSVC cân bằng lớp, TF-IDF từ 1–2 gram + ký tự 2–5, C = 0,1
"""
import json
import time
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from data_utils import LABEL_NAMES, LABEL_ORDER, ROOT, clean_splits, load_raw
from text_preprocess import Preprocessor

MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def word_vec(tfidf=True):
    kw = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, token_pattern=r"\S+", lowercase=False)
    if tfidf:
        return TfidfVectorizer(sublinear_tf=True, **kw)
    return TfidfVectorizer(use_idf=False, norm=None, **kw)  # BoW đếm tần suất


def build(name):
    if name == "nb":
        return Pipeline([("features", FeatureUnion([("word", word_vec(tfidf=False))])),
                         ("clf", MultinomialNB(alpha=0.2))])
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000,
                           sublinear_tf=True, lowercase=False)
    return Pipeline([("features", FeatureUnion([("word", word_vec()), ("char", char)])),
                     ("clf", LinearSVC(C=0.1, class_weight="balanced"))])


def evaluate(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABEL_ORDER, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred), "precision": mp, "recall": mr, "macro_f1": mf,
        "per_class": {LABEL_NAMES[c]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i, c in enumerate(LABEL_ORDER)},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABEL_ORDER).tolist(),
    }


def main():
    t0 = time.time()
    data, stats = clean_splits(load_raw())
    print("Làm sạch:", stats)
    pre = Preprocessor(negation=True, segment=False)
    X = {k: df["review"].map(pre).tolist() for k, df in data.items()}
    y = {k: df["label"].tolist() for k, df in data.items()}
    print(f"Tiền xử lý xong ({time.time() - t0:.1f}s)")

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    results = {"data": stats, "models": {}}
    for name, title, fname in [("nb", "Naive Bayes (α = 0,2)", "nb_foody.joblib"),
                               ("svm", "SVM cân bằng lớp (C = 0,1)", "svm_foody.joblib")]:
        t = time.time()
        model = build(name).fit(X["train"], y["train"])
        train_time = time.time() - t
        res = {"title": title, "train_time_s": train_time,
               "dev": evaluate(y["dev"], model.predict(X["dev"])),
               "test": evaluate(y["test"], model.predict(X["test"]))}
        results["models"][name] = res
        joblib.dump({"pipeline": model, "preprocess": pre.config(), "labels": LABEL_NAMES,
                     "title": title, "test": res["test"], "sklearn_version": sklearn.__version__}, MODELS / fname, compress=3)
        print(f"{title}: train {train_time:.1f}s | test acc {res['test']['accuracy']:.4f} "
              f"macro-F1 {res['test']['macro_f1']:.4f}")
    (RESULTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tổng thời gian: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
