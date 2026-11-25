"""Dự đoán cảm xúc cho bình luận mới.

  python src/predict.py "Đồ ăn ngon, nhân viên dễ thương, sẽ quay lại"
  python src/predict.py --model nb "ko ngon lắm, phục vụ chậm"
  python src/predict.py --file binh_luan.txt
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_preprocess import Preprocessor  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILES = {"svm": "svm_foody.joblib", "nb": "nb_foody.joblib"}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SentimentModel:
    def __init__(self, name="svm"):
        path = MODEL_DIR / MODEL_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"Chưa có {path} - hãy chạy: python src/experiments.py")
        bundle = joblib.load(path)
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError(f"Mô hình lưu bằng scikit-learn {bundle.get('sklearn_version')}, máy này dùng "
                               f"{sklearn.__version__} - hãy chạy lại: python src/experiments.py")
        self.name = name
        self.title = bundle["title"]
        self.pipeline = bundle["pipeline"]
        self.labels = bundle["labels"]
        self.test_metrics = bundle.get("test")
        self.pre = Preprocessor(**bundle["preprocess"])
        self.classes = list(self.pipeline.classes_)

    def scores(self, clean_texts):
        if self.name == "nb":
            return self.pipeline.predict_proba(clean_texts)          # xác suất hậu nghiệm
        return _softmax(self.pipeline.decision_function(clean_texts))  # độ tin cậy tương đối

    def predict(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        clean = [self.pre(t) for t in texts]
        probs = self.scores(clean)
        out = []
        for t, c, p in zip(texts, clean, probs):
            conf = {self.labels[k]: float(v) for k, v in zip(self.classes, p)}
            out.append({"text": t, "clean": c, "label": self.labels[self.classes[int(p.argmax())]],
                        "confidence": conf})
        return out[0] if single else out

    def top_features(self, text, k=8):
        """N-gram từ trong câu đóng góp nhiều nhất cho nhãn dự đoán."""
        clean = self.pre(text)
        word_vec = self.pipeline.named_steps["features"].transformer_list[0][1]
        clf = self.pipeline.named_steps["clf"]
        x = word_vec.transform([clean]).tocoo()
        cls = int(self.scores([clean])[0].argmax())
        if self.name == "nb":
            lp = clf.feature_log_prob_
            weights = lp[cls] - np.delete(lp, cls, axis=0).max(axis=0)
        else:
            weights = clf.coef_[cls][: len(word_vec.vocabulary_)]
        names = word_vec.get_feature_names_out()
        contrib = [(names[j], float(v * weights[j])) for j, v in zip(x.col, x.data)]
        return sorted(contrib, key=lambda r: -r[1])[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--model", choices=["svm", "nb"], default="svm")
    ap.add_argument("--file")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model = SentimentModel(args.model)
    if args.file:
        texts = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        texts = [l.strip() for l in iter(lambda: input("Bình luận (Enter để thoát): "), "")]
    for r in model.predict(texts):
        conf = ", ".join(f"{k}: {v:.2f}" for k, v in r["confidence"].items())
        print(f"[{r['label']}] {r['text']}\n   sau tiền xử lý: {r['clean']}\n   độ tin cậy: {conf}")


if __name__ == "__main__":
    main()
