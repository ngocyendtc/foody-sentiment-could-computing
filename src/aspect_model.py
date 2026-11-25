"""Dùng mô hình cảm xúc theo khía cạnh đã huấn luyện (train_aspect.py) cho bình luận mới."""
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_preprocess import Preprocessor  # noqa: E402

ASPECT_KEYS = ["food", "price", "service", "ambience", "hygiene", "location"]
ASPECT_NAMES = {"food": "Đồ ăn", "price": "Giá cả", "service": "Phục vụ", "ambience": "Không gian",
                "hygiene": "Vệ sinh", "location": "Vị trí"}
MAIN_KEYS = ["food", "price", "service"]
CLASSES = ["none", "pos", "neu", "neg"]
POLAR = ["pos", "neu", "neg"]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

SPLIT_RE = re.compile(
    r"(?<!\d)[.,](?!\d)|[!?;\n]+|\s(?:nhưng mà|nhưng|tuy nhiên|mỗi tội|có điều|ngoại trừ|với lại|song|và)\s",
    flags=re.IGNORECASE)


def split_clauses(text):
    return [c.strip() for c in SPLIT_RE.split(" " + text + " ") if c and len(c.strip()) > 1]


def label_of_scale(x):
    """Thang 10: dưới 5 tiêu cực; từ 5 đến dưới 8 trung tính; từ 8 trở lên tích cực."""
    return "Tích cực" if x >= 8 else "Trung tính" if x >= 5 else "Tiêu cực"


def band_score(p, u, n):
    """Điểm /10 nằm trong khoảng của nhãn có xác suất lớn nhất:
    tích cực 8–10, trung tính 5–7,9, tiêu cực 0–4,9; vị trí trong khoảng theo độ tin cậy."""
    s = p + u + n
    if s <= 1e-9:
        return 6.5
    p, u, n = p / s, u / s, n / s
    lab = max((p, "pos"), (u, "neu"), (n, "neg"))[1]
    if lab == "pos":
        return round(min(10.0, max(8.0, 8 + 2 * (p - 1 / 3) / (2 / 3))), 1)
    if lab == "neg":
        return round(min(4.9, max(0.0, 5 - 5 * (n - 1 / 3) / (2 / 3))), 1)
    return round(min(7.9, max(5.0, 6.5 + 1.5 * (p - n))), 1)


def score10(pk):
    return band_score(pk["pos"], pk["neu"], pk["neg"])


def decide(pk, t=0.5):
    """'none' nếu P(không nhắc) >= ngưỡng t, ngược lại nhãn cảm xúc có xác suất lớn nhất."""
    if pk["none"] >= t:
        return "none"
    return max(POLAR, key=lambda c: pk[c])


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def class_proba(clf, X):
    """Ma trận xác suất n × 4 theo thứ tự CLASSES (lớp không có trong dữ liệu huấn luyện = 0)."""
    P = clf.predict_proba(X) if hasattr(clf, "predict_proba") else _softmax(3 * clf.decision_function(X))
    out = np.zeros((X.shape[0], len(CLASSES)))
    for j, c in enumerate(clf.classes_):
        out[:, CLASSES.index(c)] = P[:, j]
    return out


class AspectModel:
    def __init__(self, kind="svm"):
        path = MODEL_DIR / f"aspect_{kind}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Chưa có {path} - hãy chạy: python src/train_aspect.py")
        b = joblib.load(path)
        if b.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("Mô hình khía cạnh lưu bằng phiên bản scikit-learn khác - chạy lại train_aspect.py")
        self.kind, self.vec, self.clfs, self.test = kind, b["vectorizer"], b["clfs"], b.get("test")
        self.thresholds = b.get("thresholds", {k: 0.5 for k in ASPECT_KEYS})
        self.pre = Preprocessor(**b["preprocess"])

    def proba(self, texts):
        X = self.vec.transform([self.pre(t) for t in texts])
        out = [dict() for _ in texts]
        for k, clf in self.clfs.items():
            for row, p in zip(out, class_proba(clf, X)):
                row[k] = dict(zip(CLASSES, map(float, p)))
        return out

    def decide(self, pk, k):
        return decide(pk, self.thresholds[k])

    def analyze(self, text):
        """Kết quả theo khía cạnh cho một bình luận.

        Căn cứ cho mỗi khía cạnh: bỏ lần lượt từng vế câu ra khỏi bình luận và xem xác suất của
        khía cạnh đó giảm bao nhiêu; vế làm giảm nhiều nhất là vế mô hình dựa vào (giải thích kiểu
        occlusion, luôn khớp với dự đoán trên cả câu)."""
        p = self.proba([text])[0]
        clauses = split_clauses(text)
        variants = []
        if len(clauses) >= 2:
            variants = self.proba([text.replace(c, " ", 1) for c in clauses])
        res = {}
        for k in ASPECT_KEYS:
            pk = p[k]
            lab = self.decide(pk, k)
            if lab == "none":
                res[k] = None
                continue
            ev = []
            if variants:
                for c, q in zip(clauses, variants):
                    d = {c2: pk[c2] - q[k][c2] for c2 in POLAR}
                    best = max(d, key=d.get)
                    if d[best] > 0.05:
                        ev.append({"clause": c, "label": best, "weight": round(d[best], 3)})
                # chỉ giữ căn cứ cùng chiều với nhãn (trung tính: cả hai chiều) và đủ mạnh so với căn cứ chính
                allowed = {"pos": {"pos"}, "neg": {"neg"}, "neu": set(POLAR)}[lab]
                ev = sorted((e for e in ev if e["label"] in allowed), key=lambda e: -e["weight"])
                if ev:
                    top = ev[0]["weight"]
                    ev = [e for e in ev if e["weight"] >= max(0.05, 0.4 * top)]
            else:
                ev = [{"clause": text.strip(), "label": lab, "weight": round(pk[lab], 3)}]
            s = score10(pk)
            res[k] = {"scale": s, "label": label_of_scale(s), "pred": lab, "probs": pk, "evidence": ev}
        return res, p


def describe(r):
    q = lambda cs: ", ".join(f"“{c['clause'].strip().rstrip('.!').lower()}”" for c in cs[:2])
    ev = r["evidence"]
    pos = [c for c in ev if c["label"] == "pos"]
    neg = [c for c in ev if c["label"] == "neg"]
    neu = [c for c in ev if c["label"] == "neu"]
    lab = r.get("pred")
    if lab == "pos" and pos:
        return f"Được khen: {q(pos)}."
    if lab == "neg" and neg:
        return f"Bị chê: {q(neg)}."
    if lab == "neu":
        if pos and neg:
            return f"Vừa khen {q(pos)}, vừa chê {q(neg)}."
        if neg:
            return f"Có ý chê nhẹ: {q(neg)}."
        if pos:
            return f"Có ý khen nhưng chưa rõ: {q(pos)}."
        if neu:
            return f"Nhắc tới ở mức trung tính: {q(neu)}."
    pr = r["probs"]
    return (f"Mô hình nhận ra khía cạnh này từ cả câu (tích cực {pr['pos']:.0%} · trung tính "
            f"{pr['neu']:.0%} · tiêu cực {pr['neg']:.0%}).")
