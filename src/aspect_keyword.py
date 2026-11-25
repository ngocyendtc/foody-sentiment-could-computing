"""BASELINE dùng từ khoá (cách cũ, chỉ để so sánh). Phân tích cảm xúc mức khía cạnh (Hình 1.1).

Dữ liệu Foody chỉ có nhãn cho cả bình luận, nên ở đây dùng cách lai, không cần huấn luyện thêm:
  1. tách bình luận thành các vế theo dấu câu và từ nối tương phản ("nhưng", "tuy nhiên", "và", ...);
  2. nhận diện khía cạnh của từng vế bằng từ điển từ khoá;
  3. chấm cảm xúc từng vế = điểm từ điển cảm xúc (có xử lý phủ định) + (P(tích cực) - P(tiêu cực))
     của mô hình NB/SVM đã học;
  4. gộp các vế theo khía cạnh, quy về thang điểm 10.
"""
import math
import re

from text_preprocess import NEGATORS, Preprocessor

ASPECTS = {
    "Đồ ăn": {
        "icon": "🍲",
        "kw": ["đồ ăn", "món", "món ăn", "thức ăn", "đồ uống", "nước uống", "thức uống", "vị", "hương vị",
               "nước dùng", "nước lèo", "topping", "sốt", "khẩu phần", "phần ăn", "ngon", "dở", "nhạt",
               "mặn", "ngọt", "cay", "béo", "thơm", "tanh", "dai", "giòn", "tươi", "nguội", "đậm đà",
               "vừa miệng", "phở", "bún", "cơm", "lẩu", "nướng", "mì", "bánh", "kem", "trà sữa", "trà",
               "sữa", "cà phê", "cafe", "gà", "bò", "heo", "thịt", "cá", "hải sản", "ốc", "chè", "pizza",
               "sushi", "đau bụng", "ngộ độc", "menu"],
    },
    "Giá cả": {
        "icon": "💰",
        "kw": ["giá", "giá_tiền", "giá cả", "tiền", "rẻ", "đắt", "mắc", "chát", "phải chăng", "bình dân",
               "đáng tiền", "chi phí", "khuyến mãi", "giảm giá", "voucher", "combo", "bill", "hóa đơn",
               "cắt cổ", "hợp lý"],
    },
    "Thái độ phục vụ": {
        "icon": "🙋",
        "kw": ["thái độ", "chủ quán", "anh chủ", "chị chủ", "cô chủ", "nhiệt tình", "chu đáo",
               "niềm nở", "thân thiện", "lịch sự", "vui vẻ", "dễ thương", "khó chịu", "thô lỗ", "hách dịch",
               "cau có", "tận tình", "thu ngân", "bảo vệ", "giữ xe", "tư vấn"],
    },
    "Tốc độ phục vụ": {
        "icon": "⏱️",
        "kw": ["lên món", "ra món", "chờ", "đợi", "lâu", "chậm", "nhanh", "order", "gọi món", "giao hàng",
               "ship", "shipper", "phút"],
    },
    "Không gian": {
        "icon": "🏠",
        "kw": ["không gian", "rộng", "rộng rãi", "chật", "hẹp", "thoáng", "thoáng mát", "mát", "nóng",
               "máy lạnh", "điều hòa", "view", "decor", "trang trí", "nhạc", "ồn", "ồn ào", "yên tĩnh",
               "chỗ ngồi", "bàn ghế", "đông", "ấm cúng", "chill", "đẹp", "chỗ đậu xe", "chỗ để xe", "bãi xe",
               "sang trọng", "chỗ"],
    },
    "Vệ sinh": {
        "icon": "🧼",
        "kw": ["vệ sinh", "sạch", "sạch sẽ", "dơ", "bẩn", "ruồi", "gián", "chuột", "tóc", "mùi hôi", "hôi",
               "nhà vệ sinh", "toilet"],
    },
    "Vị trí": {
        "icon": "📍",
        "kw": ["vị trí", "địa chỉ", "dễ tìm", "khó tìm", "hẻm", "mặt tiền", "mặt đường", "gần", "xa",
               "trung tâm", "đường"],
    },
}
MAIN_ASPECTS = ["Đồ ăn", "Giá cả", "Thái độ phục vụ"]
# từ khoá chung chung: chỉ tính khi vế đó không khớp khía cạnh nào cụ thể hơn
GENERIC = {"phục vụ": "Thái độ phục vụ", "nhân viên": "Thái độ phục vụ", "chất lượng": "Đồ ăn", "quán": "Không gian"}

POS = ["ngon", "tuyệt", "tuyệt vời", "rẻ", "hợp lý", "phải chăng", "bình dân", "đáng tiền", "nhiệt tình",
       "chu đáo", "niềm nở", "thân thiện", "dễ thương", "vui vẻ", "lịch sự", "tận tình", "nhanh", "sạch",
       "sạch sẽ", "thoáng", "thoáng mát", "rộng", "rộng rãi", "đẹp", "xinh", "ấm cúng", "yên tĩnh", "mát",
       "tươi", "thơm", "giòn", "đậm đà", "vừa miệng", "ổn", "tốt", "hài lòng", "thích", "ưng", "xuất sắc",
       "đầy đặn", "dễ tìm", "thuận tiện", "chill", "xịn", "sang trọng", "cao cấp", "hấp dẫn",
       "ủng hộ", "quay lại", "emo_pos", "gần"]
NEG = ["dở", "tệ", "chán", "nhạt", "mặn", "dai", "khô", "nguội", "ôi", "thiu", "hôi", "tanh", "đắt", "mắc",
       "chát", "cắt cổ", "chậm", "lâu", "khó chịu", "cáu", "gắt", "thô lỗ", "hách dịch", "cau có", "vô duyên",
       "bực", "bẩn", "dơ", "ruồi", "gián", "chật", "nóng", "ồn", "ồn ào", "hẹp", "khó tìm", "xa", "thất vọng",
       "ít", "kém", "lừa đảo", "đau bụng", "ngộ độc", "cao", "emo_neg", "không đáng", "quá tệ", "chờ", "đợi",
       "dơ bẩn", "tạm", "mất vệ sinh"]
LEXICON = {**{w: 1.0 for w in POS}, **{w: -1.0 for w in NEG}}
LEXICON["tạm"] = -0.5

SPLIT_RE = re.compile(
    r"(?<!\d)[.,](?!\d)|[!?;\n]+|\s(?:nhưng mà|nhưng|tuy nhiên|mỗi tội|có điều|ngoại trừ|với lại|song|và)\s",
    flags=re.IGNORECASE)
_pre = Preprocessor(negation=False)


def split_clauses(text):
    return [c.strip() for c in SPLIT_RE.split(" " + text + " ") if c and len(c.strip()) > 1]


def _find(tokens, phrases):
    """Khớp cụm từ dài nhất trước, không chồng lấn. Trả về [(vị trí, cụm)]."""
    seqs = sorted({tuple(p.split()) for p in phrases}, key=len, reverse=True)
    used, found = set(), []
    for seq in seqs:
        n = len(seq)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == seq and not used & set(range(i, i + n)):
                used |= set(range(i, i + n))
                found.append((i, " ".join(seq)))
    return sorted(found)


def _negated(tokens, i):
    for j in range(max(0, i - 3), i):
        if tokens[j] in NEGATORS and not (j + 1 < len(tokens) and tokens[j + 1] in {"gian", "khí"}):
            return True
    return False


def lexicon_score(tokens):
    score, hits = 0.0, []
    for i, w in _find(tokens, LEXICON):
        v = LEXICON[w]
        if _negated(tokens, i):
            v, w = -v, "không " + w
        score += v
        hits.append(w)
    return score, hits


def detect_aspects(tokens):
    found = {a for a, info in ASPECTS.items() if _find(tokens, info["kw"])}
    if not found:
        found = {asp for kw, asp in GENERIC.items() if _find(tokens, [kw])}
    return found


def label_of_scale(x):
    """Nhãn theo thang 10: 0–4,99 tiêu cực; 5–7,99 trung tính; từ 8 trở lên tích cực."""
    return "Tích cực" if x >= 8 else "Trung tính" if x >= 5 else "Tiêu cực"


def label_of(score):
    return label_of_scale(to_scale(score))


def to_scale(score):
    """Điểm cảm xúc -> thang 0–10 (5 = trung tính)."""
    return round(5 + 5 * math.tanh(score / 2), 1)


def describe(r):
    """Một câu nhận xét ngắn từ các vế câu làm căn cứ."""
    ev = r["evidence"]
    q = lambda cs: ", ".join(f"“{c['clause'].strip().rstrip('.!').lower()}”" for c in cs[:2])
    pos = [c for c in ev if c["label"] == "Tích cực"]
    neg = [c for c in ev if c["label"] == "Tiêu cực"]
    neu = [c for c in ev if c["label"] == "Trung tính"]
    if pos and neg:
        return f"Được khen {q(pos)}, nhưng bị chê {q(neg)}."
    if pos:
        return f"Được khen: {q(pos)}."
    if neg:
        return f"Bị chê: {q(neg)}."
    return f"Có nhắc tới {q(neu)} nhưng không rõ khen hay chê."


def analyze(text, model):
    clauses = []
    for c in split_clauses(text):
        tokens = _pre(c).split()
        if not tokens:
            continue
        aspects = detect_aspects(tokens)
        if not aspects:
            clauses.append({"clause": c, "aspects": [], "score": None})
            continue
        lex, hits = lexicon_score(tokens)
        conf = model.predict(c)["confidence"]
        mscore = conf["Tích cực"] - conf["Tiêu cực"]
        score = lex + mscore
        clauses.append({"clause": c, "aspects": sorted(aspects), "score": score, "lexicon": hits,
                        "model": mscore, "label": label_of(score)})
    result = {}
    for a in ASPECTS:
        cs = [c for c in clauses if a in c["aspects"]]
        if not cs:
            result[a] = None
            continue
        s = sum(c["score"] for c in cs) / len(cs)
        result[a] = {"score": s, "scale": to_scale(s), "label": label_of(s), "evidence": cs}
    return result, clauses


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from predict import SentimentModel
    m = SentimentModel("svm")
    for t in ["Món ăn ngon, giá hợp lý nhưng phục vụ hơi chậm và quán khá chật.",
              "Đồ ăn không tệ, nhưng giá hơi cao so với chất lượng",
              "Quán bẩn, nhân viên thái độ khó chịu, chờ 40 phút mới có đồ ăn. Không bao giờ quay lại",
              "Ngon lắm, ăn xong đau bụng cả đêm",
              "ko ngon lắm, nv phục vụ chậm :("]:
        res, cl = analyze(t, m)
        print(t)
        for a, r in res.items():
            if r:
                print(f"   {a:16s} {r['label']:10s} {r['scale']}/10  <- {[c['clause'] for c in r['evidence']]}")
