"""Tiền xử lý bình luận tiếng Việt (mục 2.3 báo cáo).

Thứ tự: NFC + chuẩn hoá dấu thanh + chữ thường -> url/email/sdt, emoji -> giá tiền, con số
-> tiếng cười, chữ lặp -> bỏ dấu câu -> teencode -> (tách từ) -> (đánh dấu phủ định).
"""
import re
import unicodedata

# --- chuẩn hoá dấu thanh kiểu cũ: "oà" -> "òa" (chỉ khi ở cuối âm tiết) ---
_TONES = {"a": "àáảãạ", "e": "èéẻẽẹ", "y": "ỳýỷỹỵ"}
_HEAD = {"o": "òóỏõọ", "u": "ùúủũụ"}
_TONE_MAP = {}
for head, second in (("o", "a"), ("o", "e"), ("u", "y")):
    for i, marked in enumerate(_TONES[second]):
        _TONE_MAP[head + marked] = _HEAD[head][i] + second
_TONE_RE = re.compile(r"(?<!q)(" + "|".join(_TONE_MAP) + r")(?!\w)")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?<!\d)(\+84|84|0)\d{9}(?!\d)")

EMO_POS = [":)))", ":))", ":)", ":-)", ":d", "=))", "=)", "<3", ";)", "^^", "^_^", ":p"]
EMO_NEG = [":(((", ":((", ":(", ":-(", ":'(", "t_t", "-_-", ":/", ">.<", "=(("]
EMOJI_POS = "😀😁😂😃😄😅😆😉😊😋😍😘🥰🤩😎👍👌💯🔥❤💕💖😻🥳🤤👏🙌"
EMOJI_NEG = "😞😟😠😡😢😣😤😥😩😫😭😒🙁☹😕🤮🤢👎💔😓😪😖"

PRICE_RE = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:k|nghìn|ngàn|nghin|ngan|đ|đồng|dong|vnd|vnđ|tr|triệu|củ)(?![^\W\d_])"
)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
LAUGH_RE = re.compile(r"^(?:k{2,}|(?:ha){2,}h?|(?:he){2,}|(?:hi){2,}|(?:hj){2,}|(?:hô){2,})$")
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")
PUNCT_RE = re.compile(r"[^\w\s]")

# 89 mục, chỉ thay khi khớp nguyên token ("k" -> "không" nhưng "kem" giữ nguyên)
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "kg": "không", "hok": "không",
    "hông": "không", "hk": "không", "khong": "không", "hem": "không", "khum": "không",
    "kô": "không", "kp": "không phải", "kb": "không biết", "kbh": "không bao giờ",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được", "duoc": "được",
    "nv": "nhân viên", "nvien": "nhân viên", "pv": "phục vụ", "nvpv": "nhân viên phục vụ",
    "ok": "ổn", "oke": "ổn", "okela": "ổn", "okie": "ổn", "oki": "ổn", "okay": "ổn",
    "p": "phút", "ph": "phút", "h": "giờ",
    "j": "gì", "gi": "gì", "z": "vậy", "v": "vậy", "vs": "với", "vl": "vãi",
    "wa": "quá", "wá": "quá", "qá": "quá", "lun": "luôn", "luon": "luôn",
    "bít": "biết", "bik": "biết", "bjt": "biết", "mn": "mọi người", "mng": "mọi người",
    "ng": "người", "ngta": "người ta", "nc": "nước", "r": "rồi", "rùi": "rồi", "ròi": "rồi",
    "cx": "cũng", "cg": "cũng", "cũg": "cũng", "trc": "trước", "trk": "trước",
    "sp": "sản phẩm", "t": "tôi", "mik": "mình", "mk": "mình", "mìh": "mình", "e": "em",
    "thik": "thích", "thjk": "thích", "iu": "yêu", "dth": "dễ thương", "dthw": "dễ thương",
    "bt": "bình thường", "bth": "bình thường", "bthg": "bình thường", "nhìu": "nhiều",
    "nhiu": "nhiều", "hnay": "hôm nay", "hqua": "hôm qua", "bh": "bao giờ", "chx": "chưa",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn", "thx": "cảm ơn",
    "good": "tốt", "nice": "tốt", "bad": "tệ", "đag": "đang", "dag": "đang", "đg": "đang",
    "mún": "muốn", "lém": "lắm", "lm": "làm", "ngonn": "ngon", "sr": "xin lỗi",
}

NEGATORS = {"không", "chẳng", "chưa", "đừng", "chớ", "hổng"}  # "chả" cố ý bỏ (bún chả)
FILLERS = {"hề", "quá", "lắm", "thật", "thực", "sự", "hẳn", "mấy", "là"}
NEG_EXCEPTIONS = {("không", "gian"), ("không", "khí"), ("chẳng", "hạn")}
SPECIAL = {"url", "email", "sdt", "emo_pos", "emo_neg", "giá_tiền", "con_số", "haha"}


def _segment(tokens, words, max_len=4):
    """So khớp cực đại tiến với từ điển (tối đa 4 âm tiết)."""
    out, i = [], 0
    while i < len(tokens):
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            cand = " ".join(tokens[i:i + n])
            if n == 1 or cand in words:
                out.append(cand.replace(" ", "_"))
                i += n
                break
    return out


def _mark_negation(tokens):
    """'không hề ngon' -> 'không không_ngon hề ngon' (giữ token gốc)."""
    out = []
    for i, tok in enumerate(tokens):
        out.append(tok)
        if tok not in NEGATORS:
            continue
        j, skipped = i + 1, 0
        while j < len(tokens) and tokens[j] in FILLERS and skipped < 2:
            j, skipped = j + 1, skipped + 1
        if j >= len(tokens):
            continue
        nxt = tokens[j]
        if (tok, nxt) in NEG_EXCEPTIONS or nxt in NEGATORS or nxt in SPECIAL:
            continue
        out.append(f"{tok}_{nxt}")
    return out


class Preprocessor:
    def __init__(self, negation=True, segment=False, dict_path=None):
        self.negation = negation
        self.segment = segment
        self.words = set()
        if segment:
            with open(dict_path, encoding="utf-8") as f:
                self.words = {unicodedata.normalize("NFC", w.strip().lower()) for w in f if " " in w.strip()}

    def config(self):
        return {"negation": self.negation, "segment": self.segment}

    def __call__(self, text):
        if not isinstance(text, str):
            return ""
        s = unicodedata.normalize("NFC", text).lower()
        s = _TONE_RE.sub(lambda m: _TONE_MAP[m.group(1)], s)
        s = URL_RE.sub(" url ", s)
        s = EMAIL_RE.sub(" email ", s)
        s = PHONE_RE.sub(" sdt ", s)
        for e in sorted(EMO_POS, key=len, reverse=True):
            s = s.replace(e, " emo_pos ")
        for e in sorted(EMO_NEG, key=len, reverse=True):
            s = s.replace(e, " emo_neg ")
        s = "".join(" emo_pos " if c in EMOJI_POS else " emo_neg " if c in EMOJI_NEG else c for c in s)
        s = PRICE_RE.sub(" giá_tiền ", s)
        s = NUM_RE.sub(" con_số ", s)
        s = PUNCT_RE.sub(" ", s)
        tokens = []
        for tok in s.split():
            if LAUGH_RE.match(tok):
                tok = "haha"
            else:
                tok = REPEAT_RE.sub(r"\1", tok)
            tokens.extend(TEENCODE.get(tok, tok).split())
        if self.segment:
            tokens = _segment(tokens, self.words)
        if self.negation:
            tokens = _mark_negation(tokens)
        return " ".join(tokens)


if __name__ == "__main__":
    p = Preprocessor()
    for t in [
        "Đồ ăn ngonnnn quá, nv phục vụ nhiệt tình!!! sẽ quay lại",
        "Quán dơ, nv thái độ ko tốt, đợi 30p mới có đồ",
        "Giá 45k/tô hơi mắc nhưng chất lượng ok, không gian thoáng mát",
        "Phở ở đây không hề ngon như quảng cáo, nước dùng nhạt :(",
    ]:
        print(t, "\n ->", p(t))
