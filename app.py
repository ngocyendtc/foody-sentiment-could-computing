"""Giao diện web demo phân tích cảm xúc bình luận quán ăn (mục 2.6). Chạy: streamlit run app.py"""
import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import SentimentModel  # noqa: E402
from aspect_model import (ASPECT_KEYS, ASPECT_NAMES, MAIN_KEYS, AspectModel, band_score,  # noqa: E402
                          describe, label_of_scale, score10)

ORDER = ["Tiêu cực", "Trung tính", "Tích cực"]
SCALE_NOTE = "Thang 10: dưới 5 là tiêu cực · từ 5 đến dưới 8 là trung tính · từ 8 trở lên là tích cực."
TONE = {"Tích cực": "pos", "Trung tính": "neu", "Tiêu cực": "neg"}
LAB_VI = {"none": "–", "pos": "Tích cực", "neu": "Trung tính", "neg": "Tiêu cực"}
EXAMPLES = [
    "Món ăn ngon, giá hợp lý nhưng phục vụ hơi chậm và quán khá chật.",
    "Đồ ăn ngon, giá cả hợp lý, nhân viên phục vụ rất nhiệt tình, nhất định sẽ quay lại",
    "Quán bẩn, nhân viên thái độ khó chịu, chờ 40 phút mới có đồ ăn. Không bao giờ quay lại",
    "Phở không hề ngon như lời quảng cáo, nước dùng nhạt",
    "Đồ ăn không tệ, nhưng giá hơi cao so với chất lượng",
    "ko ngon lắm, nv phục vụ chậm :(",
    "Ngon lắm, ăn xong đau bụng cả đêm",
    "Quán trong hẻm hơi khó tìm, không gian nhỏ nhưng ấm cúng, trà sữa ngon",
    "Quán mới mở, menu có khoảng 20 món, giá từ 30k đến 80k",
]

st.set_page_config(page_title="Phân tích cảm xúc bình luận", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 2.5rem;}
.card {background:#fff; border:1px solid #EADFCF; border-radius:26px; padding:26px 28px 14px;
       box-shadow:0 1px 3px rgba(74,52,40,.05); margin:8px 0 18px;}
.head {display:flex; gap:18px; justify-content:space-between; align-items:flex-start;}
.quote {font-size:21px; font-weight:700; line-height:1.45; color:#4A3428;}
.sub {font-size:13.5px; color:#9A8A7D; margin-top:8px;}
.stamp {flex:none; width:122px; height:122px; border-radius:50%; border:3px solid #6B5E55;
        box-shadow: inset 0 0 0 5px #fff, inset 0 0 0 6.5px #6B5E55; transform:rotate(-9deg);
        display:flex; flex-direction:column; align-items:center; justify-content:center; color:#5B4D44;}
.stamp .num {font-size:31px; font-weight:700; letter-spacing:-1px; line-height:1;}
.stamp .num small {font-size:17px; font-weight:600;}
.stamp .lab {font-size:10.5px; letter-spacing:2.5px; text-transform:uppercase; font-weight:700; margin-top:6px;}
.rule {border-top:1px solid #EFE6D8; margin:20px 0 4px;}
.row {display:flex; gap:16px; padding:13px 0;}
.badge {flex:none; width:40px; height:36px; border-radius:11px; display:flex; align-items:center;
        justify-content:center; font-weight:700; font-size:15.5px;}
.pos {background:#E3F1E4; color:#2F6B3E;} .neu {background:#EFEBE6; color:#6B5E55;}
.neg {background:#F7E3DE; color:#A2442F;} .none {background:#F7F4EF; color:#C2B6AB;}
.name {font-weight:700; font-size:16.5px; color:#4A3428;}
.desc {color:#6E5D52; font-size:15px; line-height:1.6; margin-top:2px;}
.note {font-size:13px; color:#A89A8E; line-height:1.55;}
</style>
""", unsafe_allow_html=True)


def _retrain(module):
    with st.spinner("Chưa có mô hình phù hợp với máy này, đang huấn luyện lại (vài phút)..."):
        __import__(module).main()


@st.cache_resource
def load_model(name):
    try:
        return SentimentModel(name)
    except Exception:  # chưa có mô hình hoặc khác phiên bản scikit-learn -> huấn luyện lại
        _retrain("experiments")
        return SentimentModel(name)


@st.cache_resource
def load_aspect_model(name):
    try:
        return AspectModel(name)
    except Exception:
        _retrain("train_aspect")
        return AspectModel(name)


def overall10(conf):
    """Điểm tổng /10 từ mô hình câu, đặt trong khoảng của nhãn mà mô hình chọn."""
    return band_score(conf["Tích cực"], conf["Trung tính"], conf["Tiêu cực"])


def row_html(name, score, tone, desc):
    s = "–" if score is None else f"{score:.1f}"
    return (f"<div class='row'><div class='badge {tone}'>{s}</div><div>"
            f"<div class='name'>{name}</div><div class='desc'>{html.escape(desc)}</div></div></div>")


def card_html(quote, score, label, sub, rows):
    return (f"<div class='card'><div class='head'><div><div class='quote'>{quote}</div>"
            f"<div class='sub'>{sub}</div></div>"
            f"<div class='stamp'><div class='num'>{score:.1f}<small>/10</small></div>"
            f"<div class='lab'>{label}</div></div></div>"
            f"<div class='rule'></div>{''.join(rows)}</div>")


def aspect_rows(results, desc_fn):
    rows = []
    for k in MAIN_KEYS + [a for a in ASPECT_KEYS if a not in MAIN_KEYS]:
        r = results.get(k)
        if r is None:
            if k in MAIN_KEYS:
                rows.append(row_html(ASPECT_NAMES[k], None, "none", "Không được nhắc tới."))
            continue
        rows.append(row_html(ASPECT_NAMES[k], r["scale"], TONE[r["label"]], desc_fn(k, r)))
    return rows


# ---------------- đầu trang ----------------
st.title("Phân tích cảm xúc bình luận")
st.markdown("<div class='note'>Bình luận quán ăn tiếng Việt · Naive Bayes và SVM · chấm điểm theo từng "
            "khía cạnh bằng mô hình học từ dữ liệu đã gán nhãn</div>", unsafe_allow_html=True)
choice = st.radio("Mô hình", ["SVM tuyến tính", "Naive Bayes"], horizontal=True, label_visibility="collapsed")
kind = "svm" if choice == "SVM tuyến tính" else "nb"
model = load_model(kind)
amodel = load_aspect_model(kind)

tab1, tab2, tab3 = st.tabs(["Một bình luận", "Nhiều bình luận", "Kết quả đánh giá"])

# ---------------- một bình luận ----------------
with tab1:
    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES[0]
    ex = st.selectbox("Câu mẫu", ["Tự nhập"] + EXAMPLES)
    if ex != "Tự nhập" and ex != st.session_state.get("last_ex"):
        st.session_state.text = ex
    st.session_state.last_ex = ex
    text = st.text_area("Bình luận", key="text", height=100)
    st.button("Phân tích", type="primary")
    if text.strip():
        r = model.predict(text)
        asp, probs = amodel.analyze(text)
        conf = r["confidence"]
        quote = html.escape(text.strip() if len(text) <= 160 else text.strip()[:157] + "…")
        sub = " · ".join(f"{k} {conf[k] * 100:.0f}%" for k in ORDER)
        score = overall10(conf)
        rows = aspect_rows(asp, lambda k, a: describe(a))
        st.markdown(card_html(f"“{quote}”", score, label_of_scale(score), sub, rows), unsafe_allow_html=True)
        st.markdown(f"<div class='note'>{SCALE_NOTE}<br>Điểm tổng lấy từ mô hình cảm xúc cả câu. Mỗi khía cạnh "
                    "có một bộ phân lớp (không nhắc / tích cực / trung tính / tiêu cực) học từ 2.001 bình luận đã "
                    "gán nhãn. Điểm nằm trong khoảng của nhãn mô hình chọn, cao hay thấp trong khoảng tuỳ độ tin "
                    "cậy. Căn cứ: vế câu mà khi bỏ đi thì xác suất của khía cạnh giảm nhiều nhất.</div>",
                    unsafe_allow_html=True)
        with st.expander("Chi tiết"):
            st.caption("Xác suất theo từng khía cạnh (ngưỡng 'có nhắc' chọn bằng kiểm định chéo)")
            st.dataframe(pd.DataFrame([{"Khía cạnh": ASPECT_NAMES[k], "Không nhắc": probs[k]["none"],
                                        "Tích cực": probs[k]["pos"], "Trung tính": probs[k]["neu"],
                                        "Tiêu cực": probs[k]["neg"],
                                        "Kết luận": LAB_VI[amodel.decide(probs[k], k)]} for k in ASPECT_KEYS])
                         .round(3), hide_index=True, width="stretch")
            evr = [{"Khía cạnh": ASPECT_NAMES[k], "Vế câu": e["clause"], "Hướng": LAB_VI[e["label"]],
                    "Mức đóng góp": e["weight"]} for k, a in asp.items() if a for e in a["evidence"]]
            if evr:
                st.caption("Căn cứ: bỏ vế câu ra thì xác suất giảm bao nhiêu")
                st.dataframe(pd.DataFrame(evr), hide_index=True, width="stretch")
            st.caption("Văn bản sau tiền xử lý")
            st.code(r["clean"] or "(rỗng)", language=None)

# ---------------- nhiều bình luận ----------------
with tab2:
    st.markdown("<div class='note'>Mỗi dòng một bình luận, hoặc tải tệp .txt / .csv (cột đầu tiên). "
                "Kết quả là điểm chung của cả quán.</div>", unsafe_allow_html=True)
    up = st.file_uploader("Tải tệp", type=["txt", "csv"], label_visibility="collapsed")
    many = st.text_area("Danh sách bình luận", value="\n".join(EXAMPLES[1:]), height=180)
    if st.button("Phân tích tất cả", type="primary"):
        if up is not None:
            if up.name.endswith(".csv"):
                lines = pd.read_csv(up).iloc[:, 0].astype(str).tolist()
            else:
                lines = up.read().decode("utf-8", errors="ignore").splitlines()
        else:
            lines = many.splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        if not lines:
            st.warning("Chưa có bình luận nào.")
        else:
            res = model.predict(lines)
            aps = amodel.proba(lines)
            scores = [overall10(r["confidence"]) for r in res]
            labels = [label_of_scale(x) for x in scores]
            counts = pd.Series(labels).value_counts().reindex(ORDER, fill_value=0)

            per = {k: [] for k in ASPECT_KEYS}   # điểm /10 của các bình luận có nhắc khía cạnh
            for p in aps:
                for k in ASPECT_KEYS:
                    if amodel.decide(p[k], k) != "none":
                        per[k].append(score10(p[k]))
            agg, stats = {}, {}
            for k, ss in per.items():
                if ss:
                    avg_k = round(sum(ss) / len(ss), 1)
                    agg[k] = {"scale": avg_k, "label": label_of_scale(avg_k)}
                    ls = [label_of_scale(x) for x in ss]
                    stats[k] = (len(ss), ls.count("Tích cực"), ls.count("Tiêu cực"))
            desc = lambda k, a: (f"Nhắc tới trong {stats[k][0]} bình luận: "
                                 f"{stats[k][1]} khen, {stats[k][2]} chê.")
            avg = sum(scores) / len(scores)
            sub = (f"{counts['Tích cực']} tích cực · {counts['Trung tính']} trung tính · "
                   f"{counts['Tiêu cực']} tiêu cực")
            st.markdown(card_html(f"{len(lines)} bình luận về quán", avg, label_of_scale(avg), sub,
                                  aspect_rows(agg, desc)), unsafe_allow_html=True)
            st.markdown(f"<div class='note'>{SCALE_NOTE}</div>", unsafe_allow_html=True)

            rows = []
            for r, p, sc, lb in zip(res, aps, scores, labels):
                row = {"Bình luận": r["text"], "Nhãn": lb, "Điểm /10": sc}
                for k in ASPECT_KEYS:
                    row[ASPECT_NAMES[k]] = None if amodel.decide(p[k], k) == "none" else score10(p[k])
                rows.append(row)
            df = pd.DataFrame(rows)
            with st.expander("Chi tiết từng bình luận"):
                st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Tải kết quả CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               "ket_qua_cam_xuc.csv", "text/csv")

# ---------------- kết quả đánh giá ----------------
with tab3:
    path = ROOT / "results" / "results.json"
    apath = ROOT / "results" / "aspect_results.json"
    rpath = ROOT / "data" / "aspect" / "annotation_report.json"
    if path.exists():
        R = json.loads(path.read_text(encoding="utf-8"))
        d = R["data"]
        rows = []
        for k, v in R["models"].items():
            t = v["test"]
            rows.append(row_html(v["title"], t["macro_f1"] * 10, "neu",
                                 f"Accuracy {t['accuracy'] * 100:.2f}% · Macro-F1 {t['macro_f1'] * 100:.2f}% · "
                                 f"Precision {t['precision'] * 100:.2f}% · Recall {t['recall'] * 100:.2f}%"))
        st.markdown(f"<div class='card'><div class='name'>Cảm xúc cả câu · tập kiểm tra Foody</div>"
                    f"<div class='note'>{d['train_size']:,} huấn luyện · {d['test_size']:,} kiểm tra. Ô điểm là "
                    f"Macro-F1 quy về thang 10.</div>".replace(",", ".") +
                    f"<div class='rule'></div>{''.join(rows)}</div>", unsafe_allow_html=True)
    if apath.exists():
        A = json.loads(apath.read_text(encoding="utf-8"))
        rows = []
        for k in ["svm", "nb", "keyword"]:
            if k not in A["models"]:
                continue
            o = A["models"][k]["overall"]
            am = A["models"][k].get("argmax")
            extra = (f" · nếu không dùng ngưỡng chọn bằng kiểm định chéo: F1 cặp {am['tuple_f1'] * 100:.2f}%, "
                     f"F1 phát hiện {am['detect_f1'] * 100:.2f}%") if am else ""
            rows.append(row_html(A["titles"][k], o["tuple_f1"] * 10, "neu",
                                 f"(khía cạnh, cảm xúc) F1 {o['tuple_f1'] * 100:.2f}% · phát hiện khía cạnh "
                                 f"F1 {o['detect_f1'] * 100:.2f}% (P {o['detect_p'] * 100:.1f}%, "
                                 f"R {o['detect_r'] * 100:.1f}%){extra}"))
        st.markdown(f"<div class='card'><div class='name'>Cảm xúc theo khía cạnh · {A['n_test']} bình luận kiểm "
                    f"tra đã gán nhãn</div><div class='note'>Huấn luyện trên {A['n_train']} bình luận đã gán nhãn. "
                    f"Ô điểm là F1 của cặp (khía cạnh, cảm xúc) quy về thang 10.</div>"
                    f"<div class='rule'></div>{''.join(rows)}</div>", unsafe_allow_html=True)
        with st.expander("Chi tiết theo từng khía cạnh"):
            tbl = []
            for k in ASPECT_KEYS:
                row = {"Khía cạnh": ASPECT_NAMES[k],
                       "Số lần nhắc (test)": A["models"]["svm"]["per_aspect"][k]["support_mention"]}
                for m in ["svm", "nb", "keyword"]:
                    if m in A["models"]:
                        pa = A["models"][m]["per_aspect"][k]
                        row[f"{A['titles'][m]} · F1 cặp"] = round(pa["tuple_f1"] * 100, 1)
                        row[f"{A['titles'][m]} · F1 phát hiện"] = round(pa["detect_f1"] * 100, 1)
                tbl.append(row)
            st.dataframe(pd.DataFrame(tbl), hide_index=True, width="stretch")
    if rpath.exists():
        rep = json.loads(rpath.read_text(encoding="utf-8"))
        with st.expander("Dữ liệu gán nhãn và độ đồng thuận"):
            st.markdown(f"<div class='note'>{rep['n']} bình luận × 6 khía cạnh, gán nhãn kép độc lập (A, B) theo "
                        f"hướng dẫn trong data/aspect/HUONG_DAN_GAN_NHAN.md; {rep['n_disagree_cells']} ô bất đồng "
                        f"được phân xử.</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Khía cạnh": ASPECT_NAMES[k], "Tỉ lệ khớp A–B": round(v["agreement"], 3),
                                        "Cohen's kappa (4 nhãn)": round(v["kappa"], 3),
                                        "Kappa (có nhắc / không)": round(v["kappa_mention"], 3)}
                                       for k, v in rep["agreement"].items()]), hide_index=True,
                         width="stretch")
