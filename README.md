# Phân tích cảm xúc bình luận Foody – triển khai trên Streamlit Community Cloud

Web app phân loại bình luận quán ăn Foody thành **Tích cực / Trung tính / Tiêu cực** và chấm điểm 6 khía cạnh
(Đồ ăn, Giá cả, Phục vụ, Không gian, Vệ sinh, Vị trí) bằng Naive Bayes và SVM.

- **Input:** một hoặc nhiều bình luận tiếng Việt (gõ trực tiếp, hoặc tải tệp .txt/.csv).
- **Output:** nhãn cảm xúc, điểm /10, điểm từng khía cạnh, có thể xuất CSV.

## Chạy thử trên máy

```
pip install -r requirements.txt
streamlit run app.py
```

## Triển khai lên Streamlit Community Cloud

1. Đưa thư mục này lên một repo GitHub.
2. Vào share.streamlit.io, đăng nhập bằng GitHub, bấm **Create app**.
3. Chọn repo, branch `main`, main file `app.py`. Trong **Advanced settings**, chọn Python cùng bản với máy (3.11).
4. Bấm **Deploy**.

## Cấu trúc

```
app.py               giao diện web (Streamlit)
requirements.txt     thư viện, đã ghim phiên bản
.streamlit/          theme giao diện
src/                 tiền xử lý, mô hình, script huấn luyện lại (dùng khi không nạp được mô hình)
models/              4 mô hình đã huấn luyện (.joblib)
results/             số liệu đánh giá hiển thị ở thẻ "Kết quả đánh giá"
data/foody/          dữ liệu Foody (để huấn luyện lại nếu cần)
data/aspect/         nhãn khía cạnh + báo cáo độ đồng thuận
```
