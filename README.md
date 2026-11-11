# Đối chiếu tóm tắt: Seq2Seq (226.8M) và ViT5-base (225.9M)

> 📖 **Tài liệu đầy đủ cho thành viên nhóm:** Xem chi tiết quy định đề tài, nơi tải model trên Drive và hướng dẫn từng bước tại **[HUONG_DAN_VA_YEU_CAU.md](HUONG_DAN_VA_YEU_CAU.md)**.

Hệ thống web để chạy hai mô hình đã huấn luyện trên cùng một bài báo, xem hai bản tóm tắt cạnh nhau,
chấm ROUGE khi có bản tham chiếu, bình chọn (kể cả chế độ mù) và lưu mọi lượt vào cơ sở dữ liệu.

```
frontend/   giao diện (HTML, CSS, JS thuần, không cần build)
backend/    FastAPI + SQLAlchemy
  app/models/      adapter cho từng mô hình (seq2seq, vit5) và mô hình giả
  app/vendor/      seq2seq.py nguyên bản lấy từ notebook huấn luyện
  app/scoring.py   ROUGE theo quy tắc chung của bài
data/       app.db (SQLite) và test.csv
```

## Chạy thử ngay bằng mô hình giả

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
USE_MOCK_MODELS=1 uvicorn app.main:app --port 8000      # Windows PowerShell: $env:USE_MOCK_MODELS=1
```

Mở http://localhost:8000. Mô hình giả chỉ lấy vài câu đầu của bài, dùng để xem giao diện hoạt động.

## Cắm hai mô hình thật

1. Chép từ Drive về máy: thư mục run Seq2Seq (cần `best.pt` và `tokenizer.model`) và thư mục checkpoint ViT5.
2. `pip install -r backend/requirements-models.txt` (cài `torch` theo pytorch.org cho đúng GPU/CPU).
3. `cp .env.example .env` rồi sửa `SEQ2SEQ_RUN_DIR`, `VIT5_CHECKPOINT_DIR`, `VIT5_INPUT_TEMPLATE`.
4. Chép file CSV test (cột `title`, `article`, `summary`) vào `data/test.csv` để dùng trang Bài mẫu.
5. `cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000`.

Hai chip ở góc phải màn hình báo trạng thái nạp mô hình. Nếu một mô hình lỗi, bấm vào chip để xem
nguyên nhân và nạp lại; mô hình còn lại vẫn chạy bình thường.

### Hai điểm cần đối chiếu với lúc huấn luyện

- **`VIT5_INPUT_TEMPLATE`** phải giống đúng cách ghép tiêu đề và bài báo trong notebook ViT5. Giá trị mặc định
  chỉ là phỏng đoán theo định dạng của Seq2Seq.
- **Tách từ cho ViT5.** ViT5 được fine-tune trên văn bản đã tách từ (có `_`). Bài mẫu từ CSV được đưa vào mô hình
  nguyên bản đã tách từ. Với văn bản người dùng tự dán, nên cài `pyvi` hoặc `underthesea` và đặt
  `VIT5_WORD_SEGMENTER`, rồi so sánh chất lượng có và không tách từ.

## PostgreSQL (tùy chọn)

```bash
docker compose up -d db
# trong .env: DATABASE_URL=postgresql+psycopg2://app:app@localhost:5432/summarizer
pip install psycopg2-binary
```

Bảng được tạo tự động khi khởi động.

## API chính

| Đường dẫn | Việc |
|---|---|
| `POST /api/summarize` | Chạy cả hai mô hình, lưu và trả kết quả (ROUGE nếu có `reference`) |
| `POST /api/submissions/{id}/vote` | Ghi bình chọn `seq2seq`, `vit5` hoặc `tie` |
| `GET /api/submissions`, `/api/submissions/{id}` | Lịch sử |
| `GET /api/stats` | Thời gian, tỉ lệ cắt, ROUGE trung bình, tổng bình chọn |
| `GET /api/samples`, `/api/samples/{id}` | Bài mẫu từ tập test |
| `GET /api/models`, `POST /api/models/{key}/reload` | Trạng thái và nạp lại mô hình |

Tài liệu tương tác: http://localhost:8000/docs

## Kiểm thử

```bash
cd backend && pip install pytest httpx && python -m pytest -q tests
```

Bộ kiểm thử dùng mô hình giả, kiểm tra luồng tóm tắt, ROUGE, bình chọn, thống kê, bài mẫu và việc một mô hình lỗi
không làm hỏng mô hình kia. Nó không kiểm tra chất lượng của checkpoint thật.

## Lưu ý khi diễn giải kết quả so sánh

- Hai mô hình đều có giới hạn đầu vào 1.024 token (Seq2Seq 1.024 subword BPE, ViT5 1.024 token); giao diện báo khi đầu vào bị cắt.
- Thời gian trong giao diện là thời gian sinh một bài (batch 1), không gồm thời gian nạp mô hình.
- Cột "Số không có trong bài gốc" chỉ là cảnh báo tự động theo chữ số, không phải kiểm chứng sự thật.
- ROUGE trong hệ thống dùng đúng tokenizer của bài (NFC, casefold, bỏ `_`, `[^\W_]+`, ROUGE-L, F1 × 100).
