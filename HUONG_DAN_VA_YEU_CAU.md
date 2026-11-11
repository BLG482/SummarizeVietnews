# HƯỚNG DẪN SỬ DỤNG & QUY ĐỊNH YÊU CẦU CỦA ĐỀ TÀI
**Đề tài:** Đối chiếu tóm tắt văn bản tiếng Việt: **BiGRU Seq2Seq + Luong Attention (226.8M)** và **ViT5-base (225.9M)**

Tài liệu này tổng hợp đầy đủ **yêu cầu kỹ thuật của đề tài**, **quy chuẩn đối chiếu hai mô hình**, **những file cần tải từ Google Drive**, và **cách cài đặt – chạy Web từ A đến Z** để các thành viên trong nhóm dễ dàng sử dụng.

---

## 1. Tóm tắt yêu cầu & Quy chuẩn chung của đề tài

Để phép so sánh giữa mô hình tự huấn luyện từ đầu (**Seq2Seq BiGRU**) và mô hình Pre-trained Transformer (**ViT5-base**) công bằng và chính xác, toàn bộ hệ thống tuân thủ nghiêm ngặt các quy định sau:

| Tiêu chí | Mô hình 1: BiGRU Seq2Seq (v2) | Mô hình 2: ViT5-base (Fine-tuned) |
| :--- | :--- | :--- |
| **Kiến trúc** | BiGRU 2 tầng (`embedding=1920`, `hidden=1920`) + Scaled Luong Attention ($1/\sqrt{1920}$) + Pre-Tanh LayerNorm | Transformer Encoder-Decoder (`VietAI/vit5-base`), 12 tầng Encoder + 12 tầng Decoder |
| **Số lượng tham số** | **226,798,560** (~226.80M — huấn luyện từ đầu) | **225,950,976** (~225.95M — chênh lệch chỉ 0.38%) |
| **Tập dữ liệu chung** | VietNews (Gói dữ liệu v2): **50.000** bài Train / **3.000** bài Validation / **2.000** bài Test | Cùng tập dữ liệu **50.000** bài Train / **3.000** Val / **2.000** Test |
| **Tokenizer** | SentencePiece BPE `12.000` subword | SentencePiece Unigram/BPE của ViT5 (`spiece.model`) |
| **Định dạng đầu vào** | `Tiêu đề: {title} Nội dung: {article}` (tự chuẩn hóa NFC, bỏ `_`, tách dấu câu) | `Tiêu đề: {title}\nNội dung: {article}` (xuống dòng giữa tiêu đề và nội dung) |
| **Giới hạn đầu vào** | Tối đa **1.024** subword token | Tối đa **1.024** subword token |
| **Cấu hình sinh (Decode)** | `beam_size = 4`, `max_target = 128`, `min_new = 8`, `no_repeat = 4`, `alpha = 0.6` | `num_beams = 4`, `max_new_tokens = 256`, `no_repeat_ngram_size = 3`, `length_penalty = 1.0` |
| **Checkpoint tốt nhất** | `best.pt` tại **Epoch 6 / 10** (ROUGE-1 Test: `32.00`, ROUGE-2: `17.69`, ROUGE-L: `25.16`) | `best_model-9375` tại **Epoch 3.0** (`global_step = 9375`) |

### Quy tắc chấm điểm ROUGE chung (`backend/app/scoring.py`)
- Chuẩn hóa chuỗi theo Unicode **NFC**, chuyển chữ thường (`casefold`), đổi dấu gạch dưới `_` thành khoảng trắng.
- Tách âm tiết bằng biểu thức chính quy Unicode `[^\W_]+` (không dùng stemmer tiếng Anh để không làm mất dấu tiếng Việt).
- Tính điểm **F1-measure** của **ROUGE-1, ROUGE-2, ROUGE-L** trên thang `0 – 100`.
- **Kiểm tra số liệu lạ (Unsupported Numbers):** Tự động trích xuất mọi chuỗi chữ số (`\d+`) trong bản tóm tắt và đối chiếu với bài báo gốc để cảnh báo nếu mô hình tự bịa ra con số mới (Hallucination).

---

## 2. Những file cần chuẩn bị (Tải từ Google Drive của nhóm)

Vì trọng số mô hình và tập dữ liệu rất nặng (~3.6 GB) nên không đẩy lên GitHub. Khi clone code về máy, bạn cần tải 3 mục sau từ Google Drive của nhóm:

1. **Thư mục Checkpoint Seq2Seq (`run_base_225m_v2_10ep`):**
   - Đường dẫn trên Drive: `MyDrive/seq2seq_150m_colab_3ep_v1/run_base_225m_v2_10ep`
   - Chỉ cần tải **2 file**:
     - `best.pt` (~2.6 GB)
     - `tokenizer.model` (~450 KB)
   - Lưu vào một thư mục trên máy (ví dụ: `D:/seq2seq_226m`).

2. **Thư mục Checkpoint ViT5 (`best_model-9375`):**
   - Đường dẫn trên Drive: `MyDrive/NCKH/vit5_50000_output_seed42_20261011/best_models/best_model-9375`
   - Tải nguyên thư mục `best_model-9375` (chứa `config.json`, `generation_config.json`, `model.safetensors`, `spiece.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`).
   - Giải nén ra máy (ví dụ: `D:/best_model-9375`).

3. **File dữ liệu Bài mẫu (`test_2000.csv`):**
   - Đường dẫn trên Drive: `MyDrive/Du_lieu_va_quy_dinh_danh_gia_chung_20261011_v2/test_2000.csv`
   - Tải về, đổi tên thành **`test.csv`** và bỏ vào thư mục **`data/test.csv`** trong dự án.

---

## 3. Hướng dẫn cài đặt môi trường (Chỉ làm 1 lần đầu)

Yêu cầu máy đã cài **Python 3.10 – 3.13**. Mở **Terminal (PowerShell)** tại thư mục gốc của dự án và chạy:

```powershell
# 1. Tạo môi trường ảo .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt các thư viện Backend và AI
pip install -r backend/requirements.txt
pip install -r backend/requirements-models.txt
pip install safetensors
```

---

## 4. Cấu hình file `.env`

1. Nhân bản file `.env.example` thành file `.env` ở thư mục gốc dự án:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Mở file `.env` và sửa 2 dòng đường dẫn trỏ đúng tới nơi bạn đã lưu mô hình ở **Mục 2**:
   ```env
   SEQ2SEQ_RUN_DIR=D:/seq2seq_226m
   SEQ2SEQ_BEAM=4

   VIT5_CHECKPOINT_DIR=D:/best_model-9375
   VIT5_INPUT_TEMPLATE=Tiêu đề: {title}\nNội dung: {article}
   VIT5_MAX_INPUT_TOKENS=1024
   VIT5_MAX_NEW_TOKENS=256
   VIT5_BEAM=4
   VIT5_WORD_SEGMENTER=none

   DEVICE=auto
   PRELOAD_MODELS=1
   TESTSET_CSV=./data/test.csv
   ```

> **Mẹo khi chưa tải kịp mô hình nặng (Chạy thử giao diện bằng Mock):**
> Nếu chỉ muốn xem trước giao diện web mà chưa tải 3.6 GB model về máy, bạn chỉ cần bỏ dấu `#` ở dòng cuối trong `.env`:
> `USE_MOCK_MODELS=1`

---

## 5. Cách khởi động chương trình (Siêu ngắn gọn)

Sau khi đã cài đặt xong, mỗi lần muốn mở web bạn chỉ cần làm **1 trong 3 cách**:

- **Cách 1 (Trong Terminal):**
  ```powershell
  .\run.bat
  ```
- **Cách 2 (Bấm nút Run trong VS Code):** Mở file `run.py` và bấm nút **▶️ (Run Python File)**, hoặc gõ `python run.py`.
- **Cách 3 (Ngoài File Explorer):** Nhấp đúp chuột trực tiếp vào file **`run.bat`** trong thư mục dự án.

Khi Terminal hiện:
```text
[Seq2Seq] Đã nạp xong (epoch 6, device=cpu).
[ViT5] Đã nạp xong (device=cpu).
```
Hãy mở trình duyệt truy cập vào: 👉 **http://localhost:8000**

---

## 6. Cách sử dụng các chức năng trên trang Web

1. **Tab "Bài mẫu":**
   - Hiển thị 2.000 bài báo trong tập test chuẩn (`data/test.csv`).
   - Có thể tìm kiếm theo tiêu đề, bấm **"Đưa vào ô so sánh"** để nạp tự động Tiêu đề + Nội dung + Tóm tắt tham chiếu (Reference) sang trang So sánh.
   - *Lưu ý kỹ thuật:* Khi chạy bài mẫu nguyên bản từ CSV, hệ thống tự động truyền bản gốc có dấu `_` vào mô hình (`raw=True`) để kết quả điểm ROUGE khớp tuyệt đối với lúc đánh giá trên Colab.

2. **Tab "So sánh":**
   - Dán bài báo bất kỳ hoặc lấy từ *Bài mẫu*, bấm **"Chạy hai mô hình"**.
   - Xem trực quan 2 bản tóm tắt đặt cạnh nhau kèm: **Thời gian chạy**, **Số token đầu vào**, **Cảnh báo cắt bớt (> 1.024 token)**, **Điểm ROUGE-1/2/L**, và **Cảnh báo số liệu không có trong bài gốc**.
   - **Chế độ bình chọn mù (Blind Mode):** Gạt nút *"Ẩn tên mô hình (bình chọn mù)"* để hệ thống tráo ngẫu nhiên vị trí trái/phải thành `Mô hình A` / `Mô hình B` và giấu bảng thông số. Sau khi bạn bấm bình chọn, hệ thống mới lật mở tên thật của hai mô hình và lưu kết quả vào `data/app.db`.

3. **Tab "Lịch sử":**
   - Lưu lại toàn bộ các lượt chạy tóm tắt và bình chọn trong cơ sở dữ liệu SQLite (`data/app.db`). Bấm vào bất kỳ lượt nào để xem lại chi tiết.

4. **Tab "Thống kê":**
   - Tổng hợp toàn bộ số lượt chạy, điểm ROUGE-1 / ROUGE-2 / ROUGE-L trung bình của từng mô hình, thời gian sinh trung bình, tỉ lệ bài bị cắt token và biểu đồ tỉ lệ thắng/thua/hòa từ bình chọn của người dùng.
