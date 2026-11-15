FROM python:3.11-slim

WORKDIR /app

# Cài đặt thư viện backend và mô hình (bản PyTorch CPU nhẹ cho server)
COPY backend/requirements.txt backend/requirements-models.txt ./backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r backend/requirements.txt -r backend/requirements-models.txt huggingface_hub

# Sao chép mã nguồn và dữ liệu
COPY backend ./backend
COPY frontend ./frontend
COPY data ./data

# Tạo thư mục lưu trữ và cấp quyền cho user 1000 (chuẩn Hugging Face Spaces)
RUN mkdir -p /app/data /tmp/hf_cache && chmod -R 777 /app/data /tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV PORT=7860

EXPOSE 7860

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

