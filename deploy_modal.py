"""Triển khai Web Tóm tắt 24/7 lên Modal.com (Miễn phí $30/tháng, 8GB RAM, tự bật khi có người truy cập).
Sử dụng trực tiếp 2 mô hình đã upload lên Hugging Face Hub của bạn!

Cách chạy (trong Terminal):
  1. .\.venv\Scripts\python.exe -m pip install modal
  2. .\.venv\Scripts\python.exe -m modal setup
  3. .\.venv\Scripts\python.exe -m modal deploy deploy_modal.py
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent

# Username Hugging Face chứa 2 model đã upload
HF_USERNAME = "baoluu22"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "sqlalchemy>=2.0",
        "rouge-score==0.1.2",
        "python-dotenv>=1.0",
        "transformers>=4.53,<5",
        "sentencepiece>=0.2,<0.3",
        "numpy>=1.26",
        "tqdm>=4.66",
        "safetensors>=0.4",
        "huggingface_hub>=0.23",
    )
    .run_commands(
        f'python -c "from huggingface_hub import snapshot_download; '
        f'snapshot_download(\'{HF_USERNAME}/seq2seq-vietnews-226m\', local_dir=\'/models/seq2seq\'); '
        f'snapshot_download(\'{HF_USERNAME}/vit5-vietnews-9375\', local_dir=\'/models/vit5\')"'
    )
    .add_local_dir(str(ROOT / "backend"), remote_path="/app/backend")
    .add_local_dir(str(ROOT / "frontend"), remote_path="/app/frontend")
    .add_local_file(str(ROOT / "data" / "test.csv"), remote_path="/app/data/test.csv")
)

app = modal.App("vietnews-summarizer", image=image)
volume = modal.Volume.from_name("vietnews-summarizer-data", create_if_missing=True)


@app.function(
    cpu=4.0,
    memory=8192,  # 8 GB RAM
    volumes={"/data_vol": volume},
    min_containers=0,  # Tự ngủ khi không có ai vào (tốn $0), tự thức dậy 24/7 khi có người mở link!
    timeout=600,
)
@modal.asgi_app()
def web():
    import os
    import shutil
    import sys
    import threading
    from sqlalchemy import event

    os.makedirs("/data_vol", exist_ok=True)
    if os.path.exists("/data_vol/app.db") and not os.path.exists("/tmp/app.db"):
        try:
            shutil.copyfile("/data_vol/app.db", "/tmp/app.db")
        except Exception:
            pass

    os.environ["SEQ2SEQ_RUN_DIR"] = "/models/seq2seq"
    os.environ["VIT5_CHECKPOINT_DIR"] = "/models/vit5"
    os.environ["DATABASE_URL"] = "sqlite:////tmp/app.db"
    os.environ["TESTSET_CSV"] = "/app/data/test.csv"
    os.environ["HF_HOME"] = "/tmp/hf_cache"

    sys.path.insert(0, "/app/backend")
    from app.db import Base, SessionLocal, engine
    from app.main import _preload, app as fastapi_app

    Base.metadata.create_all(engine)

    @event.listens_for(SessionLocal, "after_commit")
    def _backup_db(_session):
        try:
            shutil.copyfile("/tmp/app.db", "/data_vol/app.db")
            volume.commit()
        except Exception:
            pass

    threading.Thread(target=_preload, daemon=True).start()

    return fastapi_app
