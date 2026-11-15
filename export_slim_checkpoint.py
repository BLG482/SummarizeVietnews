"""Script rút gọn file D:/seq2seq_226m/best.pt (2.72 GB) xuống best_infer.pt (~860 MB)
để dễ dàng upload lên Hugging Face / Cloud Server.
Chạy lệnh: .\.venv\Scripts\python.exe export_slim_checkpoint.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

import torch
from app.config import settings
from app.vendor.seq2seq import load_model

if __name__ == "__main__":
    run_dir = Path(settings.seq2seq_run_dir)
    if not run_dir.exists():
        print(f"Không tìm thấy thư mục SEQ2SEQ_RUN_DIR: {run_dir}")
        sys.exit(1)
    model, _tok, info = load_model(run_dir, torch.device("cpu"))
    sha_tok = (run_dir / "tokenizer.model").read_bytes()
    import hashlib
    out_path = run_dir / "best_infer.pt"
    print(f"Đang lưu {out_path} (~860 MB)...")
    torch.save({
        "epoch": info["epoch"],
        "config": info["config"],
        "tokenizer_sha256": hashlib.sha256(sha_tok).hexdigest(),
        "model": model.state_dict(),
    }, out_path)
    sz_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"Hoàn tất! Đã tạo {out_path} ({sz_mb:.1f} MB).")

