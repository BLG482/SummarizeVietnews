"""Cấu hình đọc từ biến môi trường (hoặc file .env ở thư mục gốc dự án)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except ImportError:  # python-dotenv chỉ là tiện ích, không bắt buộc
    env_file = ROOT / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0].strip().strip("'\"")
            os.environ.setdefault(k.strip(), v)


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'app.db'}")
    use_mock = _flag("USE_MOCK_MODELS")
    preload = _flag("PRELOAD_MODELS", "1")
    device = os.getenv("DEVICE", "auto")

    # Seq2Seq: thư mục run chứa best.pt và tokenizer.model
    seq2seq_run_dir = os.getenv("SEQ2SEQ_RUN_DIR", "")
    seq2seq_beam = int(os.getenv("SEQ2SEQ_BEAM", "4"))

    # ViT5: thư mục checkpoint (vd. best_model-9375) chứa config, tokenizer, trọng số
    vit5_dir = os.getenv("VIT5_CHECKPOINT_DIR", "")
    vit5_input_template = os.getenv(
        "VIT5_INPUT_TEMPLATE", "Tiêu đề: {title}\\nNội dung: {article}"
    ).replace("\\n", "\n")
    vit5_max_input = int(os.getenv("VIT5_MAX_INPUT_TOKENS", "1024"))
    vit5_max_new = int(os.getenv("VIT5_MAX_NEW_TOKENS", "256"))
    vit5_beam = int(os.getenv("VIT5_BEAM", "4"))
    vit5_segmenter = os.getenv("VIT5_WORD_SEGMENTER", "none").strip().lower()

    _csv_env = os.getenv("TESTSET_CSV", "").strip()
    if _csv_env:
        _csv_path = Path(_csv_env)
        if not _csv_path.is_absolute() and not _csv_path.exists():
            _csv_path = (ROOT / _csv_path).resolve()
        testset_csv = str(_csv_path)
    else:
        testset_csv = str(ROOT / "data" / "test.csv")
    frontend_dir = ROOT / "frontend"


settings = Settings()
