"""Script tự động đẩy toàn bộ 2 mô hình (Seq2Seq + ViT5) và Web lên Hugging Face Spaces chạy 24/7.

Cách chạy:
  .\.venv\Scripts\python.exe deploy_hf.py
"""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

import torch
from huggingface_hub import HfApi
from app.config import settings
from app.vendor.seq2seq import load_model


def ensure_slim_seq2seq(run_dir: Path) -> Path:
    infer_path = run_dir / "best_infer.pt"
    if infer_path.is_file() and infer_path.stat().st_size > 100_000_000:
        print(f"[1/4] Đã có sẵn {infer_path} ({infer_path.stat().st_size / 1024**2:.1f} MB).")
        return infer_path
    print("[1/4] Đang rút gọn D:/seq2seq_226m/best.pt (2.72 GB) -> best_infer.pt (~865 MB)...")
    model, _tok, info = load_model(run_dir, torch.device("cpu"))
    sha_tok = (run_dir / "tokenizer.model").read_bytes()
    torch.save({
        "epoch": info["epoch"],
        "config": info["config"],
        "tokenizer_sha256": hashlib.sha256(sha_tok).hexdigest(),
        "model": model.state_dict(),
    }, infer_path)
    del model
    print(f"      -> Đã tạo {infer_path} ({infer_path.stat().st_size / 1024**2:.1f} MB)!")
    return infer_path


def main():
    print("=" * 65)
    print("  TRIỂN KHAI TỰ ĐỘNG WEB TÓM TẮT LÊN HUGGING FACE SPACES (24/7)")
    print("=" * 65)
    print("Bước chuẩn bị: Lấy Access Token (quyền Write) tại:")
    print("  👉 https://huggingface.co/settings/tokens\n")
    token = input("Dán Hugging Face Token (bắt đầu bằng hf_...) vào đây rồi Enter: ").strip()
    if not token:
        print("Bạn chưa nhập token. Hủy thao tác.")
        return

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    print(f"\n✅ Đã đăng nhập thành công tài khoản Hugging Face: {username}\n")

    seq2seq_dir = Path(settings.seq2seq_run_dir)
    vit5_dir = Path(settings.vit5_dir)
    if not seq2seq_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục Seq2Seq: {seq2seq_dir}")
    if not vit5_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ViT5: {vit5_dir}")

    # 1. Rút gọn best.pt -> best_infer.pt
    infer_path = ensure_slim_seq2seq(seq2seq_dir)

    # 2. Upload Seq2Seq model
    seq2seq_repo = f"{username}/seq2seq-vietnews-226m"
    print(f"\n[2/4] Đang tạo và đẩy mô hình Seq2Seq lên {seq2seq_repo}...")
    api.create_repo(repo_id=seq2seq_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(infer_path),
        path_in_repo="best_infer.pt",
        repo_id=seq2seq_repo,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(seq2seq_dir / "tokenizer.model"),
        path_in_repo="tokenizer.model",
        repo_id=seq2seq_repo,
        repo_type="model",
    )
    print("      -> Đã đẩy xong Seq2Seq!")

    # 3. Upload ViT5 model
    vit5_repo = f"{username}/vit5-vietnews-9375"
    print(f"\n[3/4] Đang tạo và đẩy mô hình ViT5 lên {vit5_repo}...")
    api.create_repo(repo_id=vit5_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(vit5_dir),
        repo_id=vit5_repo,
        repo_type="model",
    )
    print("      -> Đã đẩy xong ViT5!")

    # 4. Tạo Docker Space và đẩy code web
    space_repo = f"{username}/vietnews-summarizer"
    print(f"\n[4/4] Đang tạo Hugging Face Space (Docker 16GB RAM) tại {space_repo}...")
    api.create_repo(
        repo_id=space_repo,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    api.add_space_variable(repo_id=space_repo, key="SEQ2SEQ_RUN_DIR", value=seq2seq_repo)
    api.add_space_variable(repo_id=space_repo, key="VIT5_CHECKPOINT_DIR", value=vit5_repo)

    # Tạo README.md có YAML frontmatter chuẩn của Hugging Face Spaces nếu cần
    readme_text = ROOT / "README.md"
    orig_readme = readme_text.read_text(encoding="utf-8")
    hf_header = (
        "---\n"
        "title: Đối Chiếu Tóm Tắt VietNews (Seq2Seq vs ViT5)\n"
        "emoji: 📝\n"
        "colorFrom: blue\n"
        "colorTo: indigo\n"
        "sdk: docker\n"
        "app_port: 7860\n"
        "---\n\n"
    )
    try:
        if not orig_readme.startswith("---"):
            readme_text.write_text(hf_header + orig_readme, encoding="utf-8")
        api.upload_folder(
            folder_path=str(ROOT),
            repo_id=space_repo,
            repo_type="space",
            ignore_patterns=[
                ".venv/*",
                ".venv/**",
                ".git/*",
                ".git/**",
                "__pycache__/*",
                "**/__pycache__/*",
                ".pytest_cache/*",
                ".env",
                "generate_doc.py",
                "data/*.db",
            ],
        )
    finally:
        readme_text.write_text(orig_readme, encoding="utf-8")

    space_url = f"https://huggingface.co/spaces/{space_repo}"
    direct_domain = f"https://{username.lower()}-vietnews-summarizer.hf.space"
    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT TRIỂN KHAI LÊN CLOUD 24/7!")
    print(f"👉 Trang quản lý & xem quá trình Build: {space_url}")
    print(f"👉 Tên miền Web trực tiếp (mở sau ~3 phút build): {direct_domain}")
    print("=" * 65)


if __name__ == "__main__":
    main()

