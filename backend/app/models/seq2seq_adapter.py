"""Adapter Seq2Seq (BiGRU + Luong attention). Dùng nguyên mã seq2seq.py đã huấn luyện
(app/vendor/seq2seq.py) và gọi giống cell demo trong notebook."""
from ..config import settings
from .base import Summarizer, Summary


class Seq2SeqSummarizer(Summarizer):
    key = "seq2seq"
    label = "Seq2Seq"
    description = "BiGRU + Scaled Luong attention (226.8M, v2), huấn luyện từ đầu, SentencePiece BPE 12.000."
    input_limit = 1024

    def _load(self) -> None:
        if not settings.seq2seq_run_dir:
            raise RuntimeError("Chưa đặt SEQ2SEQ_RUN_DIR (thư mục có best.pt và tokenizer.model).")
        from pathlib import Path
        from ..vendor import seq2seq as s2s

        run_dir = settings.seq2seq_run_dir
        if not Path(run_dir).exists() and "/" in run_dir:
            import tempfile
            from huggingface_hub import snapshot_download
            target_dir = Path(tempfile.gettempdir()) / "hf_seq2seq_model"
            print(f"[Seq2Seq] Đang tải checkpoint từ Hugging Face Hub ({run_dir})...", flush=True)
            run_dir = snapshot_download(
                repo_id=run_dir,
                local_dir=str(target_dir),
                allow_patterns=["best_infer.pt", "best.pt", "tokenizer.model", "config.json"],
            )

        self._s2s = s2s
        device = s2s.device_for(settings.device)
        self.model, self.tokenizer, checkpoint = s2s.load_model(run_dir, device)
        self.device = str(device)
        self.input_limit = int(getattr(self.model.cfg, "max_source", 1024))
        self.epoch = checkpoint.get("epoch")
        self.description += f" Checkpoint epoch {self.epoch}."

    def _summarize(self, title: str, article: str, *, raw: bool) -> Summary:
        import time
        import torch

        t0 = time.perf_counter()
        print(f"[Seq2Seq] Đang sinh tóm tắt (beam={settings.seq2seq_beam})...", flush=True)
        # Cùng định dạng đầu vào với lúc huấn luyện; generate() tự chuẩn hóa (bỏ '_', tách dấu câu).
        source = f"Tiêu đề: {title.strip()} Nội dung: {article.strip()}"
        with torch.no_grad():
            out = self._s2s.generate(self.model, self.tokenizer, source,
                                     beam_size=settings.seq2seq_beam)
        print(f"[Seq2Seq] Xong trong {time.perf_counter() - t0:.2f}s.", flush=True)
        return Summary(text=out["summary"], truncated=bool(out["source_truncated"]),
                       input_tokens=out["input_tokens_before_truncation"])
