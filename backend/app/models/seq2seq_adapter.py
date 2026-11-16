"""Adapter Seq2Seq (BiGRU + Luong attention). Dùng nguyên mã seq2seq.py đã huấn luyện
(app/vendor/seq2seq.py) và gọi giống cell demo trong notebook."""
from ..config import settings
from .base import Summarizer, Summary, finalize_three_points, semantic_chunk_article


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
        full_source = f"Tiêu đề: {title.strip()} Nội dung: {article.strip()}"
        full_ids = self.tokenizer.encode(self._s2s.normalize(full_source), out_type=int)
        total_tokens = len(full_ids) + 1
        truncated = len(full_ids) > self.model.cfg.max_source - 1

        sections = semantic_chunk_article(title, article, target_points=3)
        beam = max(2, settings.seq2seq_beam // 2) if len(sections) > 1 else settings.seq2seq_beam
        print(f"[Seq2Seq] Đang sinh tóm tắt {len(sections)} ý bằng Semantic Chunking (beam={beam})...", flush=True)

        raw_outputs: list[str] = []
        with torch.no_grad():
            for idx, sec in enumerate(sections):
                sec_title = title.strip() if idx == 0 else ""
                src = f"Tiêu đề: {sec_title} Nội dung: {sec.strip()}"
                out = self._s2s.generate(
                    self.model, self.tokenizer, src, beam_size=beam, max_new=76
                )
                raw_outputs.append(out["summary"])

        summary_text = finalize_three_points(
            raw_outputs, sections, title=title, full_article=article
        )
        print(f"[Seq2Seq] Xong trong {time.perf_counter() - t0:.2f}s.", flush=True)
        return Summary(text=summary_text, truncated=truncated, input_tokens=total_tokens)
