"""Adapter ViT5 (Transformer encoder-decoder) nạp từ thư mục checkpoint của Hugging Face."""
import re

from ..config import settings
from .base import Summarizer, Summary, finalize_three_points, semantic_chunk_article


def word_segment(text: str) -> str:
    """ViT5 được fine-tune trên văn bản đã tách từ (có '_'). Với văn bản thường, tách từ trước
    khi đưa vào. Bỏ qua nếu văn bản đã có '_' (coi như đã tách) hoặc chưa cấu hình bộ tách."""
    mode = settings.vit5_segmenter
    if mode in ("", "none") or "_" in text:
        return text
    if mode == "pyvi":
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    if mode == "underthesea":
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    raise ValueError(f"VIT5_WORD_SEGMENTER không hợp lệ: {mode!r} (none | pyvi | underthesea)")


def clean_output(text: str) -> str:
    text = text.replace("_", " ")
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return " ".join(text.split())


class ViT5Summarizer(Summarizer):
    key = "vit5"
    label = "ViT5"
    description = "ViT5-base (Transformer encoder-decoder), fine-tune trên VietNews."

    def __init__(self) -> None:
        super().__init__()
        self.input_limit = settings.vit5_max_input

    def _load(self) -> None:
        if not settings.vit5_dir:
            raise RuntimeError("Chưa đặt VIT5_CHECKPOINT_DIR (thư mục checkpoint ViT5).")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

        name = settings.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.device = name
        import gc
        import json
        import struct
        from pathlib import Path
        from transformers import AutoConfig
        from transformers.modeling_utils import no_init_weights

        vit5_dir = settings.vit5_dir
        if not Path(vit5_dir).exists() and "/" in vit5_dir:
            import tempfile
            from huggingface_hub import snapshot_download
            target_dir = Path(tempfile.gettempdir()) / "hf_vit5_model"
            print(f"[ViT5] Đang tải checkpoint từ Hugging Face Hub ({vit5_dir})...", flush=True)
            vit5_dir = snapshot_download(repo_id=vit5_dir, local_dir=str(target_dir))

        print(f"[ViT5] 1/3 Đang nạp tokenizer từ {vit5_dir}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            vit5_dir, use_fast=False, local_files_only=True
        )
        self.tokenizer.truncation_side = "right"

        safetensors_path = Path(vit5_dir) / "model.safetensors"
        target_dtype = torch.float32
        if safetensors_path.is_file():
            print("[ViT5] 2/3 Đang khởi tạo khung ViT5...", flush=True)
            cfg = AutoConfig.from_pretrained(vit5_dir, local_files_only=True)
            cfg._attn_implementation = "eager"
            cfg.use_cache = True
            with no_init_weights():
                model = AutoModelForSeq2SeqLM.from_config(cfg, torch_dtype=target_dtype)
            model.config.use_cache = True
            model.tie_weights()
            params = dict(model.named_parameters())
            params.update(dict(model.named_buffers()))
            print("[ViT5] 3/3 Đang đọc từng lớp từ model.safetensors (không dùng mmap)...", flush=True)
            dtype_map = {
                "F32": torch.float32,
                "F16": torch.float16,
                "BF16": torch.bfloat16,
                "I64": torch.int64,
                "I32": torch.int32,
            }
            with safetensors_path.open("rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len).decode("utf-8"))
                base_offset = 8 + header_len
                with torch.no_grad():
                    for k, meta in header.items():
                        if k == "__metadata__" or k not in params:
                            continue
                        begin, end = meta["data_offsets"]
                        f.seek(base_offset + begin)
                        raw = f.read(end - begin)
                        src_dtype = dtype_map.get(meta["dtype"], torch.float32)
                        t = torch.frombuffer(bytearray(raw), dtype=src_dtype).reshape(meta["shape"])
                        params[k].copy_(t)
                        del raw, t
            model.tie_weights()
            gc.collect()
            self.model = model.to(name).eval()
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.vit5_dir,
                local_files_only=True,
                torch_dtype=target_dtype,
                attn_implementation="eager",
            ).to(name).eval()
            self.model.config.use_cache = True
        print(f"[ViT5] Đã nạp xong (device={name}).", flush=True)
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.max_new_tokens = settings.vit5_max_new
        self.generation_config.num_beams = settings.vit5_beam
        self.generation_config.do_sample = False
        self.generation_config.length_penalty = 1.0
        self.generation_config.no_repeat_ngram_size = 3
        self.generation_config.early_stopping = True
        self.generation_config.use_cache = True

    def _summarize(self, title: str, article: str, *, raw: bool) -> Summary:
        import copy
        import time

        t0 = time.perf_counter()
        full_text = settings.vit5_input_template.format(title=title.strip(), article=article.strip())
        if not raw:
            full_text = word_segment(full_text)
        limit = settings.vit5_max_input
        full_length = len(self.tokenizer(full_text, truncation=False)["input_ids"])

        sections = semantic_chunk_article(title, article, target_points=3)
        beam = max(2, settings.vit5_beam // 2) if len(sections) > 1 else settings.vit5_beam
        print(f"[ViT5] Đang sinh tóm tắt {len(sections)} ý bằng Semantic Chunking (beam={beam})...", flush=True)

        inputs: list[str] = []
        for idx, sec in enumerate(sections):
            sec_title = title.strip() if idx == 0 else ""
            sec_text = settings.vit5_input_template.format(title=sec_title, article=sec.strip())
            if not raw:
                sec_text = word_segment(sec_text)
            inputs.append(sec_text)

        batch = self.tokenizer(
            inputs, max_length=limit, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        gen_cfg = copy.copy(self.generation_config)
        gen_cfg.num_beams = beam
        gen_cfg.max_new_tokens = min(96, settings.vit5_max_new)

        with self._torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                generation_config=gen_cfg,
                use_cache=True,
            )

        raw_outputs = [
            clean_output(self.tokenizer.decode(out, skip_special_tokens=True).strip())
            for out in outputs
        ]
        summary_text = finalize_three_points(
            raw_outputs, sections, title=title, full_article=article
        )
        print(f"[ViT5] Xong trong {time.perf_counter() - t0:.2f}s.", flush=True)
        return Summary(text=summary_text, truncated=full_length > limit, input_tokens=full_length)
