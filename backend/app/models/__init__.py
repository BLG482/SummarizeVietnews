from ..config import settings
from .base import Summarizer


def build_registry() -> dict[str, Summarizer]:
    """Hai mô hình cố định: khóa 'seq2seq' và 'vit5'. USE_MOCK_MODELS=1 thay bằng mô hình giả."""
    if settings.use_mock:
        from .mock import MockSummarizer
        return {
            "seq2seq": MockSummarizer("seq2seq", "Seq2Seq (giả)", sentences=1, word_limit=500, delay=0.05),
            "vit5": MockSummarizer("vit5", "ViT5 (giả)", sentences=2, word_limit=700, delay=0.12),
        }
    from .seq2seq_adapter import Seq2SeqSummarizer
    from .vit5_adapter import ViT5Summarizer
    return {"seq2seq": Seq2SeqSummarizer(), "vit5": ViT5Summarizer()}


registry = build_registry()
