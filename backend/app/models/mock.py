"""Mô hình giả để chạy thử giao diện và kiểm thử khi chưa có checkpoint (USE_MOCK_MODELS=1)."""

import time

from .base import Summarizer, Summary, finalize_three_points, split_article_sections


class MockSummarizer(Summarizer):
    def __init__(
        self, key: str, label: str, sentences: int, word_limit: int, delay: float
    ):
        super().__init__()
        self.key, self.label = key, label
        self.description = (
            "Mô hình giả: lấy 3 ý từ 3 phần của bài. Chỉ dùng để thử giao diện."
        )
        self.input_limit = word_limit
        self._sentences, self._delay = sentences, delay
        self.last_input: tuple[str, str, bool] | None = None

    def _load(self) -> None:
        self.device = "mock"

    def _summarize(self, title: str, article: str, *, raw: bool) -> Summary:
        self.last_input = (title, article, raw)
        time.sleep(self._delay)
        sections = split_article_sections(article, target_points=3)
        text = finalize_three_points(sections, sections)
        return Summary(
            text=text,
            truncated=len(article.split()) > self.input_limit,
            input_tokens=len(article.split()),
        )
