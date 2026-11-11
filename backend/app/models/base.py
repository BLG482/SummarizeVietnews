"""Giao diện chung cho mọi mô hình tóm tắt. Muốn thêm mô hình mới, kế thừa Summarizer."""
import threading
import time
from dataclasses import dataclass


@dataclass
class Summary:
    text: str
    truncated: bool
    input_tokens: int | None = None
    latency_ms: float = 0.0


class Summarizer:
    key = ""
    label = ""
    description = ""
    input_limit: int | None = None  # giới hạn token đầu vào, để hiển thị cho người dùng

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()  # mỗi mô hình chỉ sinh một bài mỗi lúc
        self.state = "not_loaded"  # not_loaded | loading | ready | error
        self.error: str | None = None
        self.device: str | None = None

    # --- các lớp con cài đặt hai hàm này ---
    def _load(self) -> None:
        raise NotImplementedError

    def _summarize(self, title: str, article: str, *, raw: bool) -> Summary:
        raise NotImplementedError

    # --- phần dùng chung ---
    def ensure_loaded(self) -> None:
        with self._load_lock:
            if self.state == "ready":
                return
            if self.state == "error":
                raise RuntimeError(self.error)
            self.state = "loading"
            try:
                self._load()
                self.state = "ready"
            except Exception as exc:  # lỗi nạp được ghi lại, không làm sập server
                self.state = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                raise

    def reload(self) -> None:
        with self._load_lock:
            self.state, self.error = "not_loaded", None

    def summarize(self, title: str, article: str, *, raw: bool = False) -> Summary:
        """raw=True: văn bản đã tách từ sẵn (bài mẫu từ CSV), không tách lại."""
        self.ensure_loaded()
        with self._run_lock:
            start = time.perf_counter()
            result = self._summarize(title, article, raw=raw)
            result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def info(self) -> dict:
        return {"key": self.key, "label": self.label, "description": self.description,
                "input_limit": self.input_limit, "state": self.state,
                "error": self.error, "device": self.device}
