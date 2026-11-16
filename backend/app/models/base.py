"""Giao diện chung cho mọi mô hình tóm tắt. Muốn thêm mô hình mới, kế thừa Summarizer."""
import re
import threading
import time
from dataclasses import dataclass


@dataclass
class Summary:
    text: str
    truncated: bool
    input_tokens: int | None = None
    latency_ms: float = 0.0


def split_article_sections(article: str, target_points: int = 3) -> list[str]:
    """Chia văn bản nguồn thành 3 phần theo mạch bài báo (Mở đầu - Diễn biến - Kết quả)
    để mỗi phần sinh ra đúng 1 ý (1 câu tóm tắt)."""
    raw_lines = [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", article.strip()) if p.strip()]
    sentences: list[str] = []
    for chunk in raw_lines:
        if sentences and len(chunk.split()) < 6:
            sentences[-1] = f"{sentences[-1]} {chunk}"
        else:
            sentences.append(chunk)

    if not sentences:
        return [article.strip()]
    if len(sentences) <= target_points:
        return sentences

    total_words = sum(len(s.split()) for s in sentences)
    target_per_section = max(1, total_words / target_points)

    sections: list[str] = []
    current: list[str] = []
    current_words = 0

    for idx, sent in enumerate(sentences):
        current.append(sent)
        current_words += len(sent.split())
        remaining_sents = len(sentences) - (idx + 1)
        remaining_sections = target_points - (len(sections) + 1)
        if remaining_sections > 0 and (
            current_words >= target_per_section or remaining_sents == remaining_sections
        ):
            sections.append(" ".join(current))
            current = []
            current_words = 0

    if current:
        if len(sections) < target_points:
            sections.append(" ".join(current))
        else:
            sections[-1] = f"{sections[-1]} {' '.join(current)}"

    return sections[:target_points]


def normalize_one_sentence(text: str) -> str:
    """Rút trích đúng 1 câu hoàn chỉnh cho mỗi ý, chuẩn hóa chữ hoa đầu câu và dấu chấm kết câu."""
    cleaned = text.replace("_", " ").strip()
    cleaned = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", cleaned)
    cleaned = re.sub(r"([\(\[\{])\s+", r"\1", cleaned)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", cleaned) if p.strip()]
    first = parts[0] if parts else cleaned
    # Nếu câu đầu quá ngắn (< 6 từ) và còn câu thứ 2 thì ghép lại thành 1 câu bằng dấu phẩy/chấm phẩy
    if len(first.split()) < 6 and len(parts) > 1:
        first = first.rstrip(".!?…") + ", " + parts[1]
    first = first[0].upper() + first[1:] if len(first) > 1 else first.upper()
    if first[-1] not in ".!?…":
        first += "."
    return first


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower()))


def finalize_three_points(raw_outputs: list[str], sections: list[str]) -> str:
    """Đảm bảo đầu ra gồm 2-3 ý không trùng lặp, mỗi ý đúng 1 câu."""
    points: list[str] = []
    for out, sec in zip(raw_outputs, sections):
        candidate = normalize_one_sentence(out)
        # Kiểm tra nếu câu sinh ra bị trùng lặp quá cao (> 60% từ) với ý trước đó
        cand_words = _word_set(candidate)
        is_dup = False
        for prev in points:
            prev_words = _word_set(prev)
            if cand_words and prev_words:
                overlap = len(cand_words & prev_words) / min(len(cand_words), len(prev_words))
                if overlap > 0.60:
                    is_dup = True
                    break
        if is_dup or len(candidate.split()) < 4:
            # Nếu trùng, thử lấy câu thứ 2 trong output hoặc câu đầu tiên của đoạn nguồn
            extra = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", out.replace("_", " ").strip()) if p.strip()]
            if len(extra) > 1:
                candidate = normalize_one_sentence(extra[1])
            else:
                sec_sents = [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", sec.strip()) if len(p.split()) >= 6]
                if sec_sents:
                    candidate = normalize_one_sentence(sec_sents[0])
        if candidate and candidate not in points:
            points.append(candidate)
    return "\n".join(points)


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
