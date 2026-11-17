"""Giao diện chung cho mọi mô hình tóm tắt + Semantic Chunking thuần túy (100% Abstractive từ mô hình)."""
from collections import Counter
from dataclasses import dataclass
import math
import re
import threading
import time
import unicodedata


@dataclass
class Summary:
    text: str
    truncated: bool
    input_tokens: int | None = None
    latency_ms: float = 0.0


# Danh sách từ viết tắt phổ biến trong báo chí tiếng Việt (không tách câu sau dấu chấm của các từ này)
_ABBREVIATIONS = (
    r"PGS|GS|TS|ThS|BS|KS|CN|KTS|NSƯT|NSND|TP|Tp|TT|TX|Q|P|H|X|"
    r"UBND|HĐND|TAND|VKSND|MTTQ|TW|TƯ|ĐH|CĐ|THPT|THCS|GD|ĐT|LĐ|TB|XH|"
    r"BCH|BTV|TNHH|CP|NXB|BTC|CLB|HLV|VĐV|ĐBQH|TNS|TTXVN|PV|CTV|"
    r"Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Inc|Ltd|Co|Corp|vs|No"
)
_ABBR_DOT_RE = re.compile(rf"\b({_ABBREVIATIONS})\.\s+", flags=re.UNICODE)
_INITIAL_DOT_RE = re.compile(r"\b([A-ZĐÀ-Ỹ])\.\s+(?=[A-ZĐÀ-Ỹ])", flags=re.UNICODE)
_DOT_PLACEHOLDER = "∯"

_VN_STOPWORDS = {
    "và", "của", "các", "những", "trong", "được", "cho", "với", "là", "có",
    "đã", "đang", "sẽ", "này", "đó", "khi", "để", "từ", "về", "theo", "tại",
    "một", "như", "nhưng", "hay", "hoặc", "trên", "dưới", "sau", "trước",
    "cũng", "vào", "ra", "lên", "xuống", "đến", "tới", "bị", "do", "vì",
    "nếu", "thì", "mà", "còn", "rằng", "việc", "người", "ông", "bà", "anh",
    "chị", "em", "họ", "chúng", "tôi", "ta", "mình", "nơi", "lúc", "ngày",
    "tháng", "năm", "hôm", "nay", "qua", "tới", "đây", "đấy", "ấy",
}


def _clean_surface(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).replace("_", " ").strip()
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(
        r"(?<=[A-Za-zÀ-ỹ0-9])\s*-\s*(?=[A-Za-zÀ-ỹ0-9])",
        "-",
        text,
    )
    return " ".join(text.split())


def _ensure_period(sent: str) -> str:
    s = sent.strip()
    if not s:
        return ""
    if s[-1] not in ".!?…":
        s += "."
    return s


def split_vietnamese_sentences(text: str) -> list[str]:
    """Tách câu tiếng Việt an toàn, không cắt nhầm sau từ viết tắt (PGS., TS., TP., v.v.)."""
    raw = unicodedata.normalize("NFC", str(text)).strip()
    if not raw:
        return []

    raw = re.sub(r"\s*[\-–—]\s*Ảnh\s*:\s*[^\n.!?]+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\(\s*Ảnh\s*:\s*[^)]+\)", "", raw, flags=re.IGNORECASE)

    protected = _ABBR_DOT_RE.sub(rf"\1{_DOT_PLACEHOLDER} ", raw)
    protected = _INITIAL_DOT_RE.sub(rf"\1{_DOT_PLACEHOLDER} ", protected)
    protected = re.sub(r"([IVXLCDM0-9]+)\.(\d+)", rf"\1{_DOT_PLACEHOLDER}\2", protected)

    paragraphs = [p.strip() for p in re.split(r"\n+", protected) if p.strip()]
    parts: list[str] = []
    for para in paragraphs:
        sub_sents = re.split(r"(?<=[.!?…])\s+(?=[A-ZĐÀ-Ỹ“\"'(\[\-])", para)
        for s in sub_sents:
            s_clean = s.strip()
            if s_clean:
                parts.append(_ensure_period(s_clean))

    sentences: list[str] = []
    for part in parts:
        restored = part.replace(_DOT_PLACEHOLDER, ".").strip()
        if not restored:
            continue
        if sentences and (
            len(restored.split()) < 6
            or re.search(rf"\b({_ABBREVIATIONS})\.$", sentences[-1])
        ):
            sentences[-1] = f"{sentences[-1].rstrip('.')} {restored}"
        else:
            sentences.append(_ensure_period(restored))

    return sentences if sentences else [_ensure_period(raw)]


def _extract_content_terms(text: str, include_bigrams: bool = True) -> list[str]:
    clean = _clean_surface(text).lower()
    words = [
        w
        for w in re.findall(r"[^\W\d_]+|\d+", clean, flags=re.UNICODE)
        if w not in _VN_STOPWORDS and len(w) > 1
    ]
    if not include_bigrams:
        return words
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    num = sum(vec_a[k] * vec_b[k] for k in common)
    den_a = math.sqrt(sum(v * v for v in vec_a.values()))
    den_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return num / (den_a * den_b)


def semantic_chunk_article(title: str, article: str, target_points: int = 3) -> list[str]:
    """Chia văn bản gốc thành các đoạn đầy đủ theo điểm chuyển ý (Semantic Breakpoints).
    Giữ nguyên toàn bộ câu trong mỗi đoạn để mô hình tự tóm tắt trừu tượng (Abstractive),
    KHÔNG lọc bớt câu nguồn và KHÔNG trích xuất nguyên văn."""
    sentences = split_vietnamese_sentences(article)
    n = len(sentences)
    if n <= target_points:
        return sentences

    sent_terms = [_extract_content_terms(s) for s in sentences]
    df: Counter[str] = Counter()
    for terms in sent_terms:
        for t in set(terms):
            df[t] += 1

    vectors: list[dict[str, float]] = []
    for i in range(n):
        window = sent_terms[max(0, i - 1) : min(n, i + 2)]
        flat = [t for sub in window for t in sub]
        tf = Counter(flat)
        vec = {t: count * (math.log((n + 1) / (df[t] + 1)) + 1.0) for t, count in tf.items()}
        vectors.append(vec)

    word_counts = [len(s.split()) for s in sentences]
    total_words = max(1, sum(word_counts))
    ideal_chunk_words = total_words / target_points

    chunks: list[str] = []
    start_idx = 0
    for chunk_idx in range(target_points - 1):
        remaining_chunks = target_points - chunk_idx
        min_end = start_idx + 1
        max_end = n - (remaining_chunks - 1)
        if min_end >= max_end:
            best_split = min_end
        else:
            best_split = min_end
            best_score = float("inf")
            acc_words = 0
            for split_pos in range(start_idx + 1, max_end + 1):
                acc_words += word_counts[split_pos - 1]
                sim = _cosine_similarity(vectors[split_pos - 1], vectors[split_pos])
                length_penalty = abs(acc_words - ideal_chunk_words) / max(ideal_chunk_words, 20.0) * 0.35
                score = sim + length_penalty
                if acc_words >= 20 and score < best_score:
                    best_score = score
                    best_split = split_pos

        chunks.append(" ".join(sentences[start_idx:best_split]))
        start_idx = best_split

    chunks.append(" ".join(sentences[start_idx:]))
    return [c for c in chunks if c.strip()]


def split_article_sections(article: str, target_points: int = 3) -> list[str]:
    return semantic_chunk_article("", article, target_points=target_points)


def normalize_one_sentence(text: str) -> str:
    """Rút trích đúng 1 câu do mô hình sinh ra, chuẩn hóa chữ hoa đầu câu và dấu chấm cuối câu."""
    cleaned = _clean_surface(text)
    if not cleaned:
        return ""
    sents = split_vietnamese_sentences(cleaned)
    first = sents[0].strip() if sents else cleaned
    first = re.sub(r"^[\-–—•*+\s]+", "", first).strip()
    if first.count("“") > first.count("”"):
        first = first.replace("“", "").strip()
    if first.count('"') % 2 == 1:
        first = first.replace('"', "").strip()
    if not first:
        return ""
    first = first[0].upper() + first[1:] if len(first) > 1 else first.upper()
    if first[-1] not in ".!?…":
        first += "."
    return first


def _is_redundant(cand_text: str, existing_points: list[str]) -> bool:
    """Kiểm tra xem câu mới do mô hình sinh ra có bị lặp ý với câu trước đó của chính mô hình đó không."""
    cand_terms = set(_extract_content_terms(cand_text, include_bigrams=False))
    cand_vec = Counter(_extract_content_terms(cand_text, include_bigrams=True))
    if not cand_terms:
        return True
    for prev in existing_points:
        prev_terms = set(_extract_content_terms(prev, include_bigrams=False))
        prev_vec = Counter(_extract_content_terms(prev, include_bigrams=True))
        if prev_terms:
            overlap = len(cand_terms & prev_terms) / min(len(cand_terms), len(prev_terms))
            if overlap > 0.65:
                return True
        if _cosine_similarity(cand_vec, prev_vec) > 0.65:
            return True
    return False


def finalize_three_points(
    raw_outputs: list[str],
    sections: list[str],
    title: str = "",
    full_article: str = "",
) -> str:
    """Chỉ giữ kết quả tóm tắt trừu tượng (Abstractive) thật sự do chính mô hình sinh ra,
    lọc các câu bị trùng lặp giữa các đoạn và ghép lại."""
    points: list[str] = []
    for out in raw_outputs:
        sents = [normalize_one_sentence(s) for s in split_vietnamese_sentences(out) if s.strip()]
        for cand in sents:
            if not _is_redundant(cand, points):
                points.append(cand)
                break
    # Nếu tất cả các đoạn sinh ra câu trùng nhau thì vẫn giữ ít nhất câu đầu tiên của mô hình
    if not points and raw_outputs:
        first = normalize_one_sentence(raw_outputs[0])
        if first:
            points.append(first)
    return "\n".join(points[:3])


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
