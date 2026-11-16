"""Giao diện chung cho mọi mô hình tóm tắt + Pipeline Semantic Chunking & Chống bịa (Anti-Hallucination)."""
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
    return " ".join(text.split())


def split_vietnamese_sentences(text: str) -> list[str]:
    """Tách câu tiếng Việt an toàn, không cắt nhầm sau từ viết tắt (PGS., TS., TP., v.v.) và bỏ chú thích ảnh."""
    raw = unicodedata.normalize("NFC", str(text)).strip()
    if not raw:
        return []

    # Loại bỏ dòng chú thích ảnh báo chí (vd: '- Ảnh: THÁI THỊNH', '(Ảnh: ...)')
    raw = re.sub(r"\s*[\-–—]\s*Ảnh\s*:\s*[^\n.!?]+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\(\s*Ảnh\s*:\s*[^)]+\)", "", raw, flags=re.IGNORECASE)

    # Bảo vệ dấu chấm sau từ viết tắt, chữ cái viết tắt tên và số hiệu (vd: III.3)
    protected = _ABBR_DOT_RE.sub(rf"\1{_DOT_PLACEHOLDER} ", raw)
    protected = _INITIAL_DOT_RE.sub(rf"\1{_DOT_PLACEHOLDER} ", protected)
    protected = re.sub(r"([IVXLCDM0-9]+)\.(\d+)", rf"\1{_DOT_PLACEHOLDER}\2", protected)

    # Tách theo dòng mới hoặc dấu kết câu (. ! ? …) khi theo sau là khoảng trắng + chữ hoa/dấu mở ngoặc/nháy
    parts = re.split(
        r"\n+|(?<=[.!?…])\s+(?=[A-ZĐÀ-Ỹ“\"'(\[\-])",
        protected,
    )

    sentences: list[str] = []
    for part in parts:
        restored = part.replace(_DOT_PLACEHOLDER, ".").strip()
        if not restored:
            continue
        # Gộp mảnh quá ngắn (< 6 từ) hoặc kết thúc bằng từ viết tắt vào câu liền kề
        if sentences and (
            len(restored.split()) < 6
            or re.search(rf"\b({_ABBREVIATIONS})\.$", sentences[-1])
        ):
            sentences[-1] = f"{sentences[-1]} {restored}"
        else:
            sentences.append(restored)

    return sentences if sentences else [raw]


def _extract_content_terms(text: str) -> list[str]:
    """Rút trích từ và cặp từ (bigram) mang nghĩa để tính vector ngữ nghĩa cho Semantic Chunking."""
    clean = _clean_surface(text).lower()
    words = [w for w in re.findall(r"[^\W\d_]+|\d+", clean, flags=re.UNICODE) if w not in _VN_STOPWORDS and len(w) > 1]
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


def resolve_short_names(title: str, article: str) -> dict[str, str]:
    """Tìm họ tên đầy đủ (vd: 'ông Đỗ Xuân Diện') ở đầu bài để thay thế cho tên gọi tắt ('ông Diện') ở các đoạn sau."""
    sample_text = _clean_surface(f"{title}. {article[:1500]}")
    # Bắt mẫu: (ông|bà|anh|chị|thầy|cô|TS|PGS) + Họ Tên (2-4 từ viết hoa)
    pattern = re.compile(
        r"\b(ông|bà|anh|chị|thầy|cô|bác_sĩ|tiến_sĩ|thượng_nghị_sĩ|tổng_thống|cựu_tổng_thống)\s+"
        r"([A-ZĐÀ-Ỹ][a-zđà-ỹ]+(?:\s+[A-ZĐÀ-Ỹ][a-zđà-ỹ]+){1,3})\b"
    )
    mapping: dict[str, str] = {}
    for match in pattern.finditer(sample_text):
        prefix = match.group(1)
        full_name = match.group(2)
        last_token = full_name.split()[-1]
        short_key = f"{prefix.lower()} {last_token}"
        if short_key not in mapping:
            mapping[short_key] = f"{prefix} {full_name}"
    return mapping


def apply_name_resolution(chunk_text: str, name_map: dict[str, str]) -> str:
    """Thay thế 'ông Diện' -> 'ông Đỗ Xuân Diện' (lần xuất hiện đầu trong đoạn) nếu đoạn chưa có họ tên đầy đủ."""
    if not name_map:
        return chunk_text
    has_underscore = "_" in chunk_text
    cleaned = chunk_text
    for short_key, full_phrase in name_map.items():
        full_name_only = " ".join(full_phrase.split()[1:])
        if full_name_only in cleaned.replace("_", " "):
            continue
        prefix, last_name = short_key.split(" ", 1)
        replacement = full_phrase.replace(" ", "_") if has_underscore and " " in full_name_only else full_phrase
        cleaned = re.sub(
            rf"\b({prefix})\s+{re.escape(last_name)}\b",
            replacement,
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
    return cleaned


def semantic_chunk_article(title: str, article: str, target_points: int = 3) -> list[str]:
    """Bước 1: Semantic Chunking — Tách bài báo thành các đoạn theo điểm chuyển ý (Semantic Breakpoints).
    Sử dụng cửa sổ trượt TF-IDF + Cosine Similarity giữa các câu liên tiếp để tìm chỗ chuyển ý."""
    sentences = split_vietnamese_sentences(article)
    n = len(sentences)
    if n <= target_points:
        return sentences

    # Tính IDF trên toàn bộ các câu trong bài
    sent_terms = [_extract_content_terms(s) for s in sentences]
    df: Counter[str] = Counter()
    for terms in sent_terms:
        for t in set(terms):
            df[t] += 1

    # Tạo vector ngữ cảnh cửa sổ trượt (câu i kết hợp câu liền kề)
    vectors: list[dict[str, float]] = []
    for i in range(n):
        window = sent_terms[max(0, i - 1) : min(n, i + 2)]
        flat = [t for sub in window for t in sub]
        tf = Counter(flat)
        vec = {t: count * (math.log((n + 1) / (df[t] + 1)) + 1.0) for t, count in tf.items()}
        vectors.append(vec)

    # Tính độ tương đồng Cosine tại từng ranh giới câu (i, i+1)
    # Điểm chuyển ý (breakpoint) là nơi độ tương đồng thấp VÀ cân đối độ dài đoạn
    word_counts = [len(s.split()) for s in sentences]
    total_words = max(1, sum(word_counts))
    ideal_chunk_words = total_words / target_points

    name_map = resolve_short_names(title, article)

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
                # Độ tương đồng giữa câu trước ranh giới và câu sau ranh giới (càng thấp càng là chỗ chuyển ý)
                sim = _cosine_similarity(vectors[split_pos - 1], vectors[split_pos])
                # Phạt nhẹ nếu độ dài đoạn lệch quá xa mức cân bằng
                length_penalty = abs(acc_words - ideal_chunk_words) / max(ideal_chunk_words, 20.0) * 0.35
                # Thưởng nếu câu tiếp theo bắt đầu đoạn văn mới trong bài gốc
                score = sim + length_penalty
                if acc_words >= 18 and score < best_score:
                    best_score = score
                    best_split = split_pos

        chunk_str = " ".join(sentences[start_idx:best_split])
        chunks.append(apply_name_resolution(chunk_str, name_map))
        start_idx = best_split

    last_chunk = " ".join(sentences[start_idx:])
    chunks.append(apply_name_resolution(last_chunk, name_map))
    return [c for c in chunks if c.strip()]


# Giữ tên cũ tương thích ngược
def split_article_sections(article: str, target_points: int = 3) -> list[str]:
    return semantic_chunk_article("", article, target_points=target_points)


def normalize_one_sentence(text: str) -> str:
    """Rút trích đúng 1 câu hoàn chỉnh cho mỗi ý, không cắt nhầm từ viết tắt."""
    cleaned = _clean_surface(text)
    if not cleaned:
        return ""
    sents = split_vietnamese_sentences(cleaned)
    first = sents[0].strip() if sents else cleaned
    # Bỏ dấu gạch đầu dòng hoặc dấu ngoặc kép mở đầu bị lẻ
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


def is_grounded_in_source(candidate: str, full_source: str, section_text: str) -> bool:
    """Bước 3: Kiểm tra chống bịa (Anti-Hallucination Guard).
    Loại bỏ câu tóm tắt nếu:
      1. Chứa con số không có trong bài gốc.
      2. Chứa tên riêng viết hoa (như 'Nguyễn Văn Dương', 'Lê Thị Thanh Mai') không có trong bài gốc.
      3. Có quá nhiều âm tiết lạ không nằm trong bài gốc (như 'tỉnh uệ') hoặc câu quá ngắn/cụt."""
    cand = _clean_surface(candidate)
    words = cand.split()
    if len(words) < 7 or len(words) > 65:
        return False

    # Không chấp nhận câu kết thúc cụt bằng từ viết tắt (vd: 'của PGS.', 'ĐH Quốc gia TP.')
    if re.search(rf"\b({_ABBREVIATIONS})\.$", cand):
        return False

    src_clean = _clean_surface(full_source)
    src_lower = src_clean.lower()
    src_tokens = set(re.findall(r"[^\W_]+", src_lower, flags=re.UNICODE))
    src_numbers = set(re.findall(r"\d+", src_clean))

    # 1. Kiểm tra mọi con số trong câu tóm tắt phải có trong bài gốc
    cand_numbers = set(re.findall(r"\d+", cand))
    if not cand_numbers.issubset(src_numbers):
        return False

    # 2. Kiểm tra mọi từ viết hoa (tên người, địa danh, tổ chức) từ vị trí thứ 2 trong câu phải có trong bài gốc
    cand_tokens_raw = re.findall(r"[^\W_]+", cand, flags=re.UNICODE)
    for idx, tok in enumerate(cand_tokens_raw):
        if idx > 0 and tok[0].isupper() and not tok.isdigit():
            if tok.lower() not in src_tokens:
                return False

    # 3. Kiểm tra tỷ lệ âm tiết có trong bài gốc (chặn từ rác bịa như 'uệ', 'gã khùng' hoặc lệch ngữ cảnh)
    cand_lower_tokens = [t.lower() for t in cand_tokens_raw if not t.isdigit()]
    if not cand_lower_tokens:
        return False
    in_source_count = sum(1 for t in cand_lower_tokens if t in src_tokens)
    if in_source_count / len(cand_lower_tokens) < 0.85:
        return False

    # 4. Câu tóm tắt của đoạn nào phải chứa ít nhất 35% từ vựng thực của chính đoạn đó
    sec_tokens = set(re.findall(r"[^\W_]+", _clean_surface(section_text).lower(), flags=re.UNICODE)) - _VN_STOPWORDS
    cand_content = [t for t in cand_lower_tokens if t not in _VN_STOPWORDS]
    if cand_content and sec_tokens:
        sec_overlap = sum(1 for t in cand_content if t in sec_tokens) / len(cand_content)
        if sec_overlap < 0.35:
            return False

    return True


def _select_best_extractive_sentence(
    section_text: str, title: str, full_source: str, existing_points: list[str]
) -> str:
    """Khi câu sinh trừu tượng của một đoạn bị phát hiện bịa thông tin, chọn câu trung tâm (salient sentence)
    trực tiếp từ chính đoạn văn đó để đảm bảo 100% chính xác sự thật."""
    sents = split_vietnamese_sentences(_clean_surface(section_text))
    if not sents:
        return ""

    sec_vec = Counter(_extract_content_terms(f"{title} {section_text}"))
    existing_vecs = [Counter(_extract_content_terms(p)) for p in existing_points]

    best_sent = ""
    best_score = -1.0
    for s in sents:
        norm_s = normalize_one_sentence(s)
        w_count = len(norm_s.split())
        if w_count < 7:
            continue
        s_vec = Counter(_extract_content_terms(norm_s))
        # Kiểm tra không trùng ý với các câu đã chọn trước đó
        if any(_cosine_similarity(s_vec, ev) > 0.55 for ev in existing_vecs):
            continue
        centrality = _cosine_similarity(s_vec, sec_vec)
        # Ưu tiên câu có độ dài chuẩn báo chí (15 - 40 từ) và không phải câu thoại rời rạc
        len_bonus = 1.0 if 12 <= w_count <= 42 else 0.75
        quote_penalty = 0.85 if s.strip().startswith(("“", '"', "-")) else 1.0
        score = centrality * len_bonus * quote_penalty
        if score > best_score:
            best_score = score
            best_sent = norm_s

    if not best_sent:
        best_sent = normalize_one_sentence(sents[0])
    return best_sent


def finalize_three_points(
    raw_outputs: list[str],
    sections: list[str],
    title: str = "",
    full_article: str = "",
) -> str:
    """Bước 3 & 4: Kiểm tra chống bịa + Khử trùng lặp ngữ nghĩa (Semantic Deduplication) + Ghép thành 2-3 ý."""
    full_source = f"{title} {full_article or ' '.join(sections)}"
    points: list[str] = []
    point_vecs: list[dict[str, float]] = []

    for out, sec in zip(raw_outputs, sections):
        # Thử từng câu ứng viên mà model sinh ra cho đoạn này
        candidates = [normalize_one_sentence(s) for s in split_vietnamese_sentences(out) if s.strip()]
        chosen = ""
        for cand in candidates:
            if not is_grounded_in_source(cand, full_source, sec):
                continue
            cand_vec = Counter(_extract_content_terms(cand))
            # Kiểm tra trùng lặp ý với các ý đã chọn trước đó (Cosine > 0.50 là trùng ý)
            if any(_cosine_similarity(cand_vec, pv) > 0.50 for pv in point_vecs):
                continue
            chosen = cand
            break

        # Nếu model bịa tên/số hoặc lặp ý ở đoạn này -> lấy câu trọng tâm nhất của đoạn đó
        if not chosen:
            chosen = _select_best_extractive_sentence(sec, title, full_source, points)

        if chosen:
            c_vec = Counter(_extract_content_terms(chosen))
            if not any(_cosine_similarity(c_vec, pv) > 0.55 for pv in point_vecs):
                points.append(chosen)
                point_vecs.append(c_vec)

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
