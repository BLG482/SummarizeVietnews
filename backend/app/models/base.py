"""Giao diện chung cho mọi mô hình tóm tắt + Pipeline Semantic Chunking, Lọc câu trọng tâm & Chống bịa."""
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
    # Chuẩn hóa dấu gạch nối bị tách rời trong mã vũ khí/từ ghép/ngày tháng: 'F - 16' -> 'F-16', 'AIM - 9' -> 'AIM-9', 'GD - ĐT' -> 'GD-ĐT'
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
    """Tách câu tiếng Việt an toàn, không cắt nhầm sau từ viết tắt (PGS., TS., TP., v.v.) và bỏ chú thích ảnh."""
    raw = unicodedata.normalize("NFC", str(text)).strip()
    if not raw:
        return []

    # Loại bỏ dòng chú thích ảnh báo chí (vd: '- Ảnh: THÁI THỊNH', '(Ảnh: ...)')
    raw = re.sub(r"\s*[\-–—]\s*Ảnh\s*:\s*[^\n.!?]+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\(\s*Ảnh\s*:\s*[^)]+\)", "", raw, flags=re.IGNORECASE)

    # Bảo vệ dấu chấm sau từ viết tắt, chữ cái viết tắt tên và số hiệu (vd: III.3, 2.5)
    protected = _ABBR_DOT_RE.sub(rf"\1{_DOT_PLACEHOLDER} ", raw)
    protected = _INITIAL_DOT_RE.sub(rf"\1{_DOT_PLACEHOLDER} ", protected)
    protected = re.sub(r"([IVXLCDM0-9]+)\.(\d+)", rf"\1{_DOT_PLACEHOLDER}\2", protected)

    # Tách theo dòng mới trước để đảm bảo các đoạn văn không có dấu chấm cuối câu không bị dính vào câu sau
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
        # Gộp mảnh quá ngắn (< 6 từ) hoặc kết thúc bằng từ viết tắt vào câu liền kề
        if sentences and (
            len(restored.split()) < 6
            or re.search(rf"\b({_ABBREVIATIONS})\.$", sentences[-1])
        ):
            sentences[-1] = f"{sentences[-1].rstrip('.')} {restored}"
        else:
            sentences.append(_ensure_period(restored))

    return sentences if sentences else [_ensure_period(raw)]


def _extract_content_terms(text: str, include_bigrams: bool = True) -> list[str]:
    """Rút trích từ mang nghĩa để tính vector ngữ nghĩa cho Semantic Chunking & Salience Ranking."""
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


def resolve_short_names(title: str, article: str) -> dict[str, str]:
    """Tìm họ tên đầy đủ (vd: 'ông Đỗ Xuân Diện') ở đầu bài để thay thế cho tên gọi tắt ('ông Diện') ở các đoạn sau."""
    sample_text = _clean_surface(f"{title}. {article[:1500]}")
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
        replacement = (
            full_phrase.replace(" ", "_")
            if has_underscore and " " in full_name_only
            else full_phrase
        )
        cleaned = re.sub(
            rf"\b({prefix})\s+{re.escape(last_name)}\b",
            replacement,
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
    return cleaned


def _select_top_salient_sentences_in_chunk(
    chunk_sents: list[str],
    title: str,
    full_article_vec: dict[str, float],
    max_sents: int = 2,
) -> str:
    """Trong mỗi Semantic Chunk, chọn ra 1-2 câu trọng tâm nhất (Salience Filtering) trước khi đưa vào model.
    Việc này giúp:
      1. Model không ghép nhầm 2 vế của 2 câu xa nhau ('mạnh trên giấy tờ' + 'đối phó UAV').
      2. Ở đoạn cuối, không bị câu phụ đầu đoạn (như 'kiểm tra sau 300 giờ bay') che khuất ý lớn ('cạn tên lửa AIM-9, máy bay thay thế hạn chế')."""
    if len(chunk_sents) <= max_sents:
        return " ".join(chunk_sents)

    title_vec = Counter(_extract_content_terms(title, include_bigrams=False))
    chunk_vec = Counter(_extract_content_terms(" ".join(chunk_sents), include_bigrams=False))

    scored: list[tuple[float, int, str]] = []
    for idx, s in enumerate(chunk_sents):
        s_clean = _clean_surface(s)
        w_count = len(s_clean.split())
        if w_count < 7:
            continue
        s_vec = Counter(_extract_content_terms(s_clean, include_bigrams=False))
        sim_chunk = _cosine_similarity(s_vec, chunk_vec)
        sim_title = _cosine_similarity(s_vec, title_vec)
        sim_doc = _cosine_similarity(s_vec, full_article_vec)

        # Phạt các câu chỉ mang tính giải thích định nghĩa chung ('thường phải trải qua quy trình...')
        is_background_def = 0.72 if re.search(r"\bthường phải trải qua\b|\btheo quy định\b", s_clean.lower()) else 1.0
        # Ưu tiên câu nêu vấn đề cốt lõi, nguyên nhân, hệ quả hoặc số liệu tổng hợp
        has_key_signal = 1.18 if re.search(
            r"\b(tuy nhiên|ngoài|đối mặt|thiếu|hạn chế|đình trệ|cảnh báo|nguyên nhân|khiến|do đó|cho biết)\b",
            s_clean.lower(),
        ) else 1.0
        len_factor = 1.0 if 14 <= w_count <= 45 else 0.82

        score = (0.50 * sim_chunk + 0.30 * sim_doc + 0.20 * sim_title) * is_background_def * has_key_signal * len_factor
        scored.append((score, idx, s))

    if not scored:
        return " ".join(chunk_sents[:max_sents])

    scored.sort(key=lambda x: x[0], reverse=True)
    # Lấy câu điểm cao nhất; nếu câu đó ngắn (< 28 từ) và có câu thứ 2 liên quan chặt thì lấy tối đa 2 câu theo đúng thứ tự xuất hiện
    top_selected = [scored[0]]
    if len(scored) > 1 and len(_clean_surface(scored[0][2]).split()) < 26:
        top_selected.append(scored[1])
    top_selected.sort(key=lambda x: x[1])
    return " ".join(x[2] for x in top_selected)


def semantic_chunk_article(title: str, article: str, target_points: int = 3) -> list[str]:
    """Bước 1: Semantic Chunking + Intra-Chunk Salience Selection.
    Chia bài báo theo điểm chuyển ý (Semantic Breakpoints) rồi lọc câu trọng tâm của mỗi đoạn."""
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
    name_map = resolve_short_names(title, article)
    full_doc_vec = Counter(_extract_content_terms(f"{title} {article}", include_bigrams=False))

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
                if acc_words >= 18 and score < best_score:
                    best_score = score
                    best_split = split_pos

        chunk_sents = sentences[start_idx:best_split]
        focused_chunk = _select_top_salient_sentences_in_chunk(
            chunk_sents, title, full_doc_vec, max_sents=2
        )
        chunks.append(apply_name_resolution(focused_chunk, name_map))
        start_idx = best_split

    last_sents = sentences[start_idx:]
    focused_last = _select_top_salient_sentences_in_chunk(
        last_sents, title, full_doc_vec, max_sents=2
    )
    chunks.append(apply_name_resolution(focused_last, name_map))
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
    first = re.sub(r"^[\-–—•*+\s]+", "", first).strip()
    # Bỏ từ nối đứng lửng ở đầu câu nếu đứng một mình
    first = re.sub(r"^(Tuy nhiên|Do đó|Vì vậy|Ngoài ra|Theo đó),\s*", "", first, flags=re.IGNORECASE).strip()
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
    """Bước 3: Kiểm tra chống bịa & chống cắt ghép câu cơ học (Anti-Hallucination & Clause-Stitching Guard).
    Loại bỏ câu tóm tắt nếu:
      1. Chứa con số không có trong bài gốc.
      2. Chứa bất kỳ âm tiết/từ nào bị mất chữ ('tích lũ') hoặc không tồn tại trong bài gốc.
      3. Ghép nhầm 2 vế câu không liên quan ('Zelensky lên châu Âu để sửa chữa bảo dưỡng', 'mạnh trên giấy tờ để đối phó UAV')."""
    cand = _clean_surface(candidate)
    words = cand.split()
    if len(words) < 8 or len(words) > 58:
        return False

    if re.search(rf"\b({_ABBREVIATIONS})\.$", cand):
        return False

    src_clean = _clean_surface(full_source)
    src_lower = src_clean.lower()
    src_tokens = set(re.findall(r"[^\W_]+", src_lower, flags=re.UNICODE))
    src_numbers = set(re.findall(r"\d+", src_clean))

    # 1. Mọi con số phải có trong bài gốc
    cand_numbers = set(re.findall(r"\d+", cand))
    if not cand_numbers.issubset(src_numbers):
        return False

    # 2. Kiểm tra từ vựng: KHÔNG cho phép từ bị cụt âm tiết (như 'lũ' thay vì 'lũy') hoặc tên riêng bịa
    cand_tokens_raw = re.findall(r"[^\W_]+", cand, flags=re.UNICODE)
    allowed_connectors = {"cho", "biết", "theo", "tại", "vừa", "đã", "đang", "sẽ", "việc", "nhằm", "trong", "với"}
    unknown_tokens = [
        t for t in cand_tokens_raw
        if not t.isdigit() and t.lower() not in src_tokens and t.lower() not in allowed_connectors
    ]
    if unknown_tokens:
        return False

    # 3. Kiểm tra tính liên tục cụm từ (Bigram Continuity) để chặn lỗi 'Râu ông nọ cắm cằm bà kia'
    # Nếu model tự ý nối 2 vế xa nhau (như 'Zelensky lên châu Âu', 'trên giấy tờ để đối phó'), tỷ lệ bigram có thật trong bài gốc sẽ tụt xuống thấp.
    src_words_seq = [t.lower() for t in re.findall(r"[^\W_]+", src_lower, flags=re.UNICODE)]
    src_bigrams = {f"{src_words_seq[i]} {src_words_seq[i+1]}" for i in range(len(src_words_seq) - 1)}
    cand_words_seq = [t.lower() for t in cand_tokens_raw]
    if len(cand_words_seq) >= 6:
        cand_bigrams = [f"{cand_words_seq[i]} {cand_words_seq[i+1]}" for i in range(len(cand_words_seq) - 1)]
        matched_bigrams = sum(1 for bg in cand_bigrams if bg in src_bigrams)
        if matched_bigrams / len(cand_bigrams) < 0.72:
            return False

    # 4. Câu tóm tắt của đoạn nào phải chứa ít nhất 45% từ khóa thực của chính đoạn đó
    sec_tokens = set(re.findall(r"[^\W_]+", _clean_surface(section_text).lower(), flags=re.UNICODE)) - _VN_STOPWORDS
    cand_content = [t.lower() for t in cand_tokens_raw if not t.isdigit() and t.lower() not in _VN_STOPWORDS]
    if cand_content and sec_tokens:
        sec_overlap = sum(1 for t in cand_content if t in sec_tokens) / len(cand_content)
        if sec_overlap < 0.45:
            return False

    return True


def _is_redundant(cand_text: str, existing_points: list[str]) -> bool:
    """Kiểm tra trùng lặp ý giữa câu mới và các ý đã chọn (dùng cả Unigram Jaccard và Cosine)."""
    cand_terms = set(_extract_content_terms(cand_text, include_bigrams=False))
    cand_vec = Counter(_extract_content_terms(cand_text, include_bigrams=True))
    if not cand_terms:
        return True
    for prev in existing_points:
        prev_terms = set(_extract_content_terms(prev, include_bigrams=False))
        prev_vec = Counter(_extract_content_terms(prev, include_bigrams=True))
        if prev_terms:
            overlap = len(cand_terms & prev_terms) / min(len(cand_terms), len(prev_terms))
            if overlap > 0.42:
                return True
        if _cosine_similarity(cand_vec, prev_vec) > 0.42:
            return True
    return False


def _select_best_extractive_sentence(
    section_text: str, title: str, full_source: str, existing_points: list[str]
) -> str:
    """Khi câu sinh trừu tượng của một đoạn bị phát hiện cắt ghép lỗi hoặc bịa thông tin,
    chọn câu trung tâm chuẩn xác nhất từ đoạn đó."""
    sents = split_vietnamese_sentences(_clean_surface(section_text))
    if not sents:
        return ""

    sec_vec = Counter(_extract_content_terms(f"{title} {section_text}", include_bigrams=False))

    best_sent = ""
    best_score = -1.0
    for s in sents:
        norm_s = normalize_one_sentence(s)
        w_count = len(norm_s.split())
        if w_count < 7:
            continue
        if _is_redundant(norm_s, existing_points):
            continue
        s_vec = Counter(_extract_content_terms(norm_s, include_bigrams=False))
        centrality = _cosine_similarity(s_vec, sec_vec)
        len_bonus = 1.0 if 12 <= w_count <= 45 else 0.78
        quote_penalty = 0.85 if s.strip().startswith(("“", '"', "-")) else 1.0
        score = centrality * len_bonus * quote_penalty
        if score > best_score:
            best_score = score
            best_sent = norm_s

    if not best_sent:
        for s in sents:
            norm_s = normalize_one_sentence(s)
            if not _is_redundant(norm_s, existing_points):
                return norm_s
        best_sent = normalize_one_sentence(sents[0])
    return best_sent


def finalize_three_points(
    raw_outputs: list[str],
    sections: list[str],
    title: str = "",
    full_article: str = "",
) -> str:
    """Bước 3 & 4: Kiểm tra chống bịa/cắt ghép + Khử trùng lặp ngữ nghĩa + Hợp nhất 2-3 ý."""
    full_source = f"{title}. {full_article or ' '.join(sections)}"
    all_article_sents = split_vietnamese_sentences(_clean_surface(full_article)) if full_article else []
    points: list[str] = []

    for out, sec in zip(raw_outputs, sections):
        candidates = [normalize_one_sentence(s) for s in split_vietnamese_sentences(out) if s.strip()]
        chosen = ""
        for cand in candidates:
            if not is_grounded_in_source(cand, full_source, sec):
                continue
            if _is_redundant(cand, points):
                continue
            chosen = cand
            break

        if not chosen:
            chosen = _select_best_extractive_sentence(sec, title, full_source, points)

        if chosen and not _is_redundant(chosen, points):
            points.append(chosen)

    # Nếu sau khi lọc trùng mà còn dưới 3 ý và bài báo đủ dài, bổ sung câu trọng tâm chưa xuất hiện từ toàn bài
    if len(points) < 3 and len(all_article_sents) >= 3:
        extra = _select_best_extractive_sentence(full_article, title, full_source, points)
        if extra and not _is_redundant(extra, points):
            points.append(extra)

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
