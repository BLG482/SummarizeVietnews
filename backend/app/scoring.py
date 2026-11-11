"""Chấm ROUGE đúng quy tắc chung của bài: NFC, casefold, '_' thành khoảng trắng,
lấy chuỗi chữ-số Unicode bằng [^\\W_]+, không stemming, F1 × 100, ROUGE-L (không phải Lsum)."""
import re
import unicodedata

from rouge_score import rouge_scorer

_WORD = re.compile(r"[^\W_]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


class VietnameseTokenizer:
    def tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFC", text).casefold().replace("_", " ")
        return _WORD.findall(text)


_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], tokenizer=VietnameseTokenizer())


def rouge(candidate: str, reference: str) -> dict[str, float]:
    """F1 × 100 của một cặp (tóm tắt sinh, tham chiếu). Đầu ra rỗng nhận 0."""
    if not candidate.strip() or not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _scorer.score(reference, candidate)  # (target, prediction)
    return {k: round(v.fmeasure * 100, 4) for k, v in scores.items()}


def _digits(number: str) -> str:
    return re.sub(r"[.,]", "", number)


def unsupported_numbers(summary: str, source: str) -> list[str]:
    """Số xuất hiện trong tóm tắt nhưng không có trong bài gốc (chỉ là cảnh báo tự động,
    không phải kiểm chứng sự thật)."""
    known = {_digits(n) for n in _NUMBER.findall(source)}
    found, seen = [], set()
    for number in _NUMBER.findall(summary):
        if _digits(number) not in known and number not in seen:
            seen.add(number)
            found.append(number)
    return found
