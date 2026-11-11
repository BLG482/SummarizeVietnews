"""Bài mẫu lấy từ file CSV test (cột title, article, summary), đọc một lần rồi giữ trong bộ nhớ."""
import csv
import re
import sys
import threading
from pathlib import Path

from .config import settings

_lock = threading.Lock()
_rows: list[dict] | None = None


def _load() -> list[dict]:
    global _rows
    with _lock:
        if _rows is not None:
            return _rows
        path = Path(settings.testset_csv)
        if not path.exists():
            return []  # không lưu cache, để thêm file sau vẫn nhận được
        csv.field_size_limit(min(sys.maxsize, 2147483647))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            _rows = [{"title": r.get("title") or "", "article": r.get("article") or "",
                      "summary": r.get("summary") or ""} for r in csv.DictReader(handle)]
        return _rows


def display(text: str) -> str:
    """Bỏ '_' của từ ghép và dấu cách thừa trước dấu câu để người đọc thấy tự nhiên."""
    text = text.replace("_", " ")
    text = re.sub(r"[ \t]+([,.;:!?%\)\]\}])", r"\1", text)
    return re.sub(r"([\(\[\{])[ \t]+", r"\1", text)


def _canon(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).casefold()


def same_text(raw: str, shown: str) -> bool:
    """Người dùng chưa sửa nội dung bài mẫu (so sánh bỏ qua dấu câu, khoảng trắng và '_')."""
    return _canon(raw) == _canon(shown)


def get(sample_id: int) -> dict | None:
    rows = _load()
    return rows[sample_id] if 0 <= sample_id < len(rows) else None


def public(sample_id: int, row: dict) -> dict:
    return {"id": sample_id, "title": display(row["title"]),
            "article": display(row["article"]), "reference": display(row["summary"])}


def page(query: str, offset: int, limit: int) -> dict:
    rows = _load()
    needle = query.strip().casefold()
    matches = [(i, r) for i, r in enumerate(rows)
               if not needle or needle in display(r["title"]).casefold()]
    items = [{"id": i, "title": display(r["title"]),
              "preview": display(r["article"])[:180], "n_words": len(r["article"].split())}
             for i, r in matches[offset:offset + limit]]
    return {"available": bool(rows), "total": len(matches), "items": items,
            "path": settings.testset_csv}
