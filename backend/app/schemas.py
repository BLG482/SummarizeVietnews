import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SummarizeRequest(BaseModel):
    title: str = Field("", max_length=1000)
    article: str = Field(..., min_length=30, max_length=60000)
    reference: str | None = Field(None, max_length=5000)
    models: list[Literal["seq2seq", "vit5"]] | None = None
    sample_id: int | None = None

    @field_validator("title", "article", "reference", mode="before")
    @classmethod
    def _sanitize_text(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Giá trị văn bản không hợp lệ.")
        return _CONTROL_CHARS_RE.sub("", value)


class VoteRequest(BaseModel):
    choice: Literal["seq2seq", "vit5", "tie"]
    blind: bool = False
