from typing import Literal

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    title: str = Field("", max_length=1000)
    article: str = Field(..., min_length=30, max_length=60000)
    reference: str | None = Field(None, max_length=5000)
    models: list[Literal["seq2seq", "vit5"]] | None = None
    sample_id: int | None = None


class VoteRequest(BaseModel):
    choice: Literal["seq2seq", "vit5", "tie"]
    blind: bool = False
