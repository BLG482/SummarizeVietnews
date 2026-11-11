"""Mô hình dữ liệu (SQLAlchemy 2.0). Mặc định SQLite; đặt DATABASE_URL để dùng PostgreSQL."""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker)

from .config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    article: Mapped[str] = mapped_column(Text)
    reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="manual")  # manual | testset
    sample_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_words: Mapped[int] = mapped_column(Integer, default=0)

    results: Mapped[list["Result"]] = relationship(
        back_populates="submission", cascade="all, delete-orphan", order_by="Result.id")
    vote: Mapped["Vote | None"] = relationship(
        back_populates="submission", uselist=False, cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    model_key: Mapped[str] = mapped_column(String(20), index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    truncated: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rouge1: Mapped[float | None] = mapped_column(Float, nullable=True)
    rouge2: Mapped[float | None] = mapped_column(Float, nullable=True)
    rougeL: Mapped[float | None] = mapped_column(Float, nullable=True)
    unsupported_numbers: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    submission: Mapped[Submission] = relationship(back_populates="results")


class Vote(Base):
    __tablename__ = "votes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), unique=True, index=True)
    choice: Mapped[str] = mapped_column(String(10))  # seq2seq | vit5 | tie
    blind: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    submission: Mapped[Submission] = relationship(back_populates="vote")


def _make_engine():
    url = settings.database_url
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        path = url.replace("sqlite:///", "", 1)
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True, **kwargs)


engine = _make_engine()
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
