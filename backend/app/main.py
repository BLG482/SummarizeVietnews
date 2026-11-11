"""Backend FastAPI: tóm tắt bằng hai mô hình, lưu kết quả, bình chọn và thống kê."""
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, selectinload

from . import samples
from .config import settings
from .db import Base, Result, Submission, Vote, engine, get_db
from .models import registry
from .schemas import SummarizeRequest, VoteRequest
from .scoring import rouge, unsupported_numbers


def _safe_load(model) -> None:
    try:
        model.ensure_loaded()
    except Exception:  # lỗi đã được ghi vào model.error
        pass


def _preload() -> None:
    for model in registry.values():
        _safe_load(model)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    if settings.preload:
        threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="Đối chiếu tóm tắt: Seq2Seq và ViT5", lifespan=lifespan)


# ---------- chuyển đối tượng DB thành JSON ----------
def _iso(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    if moment.tzinfo is None:  # SQLite không lưu múi giờ; giá trị được ghi theo UTC
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat()


def _result_dict(r: Result, full: bool) -> dict:
    data = {"model_key": r.model_key, "latency_ms": r.latency_ms, "truncated": r.truncated,
            "rouge1": r.rouge1, "rouge2": r.rouge2, "rougeL": r.rougeL, "error": r.error}
    if full:
        data.update(summary=r.summary, input_tokens=r.input_tokens,
                    unsupported_numbers=r.unsupported_numbers or [])
    return data


def _submission_dict(s: Submission, full: bool = True) -> dict:
    data = {"id": s.id, "created_at": _iso(s.created_at), "title": s.title, "n_words": s.n_words,
            "source": s.source, "sample_id": s.sample_id, "has_reference": bool(s.reference),
            "results": [_result_dict(r, full) for r in s.results],
            "vote": ({"choice": s.vote.choice, "blind": s.vote.blind} if s.vote else None)}
    if full:
        data.update(article=s.article, reference=s.reference)
    return data


# ---------- mô hình ----------
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/models")
def list_models():
    return [m.info() for m in registry.values()]


@app.post("/api/models/{key}/reload")
def reload_model(key: str):
    model = registry.get(key)
    if model is None:
        raise HTTPException(404, f"Không có mô hình '{key}'.")
    model.reload()
    threading.Thread(target=_safe_load, args=(model,), daemon=True).start()
    return model.info()


# ---------- tóm tắt ----------
@app.post("/api/summarize")
def summarize(req: SummarizeRequest, db: Session = Depends(get_db)):
    keys = req.models or list(registry)
    title, article = req.title.strip(), req.article.strip()
    reference = (req.reference or "").strip() or None

    # Bài mẫu chưa bị sửa: đưa bản gốc đã tách từ (có '_') vào mô hình, đúng như lúc huấn luyện.
    model_title, model_article, raw = title, article, False
    if req.sample_id is not None:
        sample = samples.get(req.sample_id)
        if sample and samples.same_text(sample["article"], article):
            model_title, model_article, raw = sample["title"], sample["article"], True

    def run(key: str) -> Result:
        try:
            out = registry[key].summarize(model_title, model_article, raw=raw)
            data = {"model_key": key, "summary": out.text, "latency_ms": out.latency_ms,
                    "truncated": out.truncated, "input_tokens": out.input_tokens,
                    "unsupported_numbers": unsupported_numbers(out.text, f"{title} {article}")}
            if reference:
                data.update(rouge(out.text, reference))
            return Result(**data)
        except Exception as exc:  # một mô hình lỗi không làm hỏng kết quả của mô hình kia
            return Result(model_key=key, error=f"{type(exc).__name__}: {exc}")

    # Chạy tuần tự từng mô hình để tránh tranh chấp CPU/RAM giữa hai mô hình ~226M tham số.
    results = [run(k) for k in keys]

    submission = Submission(
        title=title, article=article, reference=reference, n_words=len(article.split()),
        source="testset" if raw else "manual", sample_id=req.sample_id if raw else None,
        results=results)
    db.add(submission)
    db.commit()
    db.refresh(submission)
    return _submission_dict(submission)


# ---------- lịch sử và bình chọn ----------
@app.get("/api/submissions")
def list_submissions(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                     db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Submission.id))) or 0
    rows = db.scalars(
        select(Submission).options(selectinload(Submission.results), selectinload(Submission.vote))
        .order_by(Submission.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "items": [_submission_dict(s, full=False) for s in rows]}


@app.get("/api/submissions/{submission_id}")
def get_submission(submission_id: int, db: Session = Depends(get_db)):
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(404, "Không tìm thấy lượt tóm tắt này.")
    return _submission_dict(submission)


@app.post("/api/submissions/{submission_id}/vote")
def vote(submission_id: int, req: VoteRequest, db: Session = Depends(get_db)):
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(404, "Không tìm thấy lượt tóm tắt này.")
    if submission.vote:  # mỗi lượt chỉ giữ một bình chọn, bình chọn mới thay bình chọn cũ
        submission.vote.choice, submission.vote.blind = req.choice, req.blind
        submission.vote.created_at = datetime.now(timezone.utc)
    else:
        submission.vote = Vote(choice=req.choice, blind=req.blind)
    db.commit()
    db.refresh(submission)
    return _submission_dict(submission)


# ---------- thống kê ----------
@app.get("/api/stats")
def stats(db: Session = Depends(get_db)):
    models = {key: {"runs": 0, "errors": 0, "avg_latency_ms": None, "truncated_rate": None,
                    "scored": 0, "avg_rouge1": None, "avg_rouge2": None, "avg_rougeL": None}
              for key in registry}
    ok = db.execute(
        select(Result.model_key, func.count(Result.id), func.avg(Result.latency_ms),
               func.avg(Result.rouge1), func.avg(Result.rouge2), func.avg(Result.rougeL),
               func.count(Result.rouge1), func.sum(case((Result.truncated.is_(True), 1), else_=0)))
        .where(Result.error.is_(None)).group_by(Result.model_key)).all()
    for key, runs, latency, r1, r2, rl, scored, truncated in ok:
        if key in models:
            models[key].update(
                runs=runs, avg_latency_ms=latency, scored=scored, avg_rouge1=r1, avg_rouge2=r2,
                avg_rougeL=rl, truncated_rate=(truncated or 0) / runs if runs else None)
    for key, count in db.execute(select(Result.model_key, func.count(Result.id))
                                 .where(Result.error.is_not(None)).group_by(Result.model_key)).all():
        if key in models:
            models[key]["errors"] = count
    votes = dict(db.execute(select(Vote.choice, func.count(Vote.id)).group_by(Vote.choice)).all())
    return {"submissions": db.scalar(select(func.count(Submission.id))) or 0, "models": models,
            "votes": {"seq2seq": votes.get("seq2seq", 0), "vit5": votes.get("vit5", 0),
                      "tie": votes.get("tie", 0), "total": sum(votes.values()),
                      "blind": db.scalar(select(func.count(Vote.id)).where(Vote.blind.is_(True))) or 0}}


# ---------- bài mẫu ----------
@app.get("/api/samples")
def list_samples(q: str = "", offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    return samples.page(q, offset, limit)


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: int):
    row = samples.get(sample_id)
    if row is None:
        raise HTTPException(404, "Không có bài mẫu này (kiểm tra TESTSET_CSV).")
    return samples.public(sample_id, row)


# Giao diện tĩnh phục vụ cùng một cổng, đặt cuối để không che các đường /api.
if settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")
