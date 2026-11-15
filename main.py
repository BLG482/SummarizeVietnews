"""Entrypoint FastAPI mỏng cho Vercel (phục vụ giao diện tĩnh, gọi API sang Modal)."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Đối chiếu tóm tắt: Seq2Seq và ViT5")


@app.get("/")
@app.get("/index.html")
@app.get("/frontend")
@app.get("/frontend/")
@app.get("/frontend/index.html")
def serve_index():
    return HTMLResponse((FRONTEND / "index.html").read_text(encoding="utf-8"))


@app.get("/styles.css")
@app.get("/frontend/styles.css")
def serve_styles():
    return Response(
        (FRONTEND / "styles.css").read_text(encoding="utf-8"),
        media_type="text/css; charset=utf-8",
    )


@app.get("/app.js")
@app.get("/frontend/app.js")
def serve_js():
    return Response(
        (FRONTEND / "app.js").read_text(encoding="utf-8"),
        media_type="application/javascript; charset=utf-8",
    )
