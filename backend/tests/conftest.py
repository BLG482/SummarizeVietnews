import csv
import os
import sys
import tempfile
from pathlib import Path

# Phải đặt biến môi trường trước khi import app.
_tmp = Path(tempfile.mkdtemp())
os.environ.update({
    "USE_MOCK_MODELS": "1",
    "PRELOAD_MODELS": "0",
    "DATABASE_URL": f"sqlite:///{_tmp / 'test.db'}",
    "TESTSET_CSV": str(_tmp / "test.csv"),
})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with open(_tmp / "test.csv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["title", "summary", "article"])
    writer.writeheader()
    writer.writerow({
        "title": "Chủ_tịch Quốc_hội thăm Trung_Quốc",
        "summary": "Chủ_tịch Quốc_hội lên_đường thăm chính_thức Trung_Quốc .",
        "article": "Sáng 11-10 , Chủ_tịch Quốc_hội lên_đường thăm chính_thức Trung_Quốc .\n"
                   "Đoàn có 25 thành_viên tham_gia chuyến thăm lần này .\n"
                   "Chuyến thăm nhằm tăng_cường hợp_tác giữa hai cơ_quan lập_pháp .",
    })

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c
