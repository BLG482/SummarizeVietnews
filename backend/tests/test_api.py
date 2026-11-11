ARTICLE = ("Sáng 11-10, Chủ tịch Quốc hội lên đường thăm chính thức Trung Quốc. "
           "Đoàn có 25 thành viên tham gia chuyến thăm lần này. "
           "Chuyến thăm nhằm tăng cường hợp tác giữa hai cơ quan lập pháp.")
REFERENCE = "Sáng 11-10, Chủ tịch Quốc hội lên đường thăm chính thức Trung Quốc."


def test_models_listed(client):
    data = client.get("/api/models").json()
    assert [m["key"] for m in data] == ["seq2seq", "vit5"]


def test_summarize_scores_and_saves(client):
    res = client.post("/api/summarize", json={"title": "Thăm Trung Quốc", "article": ARTICLE,
                                              "reference": REFERENCE})
    assert res.status_code == 200
    body = res.json()
    by_key = {r["model_key"]: r for r in body["results"]}
    assert set(by_key) == {"seq2seq", "vit5"}
    for result in by_key.values():
        assert result["summary"] and result["error"] is None
        assert result["latency_ms"] > 0
        assert 0 <= result["rougeL"] <= 100
    # Mô hình giả seq2seq chỉ lấy câu đầu, trùng khít tham chiếu.
    assert by_key["seq2seq"]["rouge1"] == 100.0
    assert client.get(f"/api/submissions/{body['id']}").json()["reference"] == REFERENCE


def test_no_reference_means_no_rouge(client):
    body = client.post("/api/summarize", json={"article": ARTICLE}).json()
    assert all(r["rouge1"] is None for r in body["results"])


def test_validation_rejects_short_article(client):
    assert client.post("/api/summarize", json={"article": "quá ngắn"}).status_code == 422


def test_unsupported_numbers_flagged(client):
    from app.scoring import unsupported_numbers
    assert unsupported_numbers("Có 30 người và 25 xe", "Đoàn có 25 thành viên") == ["30"]
    assert unsupported_numbers("Giá 1.500 đồng", "giá 1500 đồng") == []


def test_vote_replaces_previous_and_feeds_stats(client):
    sid = client.post("/api/summarize", json={"article": ARTICLE, "reference": REFERENCE}).json()["id"]
    client.post(f"/api/submissions/{sid}/vote", json={"choice": "vit5", "blind": True})
    after = client.post(f"/api/submissions/{sid}/vote", json={"choice": "tie", "blind": False}).json()
    assert after["vote"] == {"choice": "tie", "blind": False}
    stats = client.get("/api/stats").json()
    assert stats["submissions"] >= 1 and stats["votes"]["tie"] >= 1
    assert stats["models"]["vit5"]["runs"] >= 1 and stats["models"]["vit5"]["scored"] >= 1


def test_history_pagination(client):
    page = client.get("/api/submissions", params={"limit": 2}).json()
    assert page["total"] >= 2 and len(page["items"]) == 2
    assert "article" not in page["items"][0]


def test_samples_keep_segmented_text_for_models(client):
    from app.models import registry
    listing = client.get("/api/samples").json()
    assert listing["available"] and listing["total"] == 1
    sample = client.get("/api/samples/0").json()
    assert "_" not in sample["article"]  # hiển thị tự nhiên cho người đọc
    client.post("/api/summarize", json={"title": sample["title"], "article": sample["article"],
                                        "reference": sample["reference"], "sample_id": 0})
    title, article, raw = registry["vit5"].last_input
    assert raw is True and "Chủ_tịch" in article  # mô hình nhận bản gốc đã tách từ


def test_edited_sample_is_treated_as_free_text(client):
    from app.models import registry
    body = client.post("/api/summarize", json={
        "article": "Một bài hoàn toàn khác với nội dung được người dùng tự viết ra ở đây.",
        "sample_id": 0}).json()
    assert body["source"] == "manual" and body["sample_id"] is None
    assert registry["vit5"].last_input[2] is False


def test_one_model_failing_does_not_break_the_other(client, monkeypatch):
    from app.models import registry

    def boom(*args, **kwargs):
        raise RuntimeError("hết bộ nhớ GPU")

    monkeypatch.setattr(registry["vit5"], "_summarize", boom)
    body = client.post("/api/summarize", json={"article": ARTICLE}).json()
    by_key = {r["model_key"]: r for r in body["results"]}
    assert by_key["seq2seq"]["summary"] and by_key["seq2seq"]["error"] is None
    assert "hết bộ nhớ GPU" in by_key["vit5"]["error"]
    assert client.get("/api/stats").json()["models"]["vit5"]["errors"] >= 1


def test_unknown_model_key_rejected(client):
    assert client.post("/api/summarize", json={"article": ARTICLE, "models": ["gpt"]}).status_code == 422
