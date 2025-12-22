import json
import time
from pathlib import Path

import pytest

import flask_app


@pytest.fixture
def ai_setup(tmp_path, monkeypatch):
    data_dir = tmp_path / "us_market"
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries_file = data_dir / "ai_summaries.json"
    summaries_file.write_text(
        json.dumps({"TSLA": {"summary": "<script>alert(1)</script>hello", "updated": "2024-01-01T00:00:00Z"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(flask_app, "US_DIR", data_dir)
    monkeypatch.setattr(flask_app, "ensure_contracts", lambda *args, **kwargs: None)
    monkeypatch.setattr(flask_app, "AI_SUMMARY_CACHE", {})
    monkeypatch.setattr(flask_app, "AI_LAST_REGEN", {})
    monkeypatch.setattr(flask_app, "AI_SUMMARY_LOCKS", {})
    return summaries_file


def write_summary(path: Path, text: str):
    data = {"TSLA": {"summary": text, "updated": "2024-01-01T00:00:00Z"}}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_sanitize_and_validation(ai_setup):
    client = flask_app.app.test_client()
    resp = client.get("/api/us/ai-summary/TSLA")
    assert resp.status_code == 200
    summary = resp.get_json()["summary"]
    assert "<script>" not in summary  # escaped
    bad = client.get("/api/us/ai-summary/???")
    assert bad.status_code == 400


def test_truncate_flag(ai_setup, monkeypatch):
    monkeypatch.setattr(flask_app, "AI_SUMMARY_MAX_CHARS", 5)
    write_summary(ai_setup, "abcdefghij")
    client = flask_app.app.test_client()
    resp = client.get("/api/us/ai-summary/TSLA")
    data = resp.get_json()
    assert data["meta"]["truncated"] is True
    assert len(data["summary"]) <= flask_app.AI_SUMMARY_MAX_CHARS + 1  # allow ellipsis


def test_cache_hit_sets_cached_flag(ai_setup, monkeypatch):
    monkeypatch.setattr(flask_app, "AI_SUMMARY_TTL_SEC", 100)
    client = flask_app.app.test_client()
    first = client.get("/api/us/ai-summary/TSLA").get_json()
    assert first["meta"]["cached"] is False
    second_resp = client.get("/api/us/ai-summary/TSLA")
    assert second_resp.status_code == 200
    second = second_resp.get_json()
    assert second["meta"]["cached"] is True


def test_force_regen_cooldown(ai_setup, monkeypatch):
    monkeypatch.setattr(flask_app, "AI_SUMMARY_REGEN_COOLDOWN_SEC", 2)
    client = flask_app.app.test_client()
    ok = client.get("/api/us/ai-summary/TSLA?force=1")
    assert ok.status_code == 200
    too_soon = client.get("/api/us/ai-summary/TSLA?force=1")
    assert too_soon.status_code == 429
    retry_after = too_soon.get_json().get("retry_after_sec")
    assert retry_after is not None
