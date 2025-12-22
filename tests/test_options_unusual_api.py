import json
from pathlib import Path

import flask_app


def make_client(tmp_path, data):
    data_dir = tmp_path / "us_market"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "options_flow.json").write_text(json.dumps(data), encoding="utf-8")
    flask_app.US_DIR = data_dir
    flask_app.ensure_contracts = lambda *args, **kwargs: None
    return flask_app.app.test_client()


def test_unusual_no_file_returns_empty(tmp_path):
    data_dir = tmp_path / "us_market"
    data_dir.mkdir(parents=True, exist_ok=True)
    flask_app.US_DIR = data_dir
    flask_app.ensure_contracts = lambda *args, **kwargs: None
    client = flask_app.app.test_client()
    resp = client.get("/options/unusual")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["data"]["count"] == 0
    assert data["data"]["items"] == []


def test_unusual_min_score_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(flask_app, "OPT_PREMIUM_HIGH_USD", 1000.0)
    monkeypatch.setattr(flask_app, "OPT_PREMIUM_MID_USD", 500.0)
    monkeypatch.setattr(flask_app, "OPT_VOL_OI_RATIO_HIGH", 1.0)
    monkeypatch.setattr(flask_app, "OPT_VOL_OI_RATIO_MID", 0.5)
    monkeypatch.setattr(flask_app, "OPT_VOLUME_LARGE", 1.0)
    monkeypatch.setattr(flask_app, "OPT_IV_HIGH", 10.0)
    raw = {
        "options_flow": [
            {"ticker": "TSLA", "type": "call", "premium": 2000, "volume": 10, "open_interest": 1},
            {"ticker": "AAPL", "type": "put", "premium": 10, "volume": 1, "open_interest": 10},
        ]
    }
    client = make_client(tmp_path, raw)
    resp = client.get("/options/unusual?min_score=20&sort=score")
    assert resp.status_code == 200
    data = resp.get_json()
    items = data["data"]["items"]
    assert len(items) == 1
    assert items[0]["ticker"] == "TSLA"
