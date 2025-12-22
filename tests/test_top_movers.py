import json
from pathlib import Path

import pandas as pd

import flask_app


def _setup_prices(tmp_path, rows):
    data_dir = tmp_path / "us_market"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "us_daily_prices.csv", index=False)
    flask_app.US_DIR = data_dir
    flask_app.ensure_contracts = lambda *args, **kwargs: None
    return flask_app.app.test_client()


def test_top_movers_no_file(tmp_path):
    data_dir = tmp_path / "us_market"
    data_dir.mkdir(parents=True, exist_ok=True)
    flask_app.US_DIR = data_dir
    flask_app.ensure_contracts = lambda *args, **kwargs: None
    client = flask_app.app.test_client()
    resp = client.get("/top-movers")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["data"]["gainers"] == []
    assert data["data"]["losers"] == []


def test_top_movers_1d(tmp_path):
    rows = [
        {"ticker": "AAPL", "date": "2024-01-01", "close": 100},
        {"ticker": "AAPL", "date": "2024-01-02", "close": 110},
        {"ticker": "TSLA", "date": "2024-01-01", "close": 200},
        {"ticker": "TSLA", "date": "2024-01-02", "close": 180},
    ]
    client = _setup_prices(tmp_path, rows)
    resp = client.get("/top-movers?window=1d&limit=10")
    assert resp.status_code == 200
    data = resp.get_json()
    gainers = data["data"]["gainers"]
    losers = data["data"]["losers"]
    assert gainers[0]["ticker"] == "AAPL"
    assert losers[0]["ticker"] == "TSLA"
    assert gainers[0]["pct_change"] == 10.0
    assert losers[0]["pct_change"] == -10.0


def test_top_movers_1w(tmp_path):
    rows = []
    closes = [10, 11, 12, 13, 14, 20]
    for i, c in enumerate(closes):
        rows.append({"ticker": "AAPL", "date": f"2024-01-0{i+1}", "close": c})
    client = _setup_prices(tmp_path, rows)
    resp = client.get("/top-movers?window=1w&limit=5")
    assert resp.status_code == 200
    data = resp.get_json()
    gainers = data["data"]["gainers"]
    assert len(gainers) == 1
    assert gainers[0]["pct_change"] == 100.0
