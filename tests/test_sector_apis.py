import pandas as pd

import flask_app


def _setup_prices(tmp_path, rows, sectors=None):
    data_dir = tmp_path / "us_market"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "us_daily_prices.csv", index=False)
    if sectors:
        pd.DataFrame(sectors).to_csv(data_dir / "sectors.csv", index=False)
    flask_app.US_DIR = data_dir
    flask_app.ensure_contracts = lambda *args, **kwargs: None
    return flask_app.app.test_client()


def test_heatmap_no_sector_map(tmp_path):
    client = _setup_prices(tmp_path, [])
    resp = client.get("/sectors/heatmap")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["data"] == []
    assert "NO_SECTOR_MAP" in data["errors"]


def test_heatmap_with_sector_map(tmp_path):
    rows = [
        {"ticker": "AAPL", "date": "2024-01-01", "close": 100},
        {"ticker": "AAPL", "date": "2024-01-02", "close": 110},
        {"ticker": "TSLA", "date": "2024-01-01", "close": 200},
        {"ticker": "TSLA", "date": "2024-01-02", "close": 180},
    ]
    sectors = [{"ticker": "AAPL", "sector": "TECH"}, {"ticker": "TSLA", "sector": "AUTO"}]
    client = _setup_prices(tmp_path, rows, sectors)
    resp = client.get("/sectors/heatmap?window=1d")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["data"]) == 2


def test_sector_movers(tmp_path):
    rows = [
        {"ticker": "AAPL", "date": "2024-01-01", "close": 100},
        {"ticker": "AAPL", "date": "2024-01-02", "close": 110},
        {"ticker": "TSLA", "date": "2024-01-01", "close": 200},
        {"ticker": "TSLA", "date": "2024-01-02", "close": 180},
    ]
    sectors = [{"ticker": "AAPL", "sector": "TECH"}, {"ticker": "TSLA", "sector": "AUTO"}]
    client = _setup_prices(tmp_path, rows, sectors)
    resp = client.get("/sectors/TECH/movers?window=1d&limit=5")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["data"]["sector"] == "TECH"
    assert data["data"]["gainers"][0]["ticker"] == "AAPL"
