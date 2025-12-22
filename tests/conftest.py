import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import pytest


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _prepare_minimal_files(base: Path) -> None:
    _write_json(
        base / "smart_money_current.json",
        {"analysis_date": "2024-01-01", "analysis_timestamp": "2024-01-01T00:00:00Z", "picks": [], "summary": {"total_analyzed": 0, "avg_score": 0}},
    )
    _write_csv(
        base / "smart_money_picks_v2.csv",
        ["ticker", "name", "current_price", "smart_money_score", "composite_score", "target_upside", "grade"],
        [{"ticker": "AAPL", "name": "Apple", "current_price": 10, "smart_money_score": 80, "composite_score": 80, "target_upside": 5, "grade": "A"}],
    )
    _write_csv(
        base / "us_etf_flows.csv",
        ["ticker", "name", "category", "current_price", "price_1w_pct", "price_1m_pct", "vol_ratio_5d_20d", "obv_change_20d_pct", "avg_volume_20d", "flow_score", "flow_status"],
        [{"ticker": "SPY", "name": "SPY", "category": "Market", "current_price": 1, "price_1w_pct": 0, "price_1m_pct": 0, "vol_ratio_5d_20d": 1, "obv_change_20d_pct": 0, "avg_volume_20d": 1, "flow_score": 50, "flow_status": "Neutral"}],
    )
    _write_json(base / "etf_flow_analysis.json", {"ai_analysis": "", "timestamp": "2024-01-01T00:00:00Z"})
    _write_json(base / "macro_analysis.json", {"macro_indicators": {"VIX": {"value": 10}}, "ai_analysis": "ok", "timestamp": "2024-01-01T00:00:00Z"})
    _write_json(base / "macro_analysis_en.json", {"macro_indicators": {"VIX": {"value": 10}}, "ai_analysis": "ok", "timestamp": "2024-01-01T00:00:00Z"})
    _write_json(base / "options_flow.json", {"options_flow": [], "summary": {}, "timestamp": "2024-01-01T00:00:00Z"})
    _write_json(base / "weekly_calendar.json", {"week_start": "2024-01-01", "week_end": "2024-01-07", "events": []})
    _write_csv(base / "us_daily_prices.csv", ["ticker", "date", "close"], [{"ticker": "AAPL", "date": "2024-01-01", "close": 1}])
    _write_csv(base / "us_volume_analysis.csv", ["ticker", "name", "supply_demand_score", "supply_demand_stage"], [{"ticker": "AAPL", "name": "Apple", "supply_demand_score": 50, "supply_demand_stage": "Neutral"}])
    _write_csv(base / "us_13f_holdings.csv", ["ticker", "institutional_pct", "institutional_score"], [{"ticker": "AAPL", "institutional_pct": 90, "institutional_score": 60}])
    _write_csv(base / "us_stocks_list.csv", ["ticker", "name", "sector", "market"], [{"ticker": "AAPL", "name": "Apple", "sector": "Tech", "market": "S&P500"}])


@pytest.fixture
def mock_yfinance(monkeypatch):
    import yfinance as yf

    class DummyTicker:
        def __init__(self, ticker):
            self.ticker = ticker
            self.info = {"sector": "Technology"}
            idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
            self._hist = pd.DataFrame(
                {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2], "Volume": [100, 100]},
                index=idx,
            )

        @property
        def options(self):
            return ["2024-01-19"]

        def history(self, period="1y"):
            return self._hist

        def option_chain(self, exp):
            df = pd.DataFrame({"volume": [0], "openInterest": [0], "impliedVolatility": [0]})
            return type("OC", (), {"calls": df, "puts": df})

    def fake_download(tickers, *args, **kwargs):
        idx = pd.date_range("2024-01-01", periods=5, tz="UTC")
        if isinstance(tickers, list):
            cols_close = pd.MultiIndex.from_product([["Close"], tickers])
            cols_vol = pd.MultiIndex.from_product([["Volume"], tickers])
            df_close = pd.DataFrame(1, index=idx, columns=cols_close)
            df_vol = pd.DataFrame(1, index=idx, columns=cols_vol)
            return pd.concat([df_close, df_vol], axis=1)
        return pd.DataFrame({"Close": [1, 2], "Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Volume": [1, 1]}, index=idx[:2])

    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    monkeypatch.setattr(yf, "download", fake_download)
    return yf


@pytest.fixture
def make_client(tmp_path, monkeypatch, mock_yfinance):
    def _make(with_data: bool = False):
        data_dir = tmp_path / ("data_with" if with_data else "data_empty")
        data_dir.mkdir()
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("HISTORY_DIR", str(tmp_path / "history"))
        if with_data:
            _prepare_minimal_files(data_dir)
        if "flask_app" in sys.modules:
            app_module = sys.modules["flask_app"]
        else:
            import flask_app as app_module  # type: ignore
        importlib.reload(app_module)
        return app_module.app.test_client()

    return _make
