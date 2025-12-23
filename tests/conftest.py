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
def mock_fmp_client():
    class DummyFMP:
        def quote(self, symbols):
            if isinstance(symbols, str):
                symbols_list = [symbols]
            else:
                symbols_list = list(symbols)
            return [
                {
                    "symbol": s,
                    "price": 1.0,
                    "previousClose": 1.0,
                    "open": 1.0,
                    "dayHigh": 1.0,
                    "dayLow": 1.0,
                    "timestamp": 1704067200,
                }
                for s in symbols_list
            ]

        def profile(self, symbol):
            return [{"symbol": symbol, "sector": "Technology", "companyName": "Dummy Co"}]

        def historical_price_full(self, symbol, from_date=None, to_date=None):
            return {
                "symbol": symbol,
                "historical": [
                    {"date": "2024-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 100},
                    {"date": "2024-01-02", "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.1, "volume": 100},
                ],
            }

        def treasury_rates(self, from_date=None, to_date=None):
            return [{"10Y": 4.0, "2Y": 4.5}, {"10Y": 3.9, "2Y": 4.4}]

    return DummyFMP()


@pytest.fixture
def make_client(tmp_path, monkeypatch, mock_fmp_client):
    def _make(with_data: bool = False):
        data_dir = tmp_path / ("data_with" if with_data else "data_empty")
        data_dir.mkdir()
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("HISTORY_DIR", str(tmp_path / "history"))
        monkeypatch.setenv("DATA_PROVIDER", "FMP")
        if with_data:
            _prepare_minimal_files(data_dir)
        if "flask_app" in sys.modules:
            app_module = sys.modules["flask_app"]
        else:
            import flask_app as app_module  # type: ignore
        importlib.reload(app_module)
        monkeypatch.setattr(app_module, "get_fmp_client", lambda: mock_fmp_client)
        return app_module.app.test_client()

    return _make
