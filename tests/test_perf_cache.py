import types
import pandas as pd

import flask_app
from utils import perf_utils


def test_request_cache_loader_runs_once(monkeypatch):
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        return 123

    with flask_app.app.test_request_context("/"):
        a = perf_utils.cache_get_or_set(("k", 1), loader)
        b = perf_utils.cache_get_or_set(("k", 1), loader)
        assert a == b == 123
        assert calls["count"] == 1


def test_fetch_price_map_batches(monkeypatch):
    recorded = {"calls": 0}

    def fake_download(tickers, period=None, progress=None, threads=None, group_by=None):
        recorded["calls"] += 1
        if isinstance(tickers, str):
            tickers_list = [tickers]
        else:
            tickers_list = tickers
        data = pd.DataFrame({t: [1.0] for t in tickers_list})

        class FakeDf:
            empty = False

            def __getitem__(self, key):
                return data if key == "Close" else data

        return FakeDf()

    monkeypatch.setattr(flask_app, "yf", types.SimpleNamespace(download=fake_download))
    with flask_app.app.test_request_context("/"):
        res = flask_app.fetch_price_map(["AAPL", "MSFT"])
        assert recorded["calls"] == 1
        assert isinstance(res, dict)
