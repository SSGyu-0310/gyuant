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

    class DummyFMPClient:
        def quote(self, symbols):
            recorded["calls"] += 1
            return [{"symbol": s, "price": 1.0, "previousClose": 1.0} for s in symbols]

    monkeypatch.setattr(flask_app, "get_fmp_client", lambda: DummyFMPClient())
    with flask_app.app.test_request_context("/"):
        res = flask_app.fetch_price_map(["AAPL", "MSFT"])
        assert recorded["calls"] == 1
        assert isinstance(res, dict)
