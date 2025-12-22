def _assert_json(resp):
    assert resp.status_code != 500
    assert resp.is_json
    return resp.get_json()


def test_health(make_client):
    client = make_client()
    resp = client.get("/health")
    data = _assert_json(resp)
    assert resp.status_code == 200
    assert data.get("ok") is True


def test_endpoints_with_no_data(make_client):
    client = make_client()
    paths = [
        "/api/us/portfolio",
        "/api/us/smart-money",
        "/api/us/etf-flows",
        "/api/us/macro-analysis",
        "/api/us/options-flow",
        "/status",
    ]
    for path in paths:
        resp = client.get(path)
        assert resp.status_code in (200, 404)
        data = _assert_json(resp)
        if resp.status_code == 200:
            assert isinstance(data, dict)


def test_endpoints_with_data(make_client):
    client = make_client(with_data=True)
    cases = [
        ("/api/us/portfolio", lambda d: isinstance(d.get("market_indices", []), list)),
        ("/api/us/smart-money", lambda d: isinstance(d.get("top_picks", []), list)),
        ("/api/us/etf-flows", lambda d: isinstance(d.get("top_inflows", []), list)),
        ("/api/us/macro-analysis", lambda d: isinstance(d.get("macro_indicators", {}), dict)),
        ("/api/us/options-flow", lambda d: isinstance(d.get("options_flow", []), list) or isinstance(d.get("options"), list)),
        ("/status", lambda d: isinstance(d.get("modules", {}), dict)),
    ]
    for path, check in cases:
        resp = client.get(path)
        assert resp.status_code in (200, 404)
        data = _assert_json(resp)
        if resp.status_code == 200:
            assert check(data)
