import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv()
    root_env = Path(__file__).resolve().parents[1] / ".env"
    if root_env.exists():
        load_dotenv(root_env)

logger = logging.getLogger(__name__)

DEFAULT_FMP_BASE_URL = "https://financialmodelingprep.com"


class FMPClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
    ) -> None:
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        self.base_url = (base_url or os.getenv("FMP_BASE_URL", DEFAULT_FMP_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.session = requests.Session()

    def _build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.api_key:
            logger.warning("FMP_API_KEY is not set; skipping request to %s", path)
            return None
        url = self._build_url(path)
        payload = dict(params or {})
        payload["apikey"] = self.api_key

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self.session.get(url, params=payload, timeout=self.timeout)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.debug("FMP GET %s status=%s latency_ms=%s", path, resp.status_code, latency_ms)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                logger.warning("FMP request failed: %s", exc)
                return None

        if last_exc:
            logger.warning("FMP request failed: %s", last_exc)
        return None

    def quote(self, symbols: Iterable[str]) -> List[Dict[str, Any]]:
        symbols = [s for s in symbols if s]
        if not symbols:
            return []
        data = self.get_json(f"/api/v3/quote/{','.join(symbols)}")
        return data if isinstance(data, list) else []

    def profile(self, symbol: str) -> Dict[str, Any]:
        data = self.get_json(f"/api/v3/profile/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def historical_price_full(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self.get_json(f"/api/v3/historical-price-full/{symbol}", params=params)
        return data if isinstance(data, dict) else {}

    def historical_chart(
        self,
        symbol: str,
        interval: str = "1min",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self.get_json(f"/api/v3/historical-chart/{interval}/{symbol}", params=params)
        return data if isinstance(data, list) else []

    def key_metrics_ttm(self, symbol: str) -> Dict[str, Any]:
        data = self.get_json(f"/api/v3/key-metrics-ttm/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def ratios_ttm(self, symbol: str) -> Dict[str, Any]:
        data = self.get_json(f"/api/v3/ratios-ttm/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def ratings_snapshot(self, symbol: str) -> Dict[str, Any]:
        data = self.get_json("/stable/ratings-snapshot", params={"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def price_target_consensus(self, symbol: str) -> Dict[str, Any]:
        data = self.get_json("/stable/price-target-consensus", params={"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def insider_trading(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        data = self.get_json("/stable/insider-trading/search", params={"symbol": symbol, "limit": limit})
        return data if isinstance(data, list) else []

    def treasury_rates(self) -> List[Dict[str, Any]]:
        data = self.get_json("/stable/treasury-rates")
        return data if isinstance(data, list) else []


_CLIENT: Optional[FMPClient] = None


def get_fmp_client() -> FMPClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = FMPClient()
    return _CLIENT
