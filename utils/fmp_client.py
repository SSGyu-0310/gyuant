import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urljoin

import requests

logger = logging.getLogger("fmp_client")

DEFAULT_BASE_URL = "https://financialmodelingprep.com"
DEFAULT_TIMEOUT_SEC = float(os.getenv("FMP_TIMEOUT_SEC", "15"))
DEFAULT_MAX_RETRIES = int(os.getenv("FMP_MAX_RETRIES", "2"))
DEFAULT_BACKOFF_SEC = float(os.getenv("FMP_RETRY_BACKOFF_SEC", "1.0"))


SYMBOL_MAP = {
    "BTC-USD": "BTCUSD",
    "ETH-USD": "ETHUSD",
    "GC=F": "GCUSD",
    "CL=F": "CLUSD",
    "SI=F": "SIUSD",
    "NG=F": "NGUSD",
    "HG=F": "HGUSD",
    "KRW=X": "USDKRW",
    "JPY=X": "USDJPY",
    "EURUSD=X": "EURUSD",
    "DX-Y.NYB": "DXY",
    "BRK-B": "BRK.B",
    "BF-B": "BF.B",
}


def map_symbol(symbol: str) -> str:
    sym = str(symbol).strip().upper()
    return SYMBOL_MAP.get(sym, sym)


def map_symbols(symbols: Sequence[str]) -> List[str]:
    return [map_symbol(sym) for sym in symbols]


def get_data_provider() -> str:
    return str(os.getenv("DATA_PROVIDER", "FMP")).strip().upper()


class FMPClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        max_retries: Optional[int] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self.base_url = (base_url or os.getenv("FMP_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_sec = timeout_sec if timeout_sec is not None else DEFAULT_TIMEOUT_SEC
        self.max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
        self.session = session or requests.Session()

    def _bump_perf(self, key: str) -> None:
        try:
            from utils.perf_utils import get_perf_stats

            stats = get_perf_stats()
            stats[key] = stats.get(key, 0) + 1
        except Exception:
            return

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("FMP_API_KEY is not set")

    def _build_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._require_key()
        params = dict(params or {})
        params.setdefault("apikey", self.api_key)
        url = self._build_url(path)
        backoff = DEFAULT_BACKOFF_SEC
        last_exc = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_sec)
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.info("FMP GET %s %s %sms", url, resp.status_code, latency_ms)
                self._bump_perf("fmp_calls")
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("FMP request failed without exception")

    def quote(self, symbols: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
        if isinstance(symbols, str):
            symbol_list = [symbols]
        else:
            symbol_list = list(symbols)
        mapped = map_symbols(symbol_list)
        if not mapped:
            return []
        if len(mapped) == 1:
            data = self.get_json("/stable/quote", {"symbol": mapped[0]})
        else:
            data = self.get_json("/stable/batch-quote", {"symbols": ",".join(mapped)})
            self._bump_perf("fmp_batches")
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            if "symbol" in data:
                return [data]
            return []
        return data or []

    def profile(self, symbol: str) -> List[Dict[str, Any]]:
        data = self.get_json("/stable/profile", {"symbol": map_symbol(symbol)})
        if isinstance(data, dict) and "data" in data:
            return data["data"] or []
        return data or []

    def historical_price_full(
        self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self.get_json(f"/api/v3/historical-price-full/{map_symbol(symbol)}", params)

    def historical_chart(
        self, symbol: str, interval: str = "1min", from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"symbol": map_symbol(symbol)}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self.get_json(f"/stable/historical-chart/{interval}", params)
        return data or []

    def key_metrics_ttm(self, symbol: str) -> List[Dict[str, Any]]:
        data = self.get_json("/stable/key-metrics-ttm", {"symbol": map_symbol(symbol)})
        return data.get("data") if isinstance(data, dict) else data or []

    def ratios_ttm(self, symbol: str) -> List[Dict[str, Any]]:
        data = self.get_json("/stable/ratios-ttm", {"symbol": map_symbol(symbol)})
        return data.get("data") if isinstance(data, dict) else data or []

    def ratings_snapshot(self, symbol: str) -> List[Dict[str, Any]]:
        data = self.get_json("/stable/ratings-snapshot", {"symbol": map_symbol(symbol)})
        return data.get("data") if isinstance(data, dict) else data or []

    def price_target_consensus(self, symbol: str) -> List[Dict[str, Any]]:
        data = self.get_json("/stable/price-target-consensus", {"symbol": map_symbol(symbol)})
        return data.get("data") if isinstance(data, dict) else data or []

    def insider_trading(self, symbol: str, page: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        params = {"symbol": map_symbol(symbol), "page": page, "limit": limit}
        data = self.get_json("/api/v4/insider-trading", params)
        return data or []

    def institutional_ownership(self, symbol: str) -> List[Dict[str, Any]]:
        params = {"symbol": map_symbol(symbol), "includeCurrentQuarter": "false"}
        data = self.get_json("/api/v4/institutional-ownership/symbol-ownership", params)
        return data or []

    def institutional_holders(self, symbol: str) -> List[Dict[str, Any]]:
        data = self.get_json(f"/api/v3/institutional-holder/{map_symbol(symbol)}")
        return data or []

    def treasury_rates(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self.get_json("/stable/treasury-rates", params)
        return data.get("data") if isinstance(data, dict) else data or []

    def economic_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self.get_json("/stable/economic-calendar", params)
        return data.get("data") if isinstance(data, dict) else data or []


_CLIENT: Optional[FMPClient] = None


def get_fmp_client() -> FMPClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = FMPClient()
    return _CLIENT
