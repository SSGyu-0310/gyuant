import html
import json
import logging
import math
import os
import re
import threading
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, g

from utils.logger import setup_logger
from utils.pipeline_utils import ensure_contracts
from utils.perf_utils import cache_get_or_set, get_perf_stats, get_request_cache, perf_env_threads_enabled, perf_max_workers
from utils.options_unusual import score_options_flow, load_options_raw
from collections import defaultdict

logger = setup_logger('flask_server', 'server.log')
request_logger = logging.getLogger('request_logger')
if not request_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    request_logger.addHandler(handler)
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
US_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "us_market")).resolve()
STALE_THRESHOLD_MIN = int(os.getenv("STALE_THRESHOLD_MIN", "60"))
AI_SUMMARY_MAX_CHARS = int(os.getenv("AI_SUMMARY_MAX_CHARS", "1500"))
AI_SUMMARY_TTL_SEC = int(os.getenv("AI_SUMMARY_TTL_SEC", "1800"))
AI_SUMMARY_REGEN_COOLDOWN_SEC = int(os.getenv("AI_SUMMARY_REGEN_COOLDOWN_SEC", "30"))
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")
LANG_WHITELIST = {"ko", "en"}
AI_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}
AI_SUMMARY_LOCKS: Dict[str, threading.Lock] = {}
AI_LAST_REGEN: Dict[str, datetime] = {}
OPT_PREMIUM_HIGH_USD = float(os.getenv("OPT_PREMIUM_HIGH_USD", "1000000"))
OPT_PREMIUM_MID_USD = float(os.getenv("OPT_PREMIUM_MID_USD", "300000"))
OPT_VOL_OI_RATIO_HIGH = float(os.getenv("OPT_VOL_OI_RATIO_HIGH", "5"))
OPT_VOL_OI_RATIO_MID = float(os.getenv("OPT_VOL_OI_RATIO_MID", "3"))
OPT_VOLUME_LARGE = float(os.getenv("OPT_VOLUME_LARGE", "5000"))
OPT_IV_HIGH = float(os.getenv("OPT_IV_HIGH", "80"))


# ------------- Error Handling -------------
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 Error: {request.url}")
    return jsonify({'error': 'Resource not found', 'path': request.url}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Error: {error}\n{traceback.format_exc()}")
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    rid = getattr(g, 'request_id', uuid.uuid4().hex)
    event = {
        "ts": _now_iso(),
        "level": "error",
        "event": "http_error",
        "request_id": rid,
        "method": request.method,
        "path": request.path,
        "status": 500,
        "latency_ms": None,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "stacktrace": traceback.format_exc(),
    }
    try:
        request_logger.error(json.dumps(event, ensure_ascii=False))
    except Exception:
        logger.error(f"Unhandled Exception: {e}\n{traceback.format_exc()}")
    return jsonify({'error': str(e)}), 500


# ------------- Static Data / Helpers -------------
# Sector mapping for major US stocks (S&P 500 + popular stocks)
SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech', 'ORCL': 'Tech',
    'CRM': 'Tech', 'AMD': 'Tech', 'ADBE': 'Tech', 'CSCO': 'Tech', 'INTC': 'Tech',
    'IBM': 'Tech', 'MU': 'Tech', 'QCOM': 'Tech', 'TXN': 'Tech', 'NOW': 'Tech',
    'AMAT': 'Tech', 'LRCX': 'Tech', 'KLAC': 'Tech', 'SNPS': 'Tech', 'CDNS': 'Tech',
    'ADI': 'Tech', 'MRVL': 'Tech', 'FTNT': 'Tech', 'PANW': 'Tech', 'CRWD': 'Tech',
    'SNOW': 'Tech', 'DDOG': 'Tech', 'ZS': 'Tech', 'NET': 'Tech', 'PLTR': 'Tech',
    'DELL': 'Tech', 'HPQ': 'Tech', 'HPE': 'Tech', 'KEYS': 'Tech', 'SWKS': 'Tech',
    'BRK-B': 'Fin', 'JPM': 'Fin', 'V': 'Fin', 'MA': 'Fin', 'BAC': 'Fin',
    'WFC': 'Fin', 'GS': 'Fin', 'MS': 'Fin', 'SPGI': 'Fin', 'AXP': 'Fin',
    'C': 'Fin', 'BLK': 'Fin', 'SCHW': 'Fin', 'CME': 'Fin', 'CB': 'Fin',
    'PGR': 'Fin', 'MMC': 'Fin', 'AON': 'Fin', 'ICE': 'Fin', 'MCO': 'Fin',
    'USB': 'Fin', 'PNC': 'Fin', 'TFC': 'Fin', 'AIG': 'Fin', 'MET': 'Fin',
    'PRU': 'Fin', 'ALL': 'Fin', 'TRV': 'Fin', 'COIN': 'Fin', 'HOOD': 'Fin',
    'LLY': 'Health', 'UNH': 'Health', 'JNJ': 'Health', 'ABBV': 'Health', 'MRK': 'Health',
    'PFE': 'Health', 'TMO': 'Health', 'ABT': 'Health', 'DHR': 'Health', 'BMY': 'Health',
    'AMGN': 'Health', 'GILD': 'Health', 'VRTX': 'Health', 'ISRG': 'Health', 'MDT': 'Health',
    'SYK': 'Health', 'BSX': 'Health', 'REGN': 'Health', 'ZTS': 'Health', 'ELV': 'Health',
    'CI': 'Health', 'HUM': 'Health', 'CVS': 'Health', 'MCK': 'Health', 'CAH': 'Health',
    'GEHC': 'Health', 'DXCM': 'Health', 'IQV': 'Health', 'BIIB': 'Health', 'MRNA': 'Health',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
    'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy', 'WMB': 'Energy',
    'DVN': 'Energy', 'HES': 'Energy', 'HAL': 'Energy', 'BKR': 'Energy', 'KMI': 'Energy',
    'FANG': 'Energy', 'PXD': 'Energy', 'TRGP': 'Energy', 'OKE': 'Energy', 'ET': 'Energy',
    'AMZN': 'Cons', 'TSLA': 'Cons', 'HD': 'Cons', 'MCD': 'Cons', 'NKE': 'Cons',
    'LOW': 'Cons', 'SBUX': 'Cons', 'TJX': 'Cons', 'BKNG': 'Cons', 'CMG': 'Cons',
    'ORLY': 'Cons', 'AZO': 'Cons', 'ROST': 'Cons', 'DHI': 'Cons', 'LEN': 'Cons',
    'GM': 'Cons', 'F': 'Cons', 'MAR': 'Cons', 'HLT': 'Cons', 'YUM': 'Cons',
    'DG': 'Cons', 'DLTR': 'Cons', 'BBY': 'Cons', 'ULTA': 'Cons', 'POOL': 'Cons',
    'LULU': 'Cons',
    'WMT': 'Staple', 'PG': 'Staple', 'COST': 'Staple', 'KO': 'Staple', 'PEP': 'Staple',
    'PM': 'Staple', 'MDLZ': 'Staple', 'MO': 'Staple', 'CL': 'Staple', 'KMB': 'Staple',
    'GIS': 'Staple', 'K': 'Staple', 'HSY': 'Staple', 'SYY': 'Staple', 'STZ': 'Staple',
    'KHC': 'Staple', 'KR': 'Staple', 'EL': 'Staple', 'CHD': 'Staple', 'CLX': 'Staple',
    'KDP': 'Staple', 'TAP': 'Staple', 'ADM': 'Staple', 'BG': 'Staple', 'MNST': 'Staple',
    'CAT': 'Indust', 'GE': 'Indust', 'RTX': 'Indust', 'HON': 'Indust', 'UNP': 'Indust',
    'BA': 'Indust', 'DE': 'Indust', 'LMT': 'Indust', 'UPS': 'Indust', 'MMM': 'Indust',
    'GD': 'Indust', 'NOC': 'Indust', 'CSX': 'Indust', 'NSC': 'Indust', 'WM': 'Indust',
    'EMR': 'Indust', 'ETN': 'Indust', 'ITW': 'Indust', 'PH': 'Indust', 'ROK': 'Indust',
    'FDX': 'Indust', 'CARR': 'Indust', 'TT': 'Indust', 'PCAR': 'Indust', 'FAST': 'Indust',
    'LIN': 'Mater', 'APD': 'Mater', 'SHW': 'Mater', 'FCX': 'Mater', 'ECL': 'Mater',
    'NEM': 'Mater', 'NUE': 'Mater', 'DOW': 'Mater', 'DD': 'Mater', 'VMC': 'Mater',
    'CTVA': 'Mater', 'PPG': 'Mater', 'MLM': 'Mater', 'IP': 'Mater', 'PKG': 'Mater',
    'ALB': 'Mater', 'GOLD': 'Mater', 'FMC': 'Mater', 'CF': 'Mater', 'MOS': 'Mater',
    'NEE': 'Util', 'SO': 'Util', 'DUK': 'Util', 'CEG': 'Util', 'SRE': 'Util',
    'AEP': 'Util', 'D': 'Util', 'PCG': 'Util', 'EXC': 'Util', 'XEL': 'Util',
    'ED': 'Util', 'WEC': 'Util', 'ES': 'Util', 'AWK': 'Util', 'DTE': 'Util',
    'PLD': 'REIT', 'AMT': 'REIT', 'EQIX': 'REIT', 'SPG': 'REIT', 'PSA': 'REIT',
    'O': 'REIT', 'WELL': 'REIT', 'DLR': 'REIT', 'CCI': 'REIT', 'AVB': 'REIT',
    'CBRE': 'REIT', 'SBAC': 'REIT', 'WY': 'REIT', 'EQR': 'REIT', 'VTR': 'REIT',
    'META': 'Comm', 'GOOGL': 'Comm', 'GOOG': 'Comm', 'NFLX': 'Comm', 'DIS': 'Comm',
    'T': 'Comm', 'VZ': 'Comm', 'CMCSA': 'Comm', 'TMUS': 'Comm', 'CHTR': 'Comm',
    'EA': 'Comm', 'TTWO': 'Comm', 'RBLX': 'Comm', 'PARA': 'Comm', 'WBD': 'Comm',
    'MTCH': 'Comm', 'LYV': 'Comm', 'OMC': 'Comm', 'IPG': 'Comm', 'FOXA': 'Comm',
    'EPAM': 'Tech', 'ALGN': 'Health',
}

SECTOR_CACHE_FILE = US_DIR / 'sector_cache.json'
_sector_cache: Dict[str, str] = {}


def _load_sector_cache() -> Dict[str, str]:
    try:
        if SECTOR_CACHE_FILE.exists():
            with open(SECTOR_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_sector_cache(cache: Dict[str, str]) -> None:
    try:
        SECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SECTOR_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Error saving sector cache: {e}")


_sector_cache = _load_sector_cache()

# Validate essential files once at startup (creates mocks if missing)
ensure_contracts([
    "smart_money_current",
    "smart_money_csv",
    "etf_flows",
    "etf_flow_analysis",
    "macro",
    "macro_en",
    "macro_gpt",
    "macro_gpt_en",
    "heatmap",
    "options",
    "ai_summaries",
    "calendar",
    "us_volume",
    "us_13f",
    "us_stocks",
    "us_prices",
])


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _cache_key(ticker: str, lang: str) -> str:
    return f"{ticker.upper()}::{lang}"


def _get_lock(key: str) -> threading.Lock:
    if key not in AI_SUMMARY_LOCKS:
        AI_SUMMARY_LOCKS[key] = threading.Lock()
    return AI_SUMMARY_LOCKS[key]


def _validate_ticker(ticker: str) -> bool:
    return bool(TICKER_RE.match(ticker.strip().upper()))


def _sanitize_summary(text: str, max_chars: int) -> Tuple[str, bool, int]:
    """Escape HTML and trim overly long content to keep UI safe."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    original_len = len(text)
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
        truncated = True
    # Escape any HTML to prevent XSS; frontend can still do light markdown replacement safely
    safe_text = html.escape(text)
    return safe_text, truncated, original_len


def _generate_ai_summary(ticker: str, lang: str) -> Dict[str, Any]:
    """Load AI summary from stored file (placeholder for real generation)."""
    ensure_contracts(["ai_summaries"])
    path = US_DIR / 'ai_summaries.json'
    summaries = load_json_file(path, {})
    entry = summaries.get(ticker.upper())
    if not entry:
        return {}
    raw_summary = entry.get('summary') or entry.get('ai_summary') or ""
    safe_summary, truncated, orig_len = _sanitize_summary(raw_summary, AI_SUMMARY_MAX_CHARS)
    updated = entry.get('updated') or entry.get('timestamp') or _now_iso()
    return {
        "summary": safe_summary,
        "updated": updated,
        "truncated": truncated,
        "original_length": orig_len,
        "model": entry.get("model") or entry.get("provider", ""),
    }


def _log_perf_component(component: str, started: float):
    try:
        event = {
            "ts": _now_iso(),
            "level": "info",
            "event": "perf_component",
            "component": component,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "request_id": getattr(g, "request_id", ""),
        }
        request_logger.info(json.dumps(event, ensure_ascii=False))
    except Exception:
        pass


@app.before_request
def _before_request():
    rid = request.headers.get('X-Request-Id') or uuid.uuid4().hex
    g.request_id = rid
    g._start_time = time.perf_counter()
    g._req_cache = {}
    g._perf_stats = {"cache_hits": 0, "cache_misses": 0, "yf_calls": 0, "yf_batches": 0, "parallel_tasks": 0}


@app.after_request
def _after_request(response):
    try:
        rid = getattr(g, 'request_id', None) or uuid.uuid4().hex
        response.headers['X-Request-Id'] = rid
        start = getattr(g, '_start_time', None)
        latency_ms = None
        if start is not None:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
        event = {
            "ts": _now_iso(),
            "level": "info",
            "event": "http_request",
            "request_id": rid,
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }
        request_logger.info(json.dumps(event, ensure_ascii=False))
    except Exception:
        # Do not block response on logging failures
        pass
    try:
        stats = get_perf_stats()
        perf_event = {
            "ts": _now_iso(),
            "level": "info",
            "event": "perf_summary",
            "request_id": rid,
            "path": request.path,
            "yfinance_calls": stats.get("yf_calls", 0),
            "yfinance_batched": stats.get("yf_batches", 0),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_misses": stats.get("cache_misses", 0),
            "parallel_tasks": stats.get("parallel_tasks", 0),
        }
        request_logger.info(json.dumps(perf_event, ensure_ascii=False))
    except Exception:
        pass
    return response


def get_sector(ticker: str) -> str:
    ticker = str(ticker).upper()
    if ticker in SECTOR_MAP:
        return SECTOR_MAP[ticker]
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    cache_key = ("sector_lookup", ticker)
    def loader():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            get_perf_stats()["yf_calls"] += 1
            sector = info.get('sector', '')
            sector_short_map = {
                'Technology': 'Tech', 'Information Technology': 'Tech',
                'Healthcare': 'Health', 'Health Care': 'Health',
                'Financials': 'Fin', 'Financial Services': 'Fin',
                'Consumer Discretionary': 'Cons', 'Consumer Cyclical': 'Cons',
                'Consumer Staples': 'Staple', 'Consumer Defensive': 'Staple',
                'Energy': 'Energy', 'Industrials': 'Indust',
                'Materials': 'Mater', 'Basic Materials': 'Mater',
                'Utilities': 'Util', 'Real Estate': 'REIT',
                'Communication Services': 'Comm'
            }
            short_sector = sector_short_map.get(sector, sector[:5] if sector else '-')
            _sector_cache[ticker] = short_sector
            _save_sector_cache(_sector_cache)
            return short_sector
        except Exception:
            return '-'

    return cache_get_or_set(cache_key, loader)


def grade_from_score(score: float) -> str:
    if score >= 85:
        return 'A+'
    if score >= 75:
        return 'A'
    if score >= 65:
        return 'B'
    if score >= 55:
        return 'C'
    return 'D'


def safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        if isinstance(val, float) and math.isnan(val):
            return default
        return float(val)
    except Exception:
        return default


def kr_to_yf_ticker(ticker: str) -> str:
    t = str(ticker).zfill(6)
    return f"{t}.KS"


def fetch_price_map(tickers: List[str]) -> Dict[str, float]:
    """Fetch latest close prices for list of tickers."""
    if not tickers:
        return {}
    cache_key = ("price_map", tuple(sorted(tickers)))
    def loader():
        started = time.perf_counter()
        try:
            df = yf.download(tickers, period='5d', progress=False, threads=False, group_by='ticker')
            get_perf_stats()["yf_calls"] += 1
            get_perf_stats()["yf_batches"] += 1
            if df.empty:
                return {}

            closes = df['Close']
            prices: Dict[str, float] = {}
            if isinstance(closes, pd.Series):
                series = closes.dropna()
                if not series.empty:
                    prices[tickers[0]] = round(float(series.iloc[-1]), 2)
            else:
                for t in tickers:
                    if t in closes.columns:
                        series = closes[t].dropna()
                        if not series.empty:
                            prices[t] = round(float(series.iloc[-1]), 2)
            return prices
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")
            return {}
        finally:
            _log_perf_component("price_map", started)
    return cache_get_or_set(cache_key, loader)


def load_json_file(path: str, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


# Sample fallback data for KR tab
SAMPLE_KR_HOLDINGS = [
    {'ticker': '005930', 'name': 'Samsung Elec', 'price': 70000, 'recommendation_price': 68000,
     'return_pct': 2.94, 'score': 78.5, 'grade': 'A', 'wave': 'Uptrend', 'sd_stage': 'Accum', 'inst_trend': 'Buy', 'ytd': 12.3},
    {'ticker': '000660', 'name': 'SK Hynix', 'price': 132000, 'recommendation_price': 125000,
     'return_pct': 5.6, 'score': 74.2, 'grade': 'A-', 'wave': 'Uptrend', 'sd_stage': 'Accum', 'inst_trend': 'Buy', 'ytd': 18.1},
    {'ticker': '035420', 'name': 'NAVER', 'price': 180000, 'recommendation_price': 170000,
     'return_pct': 5.9, 'score': 69.0, 'grade': 'B+', 'wave': 'Base', 'sd_stage': 'Neutral', 'inst_trend': 'Hold', 'ytd': 6.8},
]

SAMPLE_STYLE_BOX = {
    'large_value': 32, 'large_core': 24, 'large_growth': 18,
    'mid_value': 8, 'mid_core': 8, 'mid_growth': 5,
    'small_value': 3, 'small_core': 1, 'small_growth': 1
}


# ------------- Routes -------------
@app.route('/')
def index():
    return render_template('index.html')


# ----------- KR API -----------
@app.route('/health')
def health():
    return jsonify({"ok": True, "service": "gyuant-app", "ts": _now_iso()}), 200, {"Cache-Control": "no-store"}


# Status endpoint is placed near KR APIs for minimal intrusion
@app.route('/status')
def status():
    try:
        from utils.pipeline_utils import resolve_paths

        paths = resolve_paths()
        data_dir = paths["data_dir"]
        run_state_path = paths["run_state"]
        stale_threshold_sec = STALE_THRESHOLD_MIN * 60

        def file_info(p: Path) -> Dict[str, any]:
            if not p.exists():
                return {"path": str(p), "exists": False, "mtime": None, "age_sec": None, "size_bytes": None}
            stat = p.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
            age_sec = (datetime.now(timezone.utc) - mtime).total_seconds()
            return {
                "path": str(p),
                "exists": True,
                "mtime": mtime.isoformat(),
                "age_sec": age_sec,
                "size_bytes": stat.st_size,
            }

        modules = {
            "indices": [data_dir / "us_daily_prices.csv"],
            "smart_money": [data_dir / "smart_money_current.json", data_dir / "smart_money_picks_v2.csv"],
            "etf": [data_dir / "us_etf_flows.csv", data_dir / "etf_flow_analysis.json"],
            "macro": [data_dir / "macro_analysis.json", data_dir / "macro_analysis_en.json"],
            "options": [data_dir / "options_flow.json"],
        }

        modules_status = {}
        overall_ok = True

        for name, files in modules.items():
            infos = [file_info(p) for p in files]
            problems = []
            stale = False
            for info in infos:
                if not info["exists"]:
                    problems.append({"reason": "MISSING_FILE", "file": info["path"]})
                elif info["age_sec"] is not None and info["age_sec"] > stale_threshold_sec:
                    stale = True
            ok = not problems and not stale  # Policy: ok only if all exist and not stale
            if not ok:
                overall_ok = False
            modules_status[name] = {
                "ok": ok,
                "stale": stale,
                "updated_at": max([i["mtime"] for i in infos if i["mtime"]], default=None),
                "files": infos,
                "problems": problems,
            }

        run_state = {"last_success_at": None, "last_failure_at": None, "error_summary": None}
        try:
            if run_state_path.exists():
                run_state.update(json.loads(run_state_path.read_text(encoding="utf-8")))
        except Exception:
            run_state = {"last_success_at": None, "last_failure_at": None, "error_summary": "run_state read failed"}

        payload = {
            "ok": overall_ok,
            "ts": _now_iso(),
            "data_dir": str(data_dir),
            "stale_threshold_min": STALE_THRESHOLD_MIN,
            "last_run": run_state,
            "modules": modules_status,
        }
        return jsonify(payload)
    except Exception as e:
        event = {
            "ts": _now_iso(),
            "level": "error",
            "event": "status_error",
            "request_id": getattr(g, 'request_id', ''),
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
        request_logger.error(json.dumps(event, ensure_ascii=False))
        return jsonify({"ok": False, "error": str(e), "ts": _now_iso()}), 200


@app.route('/api/kr/recommendations')
def get_kr_recommendations():
    try:
        csv_path = BASE_DIR / 'recommendation_history.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dates = sorted(df['recommendation_date'].unique().tolist(), reverse=True)
            return jsonify({'dates': dates, 'data': df.to_dict(orient='records')})
        # Fallback
        today = datetime.utcnow().strftime('%Y-%m-%d')
        return jsonify({
            'dates': [today],
            'data': [{'ticker': h['ticker'], 'name': h['name'], 'recommendation_date': today,
                      'score': h['score'], 'investment_grade': h['grade']} for h in SAMPLE_KR_HOLDINGS]
        })
    except Exception as e:
        logger.error(f"KR recommendations error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/performance')
def get_kr_performance():
    try:
        csv_path = BASE_DIR / 'performance_report.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            summary = {
                'total_count': len(df),
                'avg_return': float(df['return'].mean()),
                'win_rate': float((df['return'] > 0).mean() * 100),
                'top_performers': df.sort_values('return', ascending=False).head(5).to_dict(orient='records')
            }
            return jsonify({'summary': summary, 'data': df.to_dict(orient='records')})
        return jsonify({'summary': {}, 'data': []})
    except Exception as e:
        logger.error(f"KR performance error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/market-status')
def get_kr_market_status():
    try:
        proxy_tickers = [('^KS11', 'KOSPI'), ('^KQ11', 'KOSDAQ')]
        for yf_ticker, name in proxy_tickers:
            try:
                hist = yf.Ticker(yf_ticker).history(period='1y')
                if hist.empty or len(hist) < 200:
                    continue
                hist['MA50'] = hist['Close'].rolling(50).mean()
                hist['MA200'] = hist['Close'].rolling(200).mean()
                last = hist.iloc[-1]
                status = 'BULL' if last['Close'] > last['MA200'] else 'NEUTRAL'
                return jsonify({
                    'status': status,
                    'score': round((last['Close'] / last['MA200'] - 1) * 100, 2),
                    'current_price': round(float(last['Close']), 2),
                    'ma200': round(float(last['MA200']), 2),
                    'date': last.name.strftime('%Y-%m-%d'),
                    'symbol': yf_ticker,
                    'name': name
                })
            except Exception:
                continue
        return jsonify({'status': 'UNKNOWN', 'reason': 'No data'}), 404
    except Exception as e:
        logger.error(f"KR market status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio')
def get_portfolio_data():
    """KR portfolio summary (fallback data)."""
    try:
        indices = []
        for ticker, name in [('^KS11', 'KOSPI'), ('^KQ11', 'KOSDAQ'), ('KRW=X', 'USD/KRW')]:
            try:
                hist = yf.Ticker(ticker).history(period='5d')
                if not hist.empty and len(hist) >= 2:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2])
                    change = current - prev
                    change_pct = (change / prev) * 100 if prev else 0
                    indices.append({
                        'name': name,
                        'price': f"{current:,.2f}",
                        'change': f"{change:+,.2f}",
                        'change_pct': round(change_pct, 2),
                        'color': 'red' if change < 0 else 'blue'
                    })
            except Exception:
                continue

        data = {
            'key_stats': {'qtd_return': 'N/A', 'ytd_return': 'N/A', 'one_year_return': 'N/A'},
            'market_indices': indices,
            'holdings_distribution': [],
            'top_holdings': SAMPLE_KR_HOLDINGS,
            'style_box': SAMPLE_STYLE_BOX,
            'performance': {},
            'latest_date': datetime.utcnow().strftime('%Y-%m-%d')
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"KR portfolio error: {e}")
        return jsonify({'error': str(e)}), 500


# ----------- US API -----------
@app.route('/api/us/portfolio')
def get_us_portfolio_data():
    try:
        ensure_contracts(["us_prices"])
        market_indices = []
        indices_map = {
            '^DJI': 'Dow Jones', '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000', '^VIX': 'VIX', 'GC=F': 'Gold',
            'CL=F': 'Crude Oil', 'BTC-USD': 'Bitcoin', '^TNX': '10Y Treasury',
            'DX-Y.NYB': 'Dollar Index', 'KRW=X': 'USD/KRW'
        }

        for ticker, name in indices_map.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                if not hist.empty and len(hist) >= 2:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2])
                    change = current - prev
                    change_pct = (change / prev) * 100 if prev else 0
                    market_indices.append({
                        'name': name,
                        'price': f"{current:,.2f}",
                        'change': f"{change:+,.2f}",
                        'change_pct': round(change_pct, 2),
                        'color': 'green' if change >= 0 else 'red'
                    })
            except Exception as e:
                logger.debug(f"Index fetch failed {ticker}: {e}")

        return jsonify({
            'market_indices': market_indices,
            'top_holdings': [],
            'style_box': {}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def build_smart_money_from_analysis():
    vol_path = US_DIR / 'us_volume_analysis.csv'
    inst_path = US_DIR / 'us_13f_holdings.csv'
    info_path = US_DIR / 'us_stocks_list.csv'

    if not os.path.exists(vol_path):
        return [], {'total_analyzed': 0, 'avg_score': 0}

    vol_df = pd.read_csv(vol_path)
    vol_df['supply_demand_score'] = pd.to_numeric(vol_df.get('supply_demand_score', 0), errors='coerce').fillna(0)

    if os.path.exists(inst_path):
        inst_df = pd.read_csv(inst_path)
        vol_df = vol_df.merge(inst_df[['ticker', 'institutional_score']], on='ticker', how='left')
    else:
        vol_df['institutional_score'] = 50

    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path)
        vol_df = vol_df.merge(info_df[['ticker', 'name', 'sector']], on='ticker', how='left')

    vol_df['institutional_score'] = pd.to_numeric(vol_df['institutional_score'], errors='coerce').fillna(50)
    vol_df['composite_score'] = (vol_df['supply_demand_score'] * 0.6 + vol_df['institutional_score'] * 0.4).clip(0, 100)
    vol_df['grade'] = vol_df['composite_score'].apply(grade_from_score)

    top_df = vol_df.sort_values('composite_score', ascending=False).head(20)
    prices = fetch_price_map(top_df['ticker'].tolist())

    picks = []
    for _, row in top_df.iterrows():
        ticker = row['ticker']
        current_price = prices.get(ticker, 0)
        score = round(row['composite_score'], 1)
        picks.append({
            'ticker': ticker,
            'name': row.get('name', ticker),
            'sector': row.get('sector', get_sector(ticker)),
            'final_score': score,
            'composite_score': score,
            'current_price': current_price,
            'price_at_rec': current_price,
            'change_since_rec': 0,
            'target_upside': max(0, round((score - 50) / 2, 1)),
            'grade': row.get('grade', grade_from_score(score)),
            'ai_recommendation': 'Buy' if score >= 70 else 'Watch'
        })

    summary = {
        'total_analyzed': len(vol_df),
        'avg_score': round(top_df['composite_score'].mean(), 1) if not top_df.empty else 0
    }
    return picks, summary


@app.route('/api/us/smart-money')
def get_us_smart_money():
    try:
        ensure_contracts(["smart_money_current", "smart_money_csv", "us_volume", "us_13f", "us_stocks"])
        # Prefer tracked snapshot with performance
        current_file = US_DIR / 'smart_money_current.json'
        if current_file.exists():
            snapshot = load_json_file(current_file, {})
            picks = snapshot.get('picks', [])
            return jsonify({
                'analysis_date': snapshot.get('analysis_date', ''),
                'analysis_timestamp': snapshot.get('analysis_timestamp', ''),
                'top_picks': picks,
                'summary': snapshot.get('summary', {'total_analyzed': len(picks)})
            })

        csv_path = US_DIR / 'smart_money_picks_v2.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            top_df = df.head(20).copy()
            price_map = fetch_price_map(top_df['ticker'].tolist())

            picks = []
            for _, row in top_df.iterrows():
                ticker = row['ticker']
                rec_price = safe_float(row.get('current_price'), 0)
                cur_price = price_map.get(ticker, rec_price)
                change_pct = ((cur_price / rec_price - 1) * 100) if rec_price else 0
                score = safe_float(row.get('smart_money_score', row.get('composite_score', 0)))
                picks.append({
                    'ticker': ticker,
                    'name': row.get('name', ticker),
                    'sector': get_sector(ticker),
                    'final_score': score,
                    'composite_score': score,
                    'current_price': cur_price,
                    'price_at_rec': rec_price,
                    'change_since_rec': round(change_pct, 2),
                    'target_upside': safe_float(row.get('target_upside'), 0),
                    'grade': row.get('grade', grade_from_score(score)),
                    'ai_recommendation': row.get('ai_recommendation', 'Hold')
                })

            summary = {
                'total_analyzed': len(df),
                'avg_score': round(df['smart_money_score'].mean() if 'smart_money_score' in df.columns else score, 1)
            }
            return jsonify({'top_picks': picks, 'summary': summary})

        # Build from analysis outputs
        picks, summary = build_smart_money_from_analysis()
        if not picks:
            return jsonify({'error': 'No smart money data available'}), 404
        return jsonify({'top_picks': picks, 'summary': summary})
    except Exception as e:
        logger.error(f"Smart money error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/etf-flows')
def get_us_etf_flows():
    try:
        ensure_contracts(["etf_flows", "etf_flow_analysis"])
        csv_path = US_DIR / 'us_etf_flows.csv'
        if not csv_path.exists():
            return jsonify({'error': 'No ETF data'}), 404

        df = pd.read_csv(csv_path)
        df['flow_score'] = pd.to_numeric(df.get('flow_score', 0), errors='coerce').fillna(0)

        broad = df[df['category'].str.contains('Market', na=False)]
        market_sentiment_score = round(broad['flow_score'].mean(), 1) if not broad.empty else round(df['flow_score'].mean(), 1)

        top_inflows = df.nlargest(5, 'flow_score').to_dict(orient='records')
        top_outflows = df.nsmallest(5, 'flow_score').to_dict(orient='records')
        sector_flows = df[df['category'].str.contains('Sector', na=False)].to_dict(orient='records')

        ai_analysis = ""
        ai_path = US_DIR / 'etf_flow_analysis.json'
        if ai_path.exists():
            ai_json = load_json_file(ai_path, {})
            ai_analysis = ai_json.get('analysis') or ai_json.get('ai_analysis', '')

        return jsonify({
            'market_sentiment_score': market_sentiment_score,
            'sector_flows': sector_flows,
            'top_inflows': top_inflows,
            'top_outflows': top_outflows,
            'all_etfs': df.to_dict(orient='records'),
            'ai_analysis': ai_analysis
        })
    except Exception as e:
        logger.error(f"ETF flows error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/stock-chart/<ticker>')
def get_us_stock_chart(ticker):
    try:
        period = request.args.get('period', '1y')
        if period not in ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']:
            period = '1y'
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 404

        candles = [{
            'time': int(idx.timestamp()),
            'open': round(row['Open'], 2),
            'high': round(row['High'], 2),
            'low': round(row['Low'], 2),
            'close': round(row['Close'], 2)
        } for idx, row in hist.iterrows()]

        return jsonify({'ticker': ticker, 'period': period, 'candles': candles})
    except Exception as e:
        logger.error(f"US stock chart error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/history-dates')
def get_us_history_dates():
    try:
        history_dir = US_DIR / 'history'
        if not history_dir.exists():
            return jsonify({'dates': []})
        dates = []
        for f in history_dir.iterdir():
            if f.name.startswith('picks_') and f.name.endswith('.json'):
                dates.append(f.name[6:-5])
        dates.sort(reverse=True)
        return jsonify({'dates': dates, 'count': len(dates)})
    except Exception as e:
        logger.error(f"History dates error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/history/<date>')
def get_us_history_by_date(date):
    try:
        history_file = US_DIR / 'history' / f'picks_{date}.json'
        if not history_file.exists():
            return jsonify({'error': f'No analysis found for {date}'}), 404

        snapshot = load_json_file(history_file, {})
        picks = snapshot.get('picks', [])
        tickers = [p['ticker'] for p in picks]
        prices = fetch_price_map(tickers)

        picks_with_perf = []
        for pick in picks:
            ticker = pick['ticker']
            price_at_rec = safe_float(pick.get('price_at_analysis', pick.get('price_at_rec', 0)))
            cur_price = prices.get(ticker, price_at_rec)
            change_pct = ((cur_price / price_at_rec - 1) * 100) if price_at_rec else 0
            picks_with_perf.append({
                **pick,
                'sector': get_sector(ticker),
                'current_price': round(cur_price, 2),
                'price_at_rec': round(price_at_rec, 2),
                'change_since_rec': round(change_pct, 2)
            })

        avg_perf = np.nanmean([p['change_since_rec'] for p in picks_with_perf]) if picks_with_perf else 0

        return jsonify({
            'analysis_date': snapshot.get('analysis_date', date),
            'analysis_timestamp': snapshot.get('analysis_timestamp', ''),
            'top_picks': picks_with_perf,
            'summary': {'total': len(picks_with_perf), 'avg_performance': round(float(avg_perf), 2)}
        })
    except Exception as e:
        logger.error(f"History {date} error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/macro-analysis')
def get_us_macro_analysis():
    try:
        lang = request.args.get('lang', 'ko')
        model = request.args.get('model', 'gemini')
        ensure_contracts(["macro", "macro_en", "macro_gpt", "macro_gpt_en"])

        # Determine file path based on model/lang
        if model == 'gpt':
            analysis_path = US_DIR / ('macro_analysis_gpt_en.json' if lang == 'en' else 'macro_analysis_gpt.json')
            if not analysis_path.exists():
                analysis_path = US_DIR / ('macro_analysis_en.json' if lang == 'en' else 'macro_analysis.json')
        else:
            analysis_path = US_DIR / ('macro_analysis_en.json' if lang == 'en' else 'macro_analysis.json')
        if not analysis_path.exists():
            analysis_path = US_DIR / 'macro_analysis.json'

        cached = load_json_file(analysis_path, {})
        macro_indicators = cached.get('macro_indicators', {})
        ai_analysis = cached.get('ai_analysis', 'Run macro_analyzer.py to refresh analysis.')

        # Update a few live indicators
        live_tickers = {'VIX': '^VIX', 'SPY': 'SPY', 'BTC': 'BTC-USD', 'GOLD': 'GC=F', 'USD/KRW': 'KRW=X'}
        try:
            cache_key = ("macro_live", tuple(sorted(live_tickers.values())))

            def load_live():
                started = time.perf_counter()
                try:
                    df = yf.download(list(live_tickers.values()), period='5d', progress=False, threads=False)
                    get_perf_stats()["yf_calls"] += 1
                    get_perf_stats()["yf_batches"] += 1
                    return df
                finally:
                    _log_perf_component("macro_live", started)

            live = cache_get_or_set(cache_key, load_live)
            if not live.empty:
                closes = live['Close']
                for name, ticker in live_tickers.items():
                    try:
                        if isinstance(closes, pd.DataFrame) and ticker in closes.columns:
                            macro_indicators[name] = {'value': round(float(closes[ticker].dropna().iloc[-1]), 2)}
                        elif isinstance(closes, pd.Series):
                            macro_indicators[name] = {'value': round(float(closes.dropna().iloc[-1]), 2)}
                    except Exception:
                        continue
        except Exception:
            pass

        return jsonify({
            'macro_indicators': macro_indicators,
            'ai_analysis': ai_analysis,
            'timestamp': cached.get('timestamp', datetime.utcnow().isoformat())
        })
    except Exception as e:
        logger.error(f"Macro analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/sector-heatmap')
def get_us_sector_heatmap():
    try:
        ensure_contracts(["heatmap"])
        path = US_DIR / 'sector_heatmap.json'
        if path.exists():
            return jsonify(load_json_file(path, {}))
        from us_market.sector_heatmap import SectorHeatmapCollector
        return jsonify(SectorHeatmapCollector().get_sector_performance('1d'))
    except Exception as e:
        logger.error(f"Sector heatmap error: {e}")
        return jsonify({'error': str(e)}), 500


def _compute_top_movers(window: str, limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, str, List[str]]:
    path = US_DIR / 'us_daily_prices.csv'
    errors: List[str] = []
    rows_read = 0
    updated_at = None
    if not path.exists():
        return [], [], rows_read, updated_at, errors

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone()
        updated_at = mtime.isoformat()
    except Exception:
        updated_at = None

    try:
        df = pd.read_csv(path, usecols=["ticker", "date", "close"])
        rows_read = len(df)
        if df.empty:
            return [], [], rows_read, updated_at, errors
        df = df.dropna(subset=["ticker", "date", "close"])
        if df.empty:
            return [], [], rows_read, updated_at, errors
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return [], [], rows_read, updated_at, errors
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            return [], [], rows_read, updated_at, errors

        gainers: List[Dict[str, Any]] = []
        losers: List[Dict[str, Any]] = []

        def compute_changes(group: pd.DataFrame):
            g = group.sort_values("date")
            if window == "1d":
                if len(g) < 2:
                    return None
                prev = g.iloc[-2]
            else:  # 1w
                if len(g) < 6:
                    return None
                prev = g.iloc[-6]
            last = g.iloc[-1]
            prev_close = float(prev["close"])
            last_close = float(last["close"])
            if prev_close == 0:
                return None
            pct = ((last_close / prev_close) - 1) * 100
            abs_ch = last_close - prev_close
            return {
                "ticker": str(last["ticker"]).upper(),
                "last_close": round(last_close, 3),
                "prev_close": round(prev_close, 3),
                "pct_change": round(pct, 2),
                "abs_change": round(abs_ch, 3),
                "window": window,
                "last_date": last["date"].isoformat(),
                "prev_date": prev["date"].isoformat(),
            }

        for _, grp in df.groupby("ticker"):
            res = compute_changes(grp)
            if not res:
                continue
            if res["pct_change"] >= 0:
                gainers.append(res)
            else:
                losers.append(res)

        gainers.sort(key=lambda r: r.get("pct_change", 0), reverse=True)
        losers.sort(key=lambda r: r.get("pct_change", 0))
        return gainers[:limit], losers[:limit], rows_read, updated_at, errors
    except Exception as e:
        errors.append(str(e))
        return [], [], rows_read, updated_at, errors


def _load_sector_map() -> Dict[str, str]:
    candidates = [
        US_DIR / "sectors.csv",
        US_DIR / "us_sectors.csv",
        US_DIR / "stocks.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            tcol = cols.get("ticker") or cols.get("symbol")
            scol = cols.get("sector")
            if not tcol or not scol:
                continue
            mapping = {}
            for _, row in df.iterrows():
                t = str(row[tcol]).upper()
                s = str(row[scol]).strip()
                if t and s:
                    mapping[t] = s
            if mapping:
                return mapping
        except Exception:
            continue
    return {}


@app.route('/top-movers')
def top_movers():
    start = time.perf_counter()
    rid = getattr(g, "request_id", uuid.uuid4().hex)
    window = request.args.get("window", "1d")
    limit_param = request.args.get("limit", "20")
    try:
        limit = int(limit_param)
    except Exception:
        return jsonify({"error": "invalid limit"}), 400
    if window not in ("1d", "1w"):
        return jsonify({"error": "invalid window"}), 400
    if limit < 1 or limit > 200:
        return jsonify({"error": "limit out of range"}), 400

    gainers, losers, rows_read, updated_at, errors = _compute_top_movers(window, limit)
    payload = {
        "data": {"gainers": gainers, "losers": losers, "window": window},
        "meta": {"updated_at": updated_at, "source_files": [str(US_DIR / 'us_daily_prices.csv')], "stale": False},
        "errors": errors,
    }
    try:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        event = {
            "ts": _now_iso(),
            "level": "info",
            "event": "top_movers_api",
            "request_id": rid,
            "window": window,
            "limit": limit,
            "latency_ms": latency_ms,
            "rows_read": rows_read,
            "gainers": len(gainers),
            "losers": len(losers),
        }
        request_logger.info(json.dumps(event, ensure_ascii=False))
    except Exception:
        pass
    return jsonify(payload)


@app.route('/sectors/heatmap')
def sectors_heatmap():
    start = time.perf_counter()
    rid = getattr(g, "request_id", uuid.uuid4().hex)
    window = request.args.get("window", "1d")
    if window not in ("1d", "1w"):
        return jsonify({"error": "invalid window"}), 400
    sector_map = _load_sector_map()
    if not sector_map:
        payload = {
            "data": [],
            "meta": {"updated_at": None, "source_files": [str(US_DIR / 'us_daily_prices.csv')], "stale": False},
            "errors": ["NO_SECTOR_MAP"],
        }
        return jsonify(payload)

    gainers, losers, rows_read, updated_at, errors = _compute_top_movers(window, 99999)
    agg = defaultdict(list)
    for rec in gainers + losers:
        sec = sector_map.get(rec["ticker"])
        if not sec:
            continue
        agg[sec].append(rec)
    items = []
    for sec, recs in agg.items():
        pct_values = [r["pct_change"] for r in recs if r.get("pct_change") is not None]
        if not pct_values:
            continue
        avg = sum(pct_values) / len(pct_values)
        top_ticker = max(recs, key=lambda r: r.get("pct_change", 0)).get("ticker")
        items.append({"sector": sec, "pct_change": round(avg, 2), "count": len(recs), "top_ticker": top_ticker})
    items.sort(key=lambda r: r.get("pct_change", 0), reverse=True)
    payload = {
        "data": items,
        "meta": {
            "updated_at": updated_at,
            "source_files": [str(US_DIR / 'us_daily_prices.csv')],
            "stale": False,
            # Policy: updated_at uses price file mtime only; sector map may differ but assumed stable
        },
        "errors": errors,
    }
    try:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        event = {
            "ts": _now_iso(),
            "level": "info",
            "event": "sectors_heatmap_api",
            "request_id": rid,
            "window": window,
            "latency_ms": latency_ms,
            "items": len(items),
            "rows_read": rows_read,
        }
        request_logger.info(json.dumps(event, ensure_ascii=False))
    except Exception:
        pass
    return jsonify(payload)


@app.route('/sectors/<sector>/movers')
def sector_movers(sector):
    start = time.perf_counter()
    rid = getattr(g, "request_id", uuid.uuid4().hex)
    window = request.args.get("window", "1d")
    limit_param = request.args.get("limit", "20")
    try:
        limit = int(limit_param)
    except Exception:
        return jsonify({"error": "invalid limit"}), 400
    if window not in ("1d", "1w"):
        return jsonify({"error": "invalid window"}), 400
    if limit < 1 or limit > 200:
        return jsonify({"error": "limit out of range"}), 400
    sector_map = _load_sector_map()
    if not sector_map:
        payload = {
            "data": {"gainers": [], "losers": [], "sector": sector},
            "meta": {"updated_at": None, "source_files": [str(US_DIR / 'us_daily_prices.csv')], "stale": False},
            "errors": ["NO_SECTOR_MAP"],
        }
        return jsonify(payload)
    sector_upper = sector.upper()
    gainers, losers, rows_read, updated_at, errors = _compute_top_movers(window, 99999)

    def filter_sector(recs):
        filtered = []
        for r in recs:
            sec = sector_map.get(r["ticker"], "").upper()
            if sec == sector_upper:
                filtered.append(r)
        return filtered

    sg = filter_sector(gainers)
    sl = filter_sector(losers)
    payload = {
        "data": {"gainers": sg[:limit], "losers": sl[:limit], "sector": sector},
        "meta": {"updated_at": updated_at, "source_files": [str(US_DIR / 'us_daily_prices.csv')], "stale": False},
        "errors": errors,
    }
    try:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        event = {
            "ts": _now_iso(),
            "level": "info",
            "event": "sector_movers_api",
            "request_id": rid,
            "sector": sector,
            "window": window,
            "limit": limit,
            "latency_ms": latency_ms,
            "gainers": len(sg[:limit]),
            "losers": len(sl[:limit]),
            "rows_read": rows_read,
        }
        request_logger.info(json.dumps(event, ensure_ascii=False))
    except Exception:
        pass
    return jsonify(payload)


@app.route('/api/us/options-flow')
def get_us_options_flow():
    try:
        ensure_contracts(["options"])
        path = US_DIR / 'options_flow.json'
        cfg = {
            "premium_high_usd": OPT_PREMIUM_HIGH_USD,
            "premium_mid_usd": OPT_PREMIUM_MID_USD,
            "vol_oi_ratio_high": OPT_VOL_OI_RATIO_HIGH,
            "vol_oi_ratio_mid": OPT_VOL_OI_RATIO_MID,
            "volume_large": OPT_VOLUME_LARGE,
            "iv_high": OPT_IV_HIGH,
        }
        raw = load_options_raw(path)
        scored = score_options_flow(raw.get("options_flow") or [], path, cfg)
        # Attach scored data while keeping original shape keys if present
        payload = {
            "options_flow": scored,
            "summary": raw.get("summary", {}),
            "timestamp": raw.get("timestamp"),
        }
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Options flow error: {e}")
        return jsonify({'error': str(e)}), 500


def _load_scored_options(cfg: Dict[str, float]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Path]:
    path = US_DIR / 'options_flow.json'
    raw = load_options_raw(path)
    scored = score_options_flow(raw.get("options_flow") or [], path, cfg)
    return scored, raw, path


@app.route('/options/unusual')
def get_options_unusual():
    start = time.perf_counter()
    rid = getattr(g, "request_id", uuid.uuid4().hex)
    cfg = {
        "premium_high_usd": OPT_PREMIUM_HIGH_USD,
        "premium_mid_usd": OPT_PREMIUM_MID_USD,
        "vol_oi_ratio_high": OPT_VOL_OI_RATIO_HIGH,
        "vol_oi_ratio_mid": OPT_VOL_OI_RATIO_MID,
        "volume_large": OPT_VOLUME_LARGE,
        "iv_high": OPT_IV_HIGH,
    }
    try:
        ticker = request.args.get("ticker")
        side = request.args.get("side")
        expiry = request.args.get("expiry")
        sort_by = request.args.get("sort", "score")
        try:
            min_score = float(request.args.get("min_score", "0"))
        except Exception:
            return jsonify({"error": "invalid min_score"}), 400
        try:
            limit = int(request.args.get("limit", "100"))
        except Exception:
            return jsonify({"error": "invalid limit"}), 400
        if limit < 1 or limit > 500:
            return jsonify({"error": "limit out of range"}), 400
        if ticker and not _validate_ticker(ticker):
            return jsonify({"error": "invalid ticker"}), 400
        if side and side.lower() not in ("call", "put"):
            return jsonify({"error": "invalid side"}), 400

        scored, raw, source_path = _load_scored_options(cfg)
        items = []
        for rec in scored:
            if ticker and rec.get("ticker") != ticker.upper():
                continue
            if side and rec.get("side") != side.lower():
                continue
            if expiry and rec.get("expiry") != expiry:
                continue
            if rec.get("unusual_score", 0) < min_score:
                continue
            items.append(rec)

        if sort_by == "premium":
            items.sort(key=lambda r: (r.get("premium") or 0), reverse=True)
        else:
            items.sort(key=lambda r: (r.get("unusual_score") or 0), reverse=True)

        updated_at = None
        stale = False
        try:
            if source_path.exists():
                mtime = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc).astimezone()
                updated_at = mtime.isoformat()
        except Exception:
            pass

        payload = {
            "data": {"items": items[:limit], "count": len(items)},
            "meta": {
                "updated_at": updated_at,
                "stale": stale,
                "source_files": [str(source_path)],
                "filters": {
                    "ticker": ticker,
                    "side": side,
                    "expiry": expiry,
                    "min_score": min_score,
                    "limit": limit,
                    "sort": sort_by,
                },
            },
            "errors": [],
        }
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Unusual options API error: {e}")
        payload = {
            "data": {"items": [], "count": 0},
            "meta": {"updated_at": None, "stale": False, "source_files": [], "filters": {}},
            "errors": [str(e)],
        }
        return jsonify(payload), 200
    finally:
        try:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            event = {
                "ts": _now_iso(),
                "level": "info",
                "event": "options_unusual_api",
                "request_id": rid,
                "ticker": request.args.get("ticker"),
                "side": request.args.get("side"),
                "min_score": request.args.get("min_score"),
                "limit": request.args.get("limit"),
                "latency_ms": latency_ms,
                "items_returned": len(locals().get("items", []) or []),
            }
            request_logger.info(json.dumps(event, ensure_ascii=False))
        except Exception:
            pass


@app.route('/api/us/ai-summary/<ticker>')
def get_us_ai_summary(ticker):
    start = time.perf_counter()
    rid = getattr(g, 'request_id', uuid.uuid4().hex)
    lang = request.args.get('lang', 'ko')
    if lang not in LANG_WHITELIST:
        lang = 'ko'

    ticker_clean = ticker.strip().upper()
    if not _validate_ticker(ticker_clean):
        return jsonify({"error": {"code": "INVALID_TICKER", "message": "Ticker must be A-Z0-9.- and <=12 chars"}}), 400

    force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes')
    now = datetime.now(timezone.utc)
    key = _cache_key(ticker_clean, lang)

    # Cooldown for force regeneration
    if force and key in AI_LAST_REGEN:
        elapsed = (now - AI_LAST_REGEN[key]).total_seconds()
        if elapsed < AI_SUMMARY_REGEN_COOLDOWN_SEC:
            retry_after = int(AI_SUMMARY_REGEN_COOLDOWN_SEC - elapsed)
            return jsonify({"error": {"code": "REGEN_COOLDOWN", "message": "Regeneration cooldown"}, "retry_after_sec": retry_after}), 429

    ttl = max(1, AI_SUMMARY_TTL_SEC)
    payload = None
    cached_used = False

    cached_entry = AI_SUMMARY_CACHE.get(key)
    if cached_entry and cached_entry.get("expires_at") and cached_entry["expires_at"] > now and not force:
        payload = cached_entry["payload"]
        payload["meta"]["cached"] = True
        payload["meta"]["expires_in_sec"] = int((cached_entry["expires_at"] - now).total_seconds())
        payload["meta"]["request_id"] = rid
        cached_used = True
    else:
        lock = _get_lock(key)
        with lock:
            cached_entry = AI_SUMMARY_CACHE.get(key)
            if cached_entry and cached_entry.get("expires_at") and cached_entry["expires_at"] > now and not force:
                payload = cached_entry["payload"]
                payload["meta"]["cached"] = True
                payload["meta"]["expires_in_sec"] = int((cached_entry["expires_at"] - now).total_seconds())
                payload["meta"]["request_id"] = rid
                cached_used = True
            else:
                data = _generate_ai_summary(ticker_clean, lang)
                if not data:
                    return jsonify({'error': 'Not found'}), 404

                generated_at_iso = _now_iso()
                expires_at = now + timedelta(seconds=ttl)
                payload = {
                    "ticker": ticker_clean,
                    "lang": lang,
                    "summary": data["summary"],
                    "updated": data.get("updated", generated_at_iso),
                    "meta": {
                        "cached": False,
                        "generated_at": generated_at_iso,
                        "expires_in_sec": ttl,
                        "truncated": data.get("truncated", False),
                        "original_length": data.get("original_length", len(data["summary"])),
                        "request_id": rid,
                    }
                }
                AI_SUMMARY_CACHE[key] = {
                    "payload": payload,
                    "expires_at": expires_at,
                }
                if force:
                    AI_LAST_REGEN[key] = now

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    event = {
        "ts": _now_iso(),
        "level": "info",
        "event": "ai_summary",
        "request_id": rid,
        "ticker": ticker_clean,
        "lang": lang,
        "cached": cached_used,
        "force": force,
        "latency_ms": latency_ms,
        "output_len": len(payload["summary"]) if payload else 0,
        "truncated": payload["meta"]["truncated"] if payload else False,
    }
    try:
        request_logger.info(json.dumps(event, ensure_ascii=False))
    except Exception:
        pass

    return jsonify(payload)


@app.route('/api/stock/<ticker>')
def get_stock_detail(ticker):
    t = str(ticker).zfill(6)
    try:
        yf_ticker = kr_to_yf_ticker(t)
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period='1y')
        if hist.empty:
            return jsonify({'error': 'No data'}), 404
        hist = hist.reset_index()
        price_history = [{
            'time': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']).split(' ')[0],
            'open': safe_float(row['Open']),
            'high': safe_float(row['High']),
            'low': safe_float(row['Low']),
            'close': safe_float(row['Close']),
            'volume': int(row.get('Volume', 0))
        } for _, row in hist.iterrows()]

        return jsonify({
            'ticker': t,
            'price_history': price_history,
            'ai_report': ''
        })
    except Exception as e:
        logger.error(f"Stock detail error {ticker}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/run-analysis', methods=['POST'])
def run_analysis():
    """
    Trigger the US Market analysis pipeline in background.
    
    JSON body options:
    - quick: bool - Skip AI analysis (uses --quick flag)
    - smart: bool - Use smart_update.py for incremental updates based on file age
    - force: bool - Force full update even if files are fresh (only with smart mode)
    """
    try:
        # Get optional parameters
        data = request.get_json(force=True) if request.data else {}
        quick_mode = data.get('quick', False)
        smart_mode = data.get('smart', False)
        force_mode = data.get('force', False)
        
        def run_pipeline():
            try:
                if smart_mode:
                    # Use smart_update.py for incremental updates
                    logger.info("🔍 Starting Smart Update (incremental)...")
                    smart_script = US_DIR / 'smart_update.py'
                    if not smart_script.exists():
                        logger.warning("smart_update.py not found, falling back to full update")
                        smart_mode_active = False
                    else:
                        cmd = [sys.executable, str(smart_script)]
                        if force_mode:
                            cmd.append('--force')
                        smart_mode_active = True
                else:
                    smart_mode_active = False
                
                if not smart_mode_active:
                    # Use full update_all.py pipeline
                    logger.info("🚀 Starting US Market Analysis Pipeline...")
                    update_script = US_DIR / 'update_all.py'
                    cmd = [sys.executable, str(update_script)]
                    if quick_mode:
                        cmd.append('--quick')
                
                # Set UTF-8 encoding for Windows compatibility
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                # Run the pipeline
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    cwd=str(US_DIR)
                )
                
                if result.returncode == 0:
                    logger.info("✅ US Market Analysis Pipeline completed successfully")
                else:
                    logger.error(f"❌ Pipeline failed with code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Error: {result.stderr[-1000:]}")
                        
            except Exception as e:
                logger.error(f"❌ Background analysis failed: {e}")
                logger.error(traceback.format_exc())

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()
        
        mode_desc = 'smart incremental' if smart_mode else ('quick' if quick_mode else 'full')
        return jsonify({
            'status': 'started',
            'message': f'Analysis pipeline ({mode_desc}) started in background.',
            'mode': mode_desc,
            'quick_mode': quick_mode,
            'smart_mode': smart_mode,
            'phases': ['Data Collection', 'Basic Analysis', 'Screening', 'AI Analysis']
        })
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/realtime-prices', methods=['POST'])
def get_realtime_prices():
    try:
        data = request.get_json(force=True) if request.data else {}
        tickers = data.get('tickers', [])
        if not tickers:
            return jsonify({})

        yf_tickers = []
        ticker_map = {}
        for t in tickers:
            t_str = str(t)
            yf_t = kr_to_yf_ticker(t_str) if t_str.isdigit() and len(t_str) <= 6 else t_str
            yf_tickers.append(yf_t)
            ticker_map[yf_t] = t_str

        df = yf.download(yf_tickers, period='1d', interval='1m', progress=False, threads=True)
        if df.empty:
            return jsonify({})
        df = df.ffill()

        prices = {}
        if len(yf_tickers) == 1:
            row = df.iloc[-1]
            prices[tickers[0]] = {
                'current': safe_float(row['Close']),
                'open': safe_float(row['Open']),
                'high': safe_float(row['High']),
                'low': safe_float(row['Low']),
                'date': df.index[-1].strftime('%Y-%m-%d')
            }
        else:
            last_row = df.iloc[-1]
            for yf_t in yf_tickers:
                orig = ticker_map.get(yf_t)
                if orig and yf_t in df['Close'].columns:
                    prices[orig] = {
                        'current': safe_float(df['Close'][yf_t]),
                        'open': safe_float(df['Open'][yf_t]),
                        'high': safe_float(df['High'][yf_t]),
                        'low': safe_float(df['Low'][yf_t]),
                        'date': last_row.name.strftime('%Y-%m-%d') if hasattr(last_row, 'name') else datetime.utcnow().strftime('%Y-%m-%d')
                    }

        return jsonify(prices)
    except Exception as e:
        logger.error(f"Realtime prices error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/calendar')
def get_us_calendar():
    try:
        ensure_contracts(["calendar"])
        calendar_path = US_DIR / 'weekly_calendar.json'
        if calendar_path.exists():
            return jsonify(load_json_file(calendar_path, {}))
        return jsonify({'events': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/technical-indicators/<ticker>')
def get_technical_indicators(ticker):
    try:
        from ta.momentum import RSIIndicator
        from ta.trend import MACD
        from ta.volatility import BollingerBands

        hist = yf.Ticker(ticker).history(period='1y').reset_index()
        if hist.empty:
            return jsonify({'error': 'No data'}), 404

        close = hist['Close']

        rsi = RSIIndicator(close).rsi()
        macd_calc = MACD(close)
        bb = BollingerBands(close)

        def series_to_points(series):
            return [{'time': int(t.timestamp()), 'value': round(v, 2)} for t, v in zip(hist['Date'], series) if pd.notna(v)]

        supports = []
        resistances = []
        try:
            rolling_low = close.rolling(50).min().dropna()
            rolling_high = close.rolling(50).max().dropna()
            if not rolling_low.empty:
                supports.append(round(float(rolling_low.iloc[-1]), 2))
            if not rolling_high.empty:
                resistances.append(round(float(rolling_high.iloc[-1]), 2))
        except Exception:
            pass

        return jsonify({
            'ticker': ticker,
            'rsi': series_to_points(rsi),
            'macd': {
                'line': series_to_points(macd_calc.macd()),
                'signal': series_to_points(macd_calc.macd_signal()),
                'hist': series_to_points(macd_calc.macd_diff())
            },
            'bollinger': {
                'upper': series_to_points(bb.bollinger_hband()),
                'middle': series_to_points(bb.bollinger_mavg()),
                'lower': series_to_points(bb.bollinger_lband())
            },
            'support_resistance': {
                'support': supports,
                'resistance': resistances
            }
        })
    except Exception as e:
        logger.error(f"Technical indicator error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Auto-run smart update in background on server startup
    def run_smart_update_background():
        """Run smart_update.py in background thread on startup."""
        import time
        time.sleep(2)  # Wait for server to fully start
        try:
            smart_update_path = BASE_DIR / "us_market" / "smart_update.py"
            if smart_update_path.exists():
                logger.info("🔄 Starting background data freshness check...")
                result = subprocess.run(
                    [sys.executable, str(smart_update_path)],
                    cwd=str(BASE_DIR / "us_market"),
                    capture_output=True,
                    text=True,
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
                )
                if result.returncode == 0:
                    logger.info("✅ Background smart update completed")
                else:
                    logger.warning(f"⚠️ Smart update had issues: {result.stderr[:200] if result.stderr else 'unknown'}")
            else:
                logger.info("ℹ️ smart_update.py not found, skipping auto-update")
        except Exception as e:
            logger.error(f"❌ Background smart update failed: {e}")
    
    # Start background update thread (non-blocking)
    update_thread = threading.Thread(target=run_smart_update_background, daemon=True)
    update_thread.start()
    
    print('Flask Server Starting on port 5001...')
    app.run(port=5001, debug=True, use_reloader=False)
