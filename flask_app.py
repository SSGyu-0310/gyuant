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
from flask import Flask, jsonify, render_template, request, g

from utils.fmp_client import get_fmp_client
from utils.logger import setup_logger
from utils.pipeline_utils import ensure_contracts
from utils.perf_utils import cache_get_or_set, get_perf_stats, get_request_cache, perf_env_threads_enabled, perf_max_workers
from utils.symbols import map_symbols_to_fmp, to_fmp_symbol
from utils.options_unusual import score_options_flow, load_options_raw
from backtest.db_schema import get_connection
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
    summaries = _db_fetch_document("ai_summaries") or {}
    if not summaries:
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
    g._perf_stats = {
        "cache_hits": 0,
        "cache_misses": 0,
        "fmp_calls": 0,
        "fmp_batches": 0,
        "parallel_tasks": 0,
    }


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
            "fmp_calls": stats.get("fmp_calls", 0),
            "fmp_batches": stats.get("fmp_batches", 0),
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
            fmp = get_fmp_client()
            fmp_symbol = to_fmp_symbol(ticker)
            profile = fmp.profile(fmp_symbol) or {}
            get_perf_stats()["fmp_calls"] += 1
            sector = profile.get('sector', '') or profile.get('industry', '')
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


def fetch_price_map(tickers: List[str]) -> Dict[str, float]:
    """Fetch latest close prices for list of tickers."""
    if not tickers:
        return {}
    cache_key = ("price_map", tuple(sorted([str(t).upper() for t in tickers])))
    def loader():
        started = time.perf_counter()
        try:
            fmp = get_fmp_client()
            fmp_symbols, reverse_map = map_symbols_to_fmp(tickers)
            quotes = fmp.quote(fmp_symbols)
            get_perf_stats()["fmp_calls"] += 1
            get_perf_stats()["fmp_batches"] += 1
            if not quotes:
                return {}
            prices: Dict[str, float] = {}
            for item in quotes:
                symbol = item.get("symbol")
                if not symbol:
                    continue
                original = reverse_map.get(symbol, symbol)
                price = item.get("price")
                if price is None:
                    price = item.get("previousClose")
                if price is None:
                    price = item.get("open")
                if price is None:
                    continue
                prices[original] = round(float(price), 2)
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


def _get_db_connection():
    try:
        return get_connection()
    except Exception:
        return None


def _db_fetch_document(doc_type: str, lang: str = "na", model: str = "na") -> Dict[str, Any]:
    conn = _get_db_connection()
    if conn is None:
        return {}
    try:
        row = conn.execute(
            """
            SELECT payload_json
            FROM market_documents
            WHERE doc_type = ? AND lang = ? AND model = ?
            ORDER BY as_of_date DESC, updated_at DESC
            LIMIT 1
            """,
            (doc_type, lang, model),
        ).fetchone()
        if not row:
            return {}
        payload = row["payload_json"] if isinstance(row, dict) or hasattr(row, "__getitem__") else None
        if not payload:
            return {}
        return json.loads(payload)
    except Exception:
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _db_fetch_smart_money():
    conn = _get_db_connection()
    if conn is None:
        return None
    try:
        run = conn.execute(
            """
            SELECT run_id, analysis_date, analysis_timestamp, summary_json
            FROM market_smart_money_runs
            ORDER BY analysis_date DESC, created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not run:
            return None
        run_id = run["run_id"]
        picks = conn.execute(
            """
            SELECT *
            FROM market_smart_money_picks
            WHERE run_id = ?
            ORDER BY rank ASC
            """,
            (run_id,),
        ).fetchall()
        summary = {}
        summary_json = run["summary_json"]
        if summary_json:
            try:
                summary = json.loads(summary_json)
            except Exception:
                summary = {}
        return {
            "run": dict(run),
            "picks": [dict(p) for p in picks],
            "summary": summary,
        }
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _db_fetch_smart_money_by_date(date_str: str):
    conn = _get_db_connection()
    if conn is None:
        return None
    try:
        run = conn.execute(
            """
            SELECT run_id, analysis_date, analysis_timestamp, summary_json
            FROM market_smart_money_runs
            WHERE analysis_date = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (date_str,),
        ).fetchone()
        if not run:
            return None
        run_id = run["run_id"]
        picks = conn.execute(
            """
            SELECT *
            FROM market_smart_money_picks
            WHERE run_id = ?
            ORDER BY rank ASC
            """,
            (run_id,),
        ).fetchall()
        summary = {}
        summary_json = run["summary_json"]
        if summary_json:
            try:
                summary = json.loads(summary_json)
            except Exception:
                summary = {}
        return {
            "run": dict(run),
            "picks": [dict(p) for p in picks],
            "summary": summary,
        }
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _db_fetch_etf_flows():
    conn = _get_db_connection()
    if conn is None:
        return None
    try:
        latest = conn.execute(
            "SELECT MAX(as_of_date) AS as_of_date FROM market_etf_flows"
        ).fetchone()
        if not latest or not latest["as_of_date"]:
            return None
        as_of_date = latest["as_of_date"]
        rows = conn.execute(
            """
            SELECT *
            FROM market_etf_flows
            WHERE as_of_date = ?
            """,
            (as_of_date,),
        ).fetchall()
        return {"as_of_date": as_of_date, "rows": [dict(r) for r in rows]}
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ------------- Routes -------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({"ok": True, "service": "gyuant-app", "ts": _now_iso()}), 200, {"Cache-Control": "no-store"}

@app.route('/favicon.ico')
def favicon():
    return "", 204


# Status endpoint is placed near APIs for minimal intrusion
@app.route('/status')
def status():
    try:
        from utils.pipeline_utils import resolve_paths

        paths = resolve_paths()
        data_dir = paths["data_dir"]
        run_state_path = paths["run_state"]
        stale_threshold_sec = STALE_THRESHOLD_MIN * 60
        now = datetime.now(timezone.utc)

        def file_info(p: Path) -> Dict[str, any]:
            if not p.exists():
                return {"path": str(p), "exists": False, "mtime": None, "age_sec": None, "size_bytes": None}
            stat = p.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
            age_sec = (now - mtime).total_seconds()
            return {
                "path": str(p),
                "exists": True,
                "mtime": mtime.isoformat(),
                "age_sec": age_sec,
                "size_bytes": stat.st_size,
            }

        def _parse_db_time(value: str) -> Optional[datetime]:
            if not value:
                return None
            try:
                if len(value) == 10:
                    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                return None

        def db_info(label: str, updated_at: Optional[str]) -> Dict[str, any]:
            dt = _parse_db_time(updated_at or "")
            if not dt:
                return {"path": f"db:{label}", "exists": False, "mtime": None, "age_sec": None, "size_bytes": None}
            age_sec = (now - dt).total_seconds()
            return {
                "path": f"db:{label}",
                "exists": True,
                "mtime": dt.isoformat(),
                "age_sec": age_sec,
                "size_bytes": None,
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

        # Prefer DB timestamps if available
        db_conn = _get_db_connection()
        db_overrides: Dict[str, List[Dict[str, Any]]] = {}
        if db_conn is not None:
            try:
                row = db_conn.execute("SELECT MAX(date) AS max_date FROM market_prices_daily").fetchone()
                db_overrides["indices"] = [db_info("market_prices_daily", row["max_date"] if row else None)]
                row = db_conn.execute(
                    """
                    SELECT analysis_date, analysis_timestamp
                    FROM market_smart_money_runs
                    ORDER BY analysis_date DESC, created_at DESC
                    LIMIT 1
                    """
                ).fetchone()
                ts = row["analysis_timestamp"] if row else None
                db_overrides["smart_money"] = [db_info("market_smart_money_runs", ts or (row["analysis_date"] if row else None))]
                row = db_conn.execute("SELECT MAX(as_of_date) AS max_date FROM market_etf_flows").fetchone()
                db_overrides["etf"] = [db_info("market_etf_flows", row["max_date"] if row else None)]
                row = db_conn.execute(
                    """
                    SELECT as_of_date, updated_at
                    FROM market_documents
                    WHERE doc_type = 'macro_analysis' AND lang = 'ko' AND model = 'gemini'
                    ORDER BY as_of_date DESC, updated_at DESC
                    LIMIT 1
                    """
                ).fetchone()
                ts = row["updated_at"] if row else None
                db_overrides["macro"] = [db_info("market_documents:macro_analysis", ts or (row["as_of_date"] if row else None))]
            finally:
                try:
                    db_conn.close()
                except Exception:
                    pass

        for name, files in modules.items():
            infos = db_overrides.get(name) or [file_info(p) for p in files]
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


# ----------- US API -----------
@app.route('/api/us/portfolio')
def get_us_portfolio_data():
    try:
        ensure_contracts(["us_prices"])
        market_indices = []
        # FMP-native symbols (no Yahoo-style conversion needed)
        indices_map = {
            'DJI': 'Dow Jones',
            'GSPC': 'S&P 500',
            'NDX': 'NASDAQ',
            'RUT': 'Russell 2000',
            'VIX': 'VIX',
            'GCUSD': 'Gold',
            'CLUSD': 'Crude Oil',
            'BTCUSD': 'Bitcoin',
            'DXUSD': 'Dollar Index',
        }

        fmp = get_fmp_client()
        quotes = fmp.quote(list(indices_map.keys()))
        get_perf_stats()["fmp_calls"] += 1
        get_perf_stats()["fmp_batches"] += 1

        quote_map = {item.get("symbol"): item for item in quotes if isinstance(item, dict)}
        for symbol, name in indices_map.items():
            try:
                item = quote_map.get(symbol)
                if not item:
                    continue
                price = safe_float(item.get("price"), 0)
                prev_close = safe_float(item.get("previousClose"), 0)
                change = item.get("change")
                if change is None and prev_close:
                    change = price - prev_close
                change = safe_float(change, 0)
                change_pct = item.get("changesPercentage")
                if isinstance(change_pct, str):
                    change_pct = change_pct.replace("%", "").replace("(", "").replace(")", "")
                change_pct = safe_float(change_pct, 0)
                if change_pct == 0 and prev_close:
                    change_pct = (change / prev_close) * 100 if prev_close else 0
                market_indices.append({
                    'name': name,
                    'price': f"{price:,.2f}",
                    'change': f"{change:+,.2f}",
                    'change_pct': round(change_pct, 2),
                    'color': 'green' if change >= 0 else 'red'
                })
            except Exception as e:
                logger.debug(f"Index parse failed {symbol}: {e}")

        # Treasury rates (10Y)
        try:
            rates = fmp.treasury_rates()
            get_perf_stats()["fmp_calls"] += 1
            if rates:
                latest = rates[0]
                prev = rates[1] if len(rates) > 1 else latest
                rate_10y = safe_float(latest.get("year10"), 0)
                prev_10y = safe_float(prev.get("year10"), rate_10y)
                change = rate_10y - prev_10y
                change_pct = (change / prev_10y) * 100 if prev_10y else 0
                market_indices.append({
                    'name': '10Y Treasury',
                    'price': f"{rate_10y:,.2f}",
                    'change': f"{change:+,.2f}",
                    'change_pct': round(change_pct, 2),
                    'color': 'green' if change >= 0 else 'red'
                })
        except Exception as e:
            logger.debug(f"Treasury rates fetch failed: {e}")

        return jsonify({
            'market_indices': market_indices,
            'top_holdings': [],
            'style_box': {}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def build_smart_money_from_analysis():
    vol_path = US_DIR / 'us_volume_analysis.csv'
    info_path = US_DIR / 'us_stocks_list.csv'

    if not os.path.exists(vol_path):
        return [], {'total_analyzed': 0, 'avg_score': 0}

    vol_df = pd.read_csv(vol_path)
    vol_df['supply_demand_score'] = pd.to_numeric(vol_df.get('supply_demand_score', 0), errors='coerce').fillna(0)

    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path)
        vol_df = vol_df.merge(info_df[['ticker', 'name', 'sector']], on='ticker', how='left')

    vol_df['composite_score'] = vol_df['supply_demand_score'].clip(0, 100)
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
        db_payload = _db_fetch_smart_money()
        if db_payload:
            picks = []
            for row in db_payload["picks"]:
                ticker = row.get("ticker")
                score = safe_float(row.get("composite_score"), 0)
                picks.append({
                    "ticker": ticker,
                    "name": row.get("name", ticker),
                    "sector": row.get("sector") or get_sector(ticker),
                    "final_score": score,
                    "composite_score": score,
                    "current_price": safe_float(row.get("current_price"), 0),
                    "price_at_rec": safe_float(row.get("price_at_rec", row.get("current_price")), 0),
                    "change_since_rec": safe_float(row.get("change_since_rec"), 0),
                    "target_upside": safe_float(row.get("target_upside"), 0),
                    "grade": row.get("grade", grade_from_score(score)),
                    "ai_recommendation": row.get("recommendation", "Hold"),
                })
            run = db_payload["run"]
            summary = db_payload.get("summary") or {"total_analyzed": len(picks)}
            return jsonify({
                "analysis_date": run.get("analysis_date", ""),
                "analysis_timestamp": run.get("analysis_timestamp", ""),
                "top_picks": picks,
                "summary": summary,
            })

        ensure_contracts(["smart_money_current", "smart_money_csv", "us_volume", "us_stocks"])
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
        db_data = _db_fetch_etf_flows()
        if db_data and db_data.get("rows"):
            df = pd.DataFrame(db_data["rows"])
        else:
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
        ai_doc = _db_fetch_document("etf_flow_analysis")
        if ai_doc:
            ai_analysis = ai_doc.get("analysis") or ai_doc.get("ai_analysis", "")
        else:
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
        end_date = datetime.utcnow().date()
        period_days = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
        }
        from_date = None
        if period != 'max':
            from_date = (end_date - timedelta(days=period_days[period])).isoformat()
        to_date = end_date.isoformat()

        fmp = get_fmp_client()
        fmp_symbol = to_fmp_symbol(ticker)
        data = fmp.historical_price_full(fmp_symbol, from_date=from_date, to_date=to_date)
        get_perf_stats()["fmp_calls"] += 1
        hist = data.get("historical", []) if isinstance(data, dict) else []
        if not hist:
            return jsonify({'error': f'No data found for {ticker}'}), 404

        hist_sorted = sorted(hist, key=lambda x: x.get("date", ""))
        candles = []
        for row in hist_sorted:
            date_str = row.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            candles.append({
                'time': int(dt.timestamp()),
                'open': round(safe_float(row.get('open')), 2),
                'high': round(safe_float(row.get('high')), 2),
                'low': round(safe_float(row.get('low')), 2),
                'close': round(safe_float(row.get('close')), 2)
            })

        return jsonify({'ticker': ticker, 'period': period, 'candles': candles})
    except Exception as e:
        logger.error(f"US stock chart error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/history-dates')
def get_us_history_dates():
    try:
        conn = _get_db_connection()
        if conn is not None:
            rows = conn.execute(
                "SELECT DISTINCT analysis_date FROM market_smart_money_runs ORDER BY analysis_date DESC"
            ).fetchall()
            try:
                conn.close()
            except Exception:
                pass
            dates = [r["analysis_date"] for r in rows if r["analysis_date"]]
            return jsonify({'dates': dates, 'count': len(dates)})
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
        db_payload = _db_fetch_smart_money_by_date(date)
        if db_payload:
            picks = db_payload.get("picks", [])
            tickers = [p.get("ticker") for p in picks if p.get("ticker")]
            prices = fetch_price_map(tickers)
            picks_with_perf = []
            for pick in picks:
                ticker = pick.get("ticker")
                price_at_rec = safe_float(pick.get("price_at_rec"), 0)
                cur_price = prices.get(ticker, price_at_rec)
                change_pct = ((cur_price / price_at_rec - 1) * 100) if price_at_rec else 0
                picks_with_perf.append({
                    **pick,
                    "sector": pick.get("sector") or get_sector(ticker),
                    "current_price": round(cur_price, 2),
                    "price_at_rec": round(price_at_rec, 2),
                    "change_since_rec": round(change_pct, 2),
                })
            avg_perf = np.nanmean([p["change_since_rec"] for p in picks_with_perf]) if picks_with_perf else 0
            run = db_payload["run"]
            return jsonify({
                "analysis_date": run.get("analysis_date", date),
                "analysis_timestamp": run.get("analysis_timestamp", ""),
                "top_picks": picks_with_perf,
                "summary": {"total": len(picks_with_perf), "avg_performance": round(float(avg_perf), 2)},
            })
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
        doc_model = "gpt" if model == "gpt" else "gemini"
        doc_lang = "en" if lang == "en" else "ko"
        db_doc = _db_fetch_document("macro_analysis", lang=doc_lang, model=doc_model)
        if not db_doc and doc_model == "gpt":
            db_doc = _db_fetch_document("macro_analysis", lang=doc_lang, model="gemini")
        if not db_doc:
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

        cached = db_doc or load_json_file(analysis_path, {})
        macro_indicators = cached.get('macro_indicators', {})
        ai_analysis = cached.get('ai_analysis', 'Run macro_analyzer.py to refresh analysis.')

        # Update a few live indicators
        live_tickers = {'VIX': '^VIX', 'SPY': 'SPY', 'BTC': 'BTC-USD', 'GOLD': 'GC=F'}
        try:
            name_to_symbol = {name: to_fmp_symbol(sym) for name, sym in live_tickers.items()}
            cache_key = ("macro_live", tuple(sorted(name_to_symbol.values())))

            def load_live():
                started = time.perf_counter()
                try:
                    fmp = get_fmp_client()
                    quotes = fmp.quote(list(name_to_symbol.values()))
                    get_perf_stats()["fmp_calls"] += 1
                    get_perf_stats()["fmp_batches"] += 1
                    return quotes
                finally:
                    _log_perf_component("macro_live", started)

            live = cache_get_or_set(cache_key, load_live)
            if live:
                quote_map = {item.get("symbol"): item for item in live if isinstance(item, dict)}
                for name, symbol in name_to_symbol.items():
                    item = quote_map.get(symbol)
                    if not item:
                        continue
                    value = item.get("price")
                    if value is None:
                        value = item.get("previousClose")
                    if value is None:
                        continue
                    macro_indicators[name] = {'value': round(float(value), 2)}

            # Treasury rates (2Y/10Y)
            try:
                rates = get_fmp_client().treasury_rates()
                get_perf_stats()["fmp_calls"] += 1
                if rates:
                    latest = rates[0]
                    two_y = safe_float(latest.get("year2"), 0)
                    ten_y = safe_float(latest.get("year10"), 0)
                    if two_y:
                        macro_indicators["2Y_Yield"] = {"value": round(two_y, 2)}
                    if ten_y:
                        macro_indicators["10Y_Yield"] = {"value": round(ten_y, 2)}
            except Exception:
                pass
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
        db_doc = _db_fetch_document("sector_heatmap")
        if db_doc:
            return jsonify(db_doc)
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
    db_conn = _get_db_connection()
    if db_conn is None and not path.exists():
        return [], [], rows_read, updated_at, errors

    if db_conn is None:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone()
            updated_at = mtime.isoformat()
        except Exception:
            updated_at = None

    try:
        if db_conn is not None:
            try:
                df = pd.read_sql("SELECT ticker, date, close FROM market_prices_daily", db_conn)
                rows_read = len(df)
                try:
                    max_date = pd.to_datetime(df["date"], errors="coerce").max()
                    if pd.notna(max_date):
                        updated_at = max_date.isoformat()
                except Exception:
                    pass
            except Exception as exc:
                errors.append(f"DB_READ_FAILED: {exc}")
                if not path.exists():
                    return [], [], rows_read, updated_at, errors
                df = pd.read_csv(path, usecols=["ticker", "date", "close"])
        else:
            df = pd.read_csv(path, usecols=["ticker", "date", "close"])
        rows_read = len(df)
        logger.debug(f"_compute_top_movers: Read {rows_read} rows from {'DB' if db_conn is not None else path}")
        if df.empty:
            errors.append("EMPTY_CSV")
            return [], [], rows_read, updated_at, errors
        
        # Track rows dropped at each step for debugging
        initial_count = len(df)
        df = df.dropna(subset=["ticker", "date", "close"])
        after_basic_dropna = len(df)
        if after_basic_dropna < initial_count:
            logger.debug(f"_compute_top_movers: Dropped {initial_count - after_basic_dropna} rows with missing ticker/date/close")
        if df.empty:
            errors.append("ALL_NULL_BASIC")
            return [], [], rows_read, updated_at, errors
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        before_date_dropna = len(df)
        df = df.dropna(subset=["date"])
        after_date_dropna = len(df)
        if after_date_dropna < before_date_dropna:
            logger.warning(f"_compute_top_movers: Dropped {before_date_dropna - after_date_dropna} rows due to date parsing failure")
        if df.empty:
            errors.append("ALL_NULL_DATE")
            return [], [], rows_read, updated_at, errors
        
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        before_close_dropna = len(df)
        df = df.dropna(subset=["close"])
        after_close_dropna = len(df)
        if after_close_dropna < before_close_dropna:
            logger.debug(f"_compute_top_movers: Dropped {before_close_dropna - after_close_dropna} rows with invalid close price")
        if df.empty:
            errors.append("ALL_NULL_CLOSE")
            return [], [], rows_read, updated_at, errors
        
        # Log data range for debugging
        min_date = df["date"].min()
        max_date = df["date"].max()
        unique_tickers = df["ticker"].nunique()
        logger.debug(f"_compute_top_movers: Data range {min_date.date()} to {max_date.date()}, {unique_tickers} unique tickers, {len(df)} valid rows")

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
    finally:
        try:
            if db_conn is not None:
                db_conn.close()
        except Exception:
            pass


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
                logger.debug(f"Loaded sector map from {path} ({len(mapping)} tickers)")
                return mapping
        except Exception as e:
            logger.warning(f"Error loading sector map from {path}: {e}")
            continue
    
    # Fallback to SECTOR_MAP if no CSV found
    if SECTOR_MAP:
        logger.debug(f"Using SECTOR_MAP fallback ({len(SECTOR_MAP)} tickers)")
        return SECTOR_MAP.copy()
    
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
        raw = _db_fetch_document("options_flow") or load_options_raw(path)
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
    raw = _db_fetch_document("options_flow") or load_options_raw(path)
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

        us_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        prices = {}

        if us_tickers:
            fmp = get_fmp_client()
            fmp_symbols, reverse_map = map_symbols_to_fmp(us_tickers)
            quotes = fmp.quote(fmp_symbols)
            get_perf_stats()["fmp_calls"] += 1
            get_perf_stats()["fmp_batches"] += 1
            for item in quotes:
                symbol = item.get("symbol")
                if not symbol:
                    continue
                original = reverse_map.get(symbol, symbol)
                current = item.get("price")
                if current is None:
                    current = item.get("previousClose")
                if current is None:
                    current = item.get("open")
                if current is None:
                    continue
                open_price = item.get("open")
                high_price = item.get("dayHigh") if item.get("dayHigh") is not None else item.get("high")
                low_price = item.get("dayLow") if item.get("dayLow") is not None else item.get("low")
                ts = item.get("timestamp")
                if ts:
                    date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                else:
                    date_str = datetime.utcnow().strftime('%Y-%m-%d')
                prices[original] = {
                    'current': safe_float(current),
                    'open': safe_float(open_price),
                    'high': safe_float(high_price),
                    'low': safe_float(low_price),
                    'date': date_str
                }

        return jsonify(prices)
    except Exception as e:
        logger.error(f"Realtime prices error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/calendar')
def get_us_calendar():
    try:
        db_doc = _db_fetch_document("calendar")
        if db_doc:
            return jsonify(db_doc)
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

        end_date = datetime.utcnow().date()
        from_date = (end_date - timedelta(days=365)).isoformat()
        to_date = end_date.isoformat()
        fmp = get_fmp_client()
        fmp_symbol = to_fmp_symbol(ticker)
        data = fmp.historical_price_full(fmp_symbol, from_date=from_date, to_date=to_date)
        get_perf_stats()["fmp_calls"] += 1
        hist_list = data.get("historical", []) if isinstance(data, dict) else []
        if not hist_list:
            return jsonify({'error': 'No data'}), 404

        hist = pd.DataFrame(hist_list)
        hist['Date'] = pd.to_datetime(hist['date'])
        hist = hist.sort_values('Date')
        hist = hist.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

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
