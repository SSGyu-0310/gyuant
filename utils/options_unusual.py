import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("options_unusual")
request_logger = logging.getLogger("request_logger")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _to_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def normalize_record(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    missing = 0
    get = raw.get
    ticker = (get("ticker") or get("symbol") or "").upper()
    side = (get("side") or get("type") or get("option_type") or "").lower()
    expiry = get("expiry") or get("expiration") or get("exp") or None
    strike = get("strike") or get("strike_price") or None
    volume = get("volume") or get("vol") or get("size") or None
    oi = get("open_interest") or get("oi") or None
    premium = get("premium") or get("notional") or get("premium_usd") or None
    price = get("price") or get("fill_price") or None
    iv = get("iv") or get("implied_volatility") or None
    delta = get("delta") or None
    underlying = get("underlying_price") or get("spot") or None
    ts = get("timestamp") or get("ts") or get("time") or None
    # Count missing
    for val in [ticker, side, expiry, strike, volume, oi, premium]:
        if val in (None, ""):
            missing += 1
    norm = {
        "ticker": ticker,
        "side": side if side in ("call", "put") else None,
        "expiry": expiry,
        "strike": _to_float(strike) if strike not in (None, "") else None,
        "volume": int(_to_float(volume)) if volume not in (None, "") else None,
        "open_interest": int(_to_float(oi)) if oi not in (None, "") else None,
        "premium": _to_float(premium) if premium not in (None, "") else None,
        "price": _to_float(price) if price not in (None, "") else None,
        "iv": _to_float(iv) if iv not in (None, "") else None,
        "delta": _to_float(delta) if delta not in (None, "") else None,
        "underlying_price": _to_float(underlying) if underlying not in (None, "") else None,
        "ts": ts,
        "raw": raw,
    }
    return norm, missing


def _score_record(rec: Dict[str, Any], cfg: Dict[str, float]) -> Tuple[float, List[str]]:
    score = 0.0
    tags: List[str] = []
    premium = rec.get("premium") or 0
    volume = rec.get("volume") or 0
    oi = rec.get("open_interest") or 0
    price = rec.get("price") or 0
    notional = premium if premium else price * volume
    vol_oi_ratio = (volume / oi) if oi else 0

    if notional and notional >= cfg["premium_high_usd"]:
        score += 50
        tags.append("HIGH_PREMIUM")
    elif notional and notional >= cfg["premium_mid_usd"]:
        score += 30
        tags.append("MID_PREMIUM")

    if volume and oi and vol_oi_ratio >= cfg["vol_oi_ratio_high"]:
        score += 30
        tags.append("VOL_OI_SPIKE")
    elif volume and oi and vol_oi_ratio >= cfg["vol_oi_ratio_mid"]:
        score += 15
        tags.append("VOL_OI_ELEVATED")

    if volume and volume >= cfg["volume_large"]:
        score += 10
        tags.append("LARGE_VOLUME")

    if rec.get("iv") and rec["iv"] >= cfg["iv_high"]:
        score += 10
        tags.append("HIGH_IV")

    return min(100, round(score, 1)), tags


def score_options_flow(raw_flow: List[Dict[str, Any]], source_file: Path, cfg: Dict[str, float]) -> List[Dict[str, Any]]:
    records_out: List[Dict[str, Any]] = []
    missing_count = 0
    for rec in raw_flow or []:
        norm, missing = normalize_record(rec or {})
        missing_count += missing
        score, tags = _score_record(norm, cfg)
        combined = {**(rec or {}), **norm}
        combined["unusual_score"] = score
        combined["unusual_tags"] = tags
        records_out.append(combined)

    # Only log in verbose mode (DEBUG_LOGS=true)
    if os.getenv("DEBUG_LOGS", "").lower() == "true":
        event = {
            "ts": _now_iso(),
            "level": "info",
            "event": "options_unusual_scoring",
            "source_file": str(source_file),
            "records_in": len(raw_flow or []),
            "records_out": len(records_out),
            "missing_fields_count": missing_count,
        }
        try:
            request_logger.info(json.dumps(event, ensure_ascii=False))
        except Exception:
            logger.info(event)
    return records_out


def load_options_raw(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"options_flow": [], "summary": {}, "timestamp": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"options_flow": [], "summary": {}, "timestamp": None}
