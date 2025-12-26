import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

from backtest.db_schema import get_connection, init_db

logger = logging.getLogger(__name__)

USE_SQLITE = os.getenv("USE_SQLITE", "true").lower() == "true"


def get_db_connection():
    if not USE_SQLITE:
        return None
    if not init_db():
        logger.warning("DB init failed; skipping DB write")
        return None
    return get_connection()


def write_market_documents(
    doc_type: str,
    payload: Any,
    as_of_date: Optional[str] = None,
    lang: str = "na",
    model: str = "na",
) -> int:
    if not USE_SQLITE:
        return 0
    conn = get_db_connection()
    if conn is None:
        return 0
    if not as_of_date:
        as_of_date = datetime.now().strftime("%Y-%m-%d")
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO market_documents
            (doc_type, as_of_date, lang, model, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doc_type,
                as_of_date,
                lang,
                model,
                json.dumps(payload, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return 1
    except Exception as e:
        logger.warning("Failed to write market_documents %s: %s", doc_type, e)
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_latest_document(doc_type: str, lang: str = "na", model: str = "na") -> Any:
    if not USE_SQLITE:
        return {}
    conn = get_db_connection()
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
        return json.loads(payload) if payload else {}
    except Exception as e:
        logger.warning("Failed to read market_documents %s: %s", doc_type, e)
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass
