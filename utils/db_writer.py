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
