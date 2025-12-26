#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Database Schema
Defines SQLite tables for Point-in-Time backtesting
"""

import sqlite3
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent / "backtest.db"


def get_db_path() -> Path:
    """Get database path from environment or default"""
    data_dir = os.getenv('DATA_DIR', str(Path(__file__).parent.parent / 'us_market'))
    return Path(data_dir) / 'us_price.db'


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get SQLite connection with proper settings.
    
    Args:
        db_path: Path to database file (uses default if None)
        
    Returns:
        SQLite connection with WAL mode enabled
    """
    if db_path is None:
        db_path = get_db_path()
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
    conn.row_factory = sqlite3.Row
    return conn


# SQL schema definitions
SCHEMA_SQL = """
-- daily_prices: 가격 데이터 (Composite PK: ticker, date)
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    change REAL,
    change_rate REAL,
    name TEXT,
    market TEXT,
    source TEXT DEFAULT 'FMP',
    as_of TEXT,
    PRIMARY KEY (ticker, date)
);

-- universe_snapshots: 유니버스 스냅샷 (Survivorship bias 방지)
CREATE TABLE IF NOT EXISTS universe_snapshots (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    market TEXT,
    PRIMARY KEY (date, ticker)
);

-- signals: 전략별 시그널 저장
CREATE TABLE IF NOT EXISTS signals (
    as_of_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    signal_value REAL,
    signal_version TEXT,
    created_at TEXT,
    PRIMARY KEY (as_of_date, ticker, signal_id)
);

-- 인덱스: 쿼리 성능 최적화
CREATE INDEX IF NOT EXISTS idx_prices_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON daily_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_universe_date ON universe_snapshots(date);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(as_of_date);
CREATE INDEX IF NOT EXISTS idx_signals_id ON signals(signal_id);
"""


def init_db(db_path: Optional[Path] = None) -> bool:
    """
    Initialize database with all tables.
    
    Args:
        db_path: Path to database file (uses default if None)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_connection(db_path)
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized: {db_path or get_db_path()}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return False


def get_table_counts(db_path: Optional[Path] = None) -> dict:
    """Get row counts for all tables"""
    try:
        conn = get_connection(db_path)
        counts = {}
        for table in ['daily_prices', 'universe_snapshots', 'signals']:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        conn.close()
        return counts
    except Exception as e:
        logger.error(f"Failed to get table counts: {e}")
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🗄️ Initializing Backtest Database...")
    if init_db():
        counts = get_table_counts()
        print(f"📊 Table counts: {counts}")
    else:
        print("❌ Database initialization failed")
