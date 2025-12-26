#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gyuant Market Database Schema
Aligned with docs/db/schema.sql - Single source of truth for operational + backtest data
"""

import sqlite3
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Database filename
DB_FILENAME = "gyuant_market.db"


def get_db_path() -> Path:
    """Get database path from environment or default"""
    data_dir = os.getenv('DATA_DIR', str(Path(__file__).parent.parent / 'us_market'))
    return Path(data_dir) / DB_FILENAME


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
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    return conn


# SQL schema - aligned with docs/db/schema.sql
SCHEMA_SQL = """
-- ==========================================================
-- Market (operational) tables
-- ==========================================================
CREATE TABLE IF NOT EXISTS market_stocks (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market TEXT,
    exchange TEXT,
    currency TEXT,
    is_active INTEGER DEFAULT 1,
    source TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_prices_daily (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume REAL,
    change REAL,
    change_rate REAL,
    source TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_market_prices_date
    ON market_prices_daily(date);

CREATE TABLE IF NOT EXISTS market_volume_analysis (
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    name TEXT,
    obv REAL,
    obv_change_20d REAL,
    ad_line REAL,
    ad_change_20d REAL,
    mfi REAL,
    vol_ratio_5d_20d REAL,
    surge_count_5d INTEGER,
    surge_count_20d INTEGER,
    supply_demand_score REAL,
    supply_demand_stage TEXT,
    source TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_market_volume_asof
    ON market_volume_analysis(as_of_date);

CREATE TABLE IF NOT EXISTS market_etf_flows (
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    name TEXT,
    category TEXT,
    current_price REAL,
    price_1w_pct REAL,
    price_1m_pct REAL,
    vol_ratio_5d_20d REAL,
    obv_change_20d_pct REAL,
    avg_volume_20d REAL,
    flow_score REAL,
    flow_status TEXT,
    source TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_market_etf_asof
    ON market_etf_flows(as_of_date);

CREATE TABLE IF NOT EXISTS market_smart_money_runs (
    run_id TEXT PRIMARY KEY,
    analysis_date TEXT NOT NULL,
    analysis_timestamp TEXT,
    summary_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_smart_money_picks (
    run_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    composite_score REAL,
    grade TEXT,
    current_price REAL,
    price_at_rec REAL,
    change_since_rec REAL,
    target_upside REAL,
    sd_score REAL,
    tech_score REAL,
    fund_score REAL,
    analyst_score REAL,
    rs_score REAL,
    recommendation TEXT,
    rsi REAL,
    ma_signal TEXT,
    pe_ratio REAL,
    market_cap_b REAL,
    size TEXT,
    rs_20d REAL,
    PRIMARY KEY (run_id, ticker),
    UNIQUE (run_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_market_picks_run
    ON market_smart_money_picks(run_id);

CREATE TABLE IF NOT EXISTS market_documents (
    doc_type TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    lang TEXT NOT NULL DEFAULT 'na',
    model TEXT NOT NULL DEFAULT 'na',
    payload_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_type, as_of_date, lang, model)
);

CREATE INDEX IF NOT EXISTS idx_market_docs_type_date
    ON market_documents(doc_type, as_of_date);

-- ==========================================================
-- Backtest tables (bt_*)
-- ==========================================================
CREATE TABLE IF NOT EXISTS bt_prices_daily (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    source TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_bt_prices_date
    ON bt_prices_daily(date);

CREATE TABLE IF NOT EXISTS bt_universe_snapshot (
    as_of_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    market TEXT,
    source TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_bt_universe_asof
    ON bt_universe_snapshot(as_of_date);

CREATE TABLE IF NOT EXISTS bt_signal_definitions (
    signal_id TEXT NOT NULL,
    version TEXT NOT NULL,
    name TEXT,
    description TEXT,
    params_schema_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    PRIMARY KEY (signal_id, version)
);

CREATE TABLE IF NOT EXISTS bt_signals (
    signal_id TEXT NOT NULL,
    signal_version TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal_value REAL,
    signal_rank INTEGER,
    meta_json TEXT,
    PRIMARY KEY (signal_id, signal_version, as_of_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_bt_signals_asof
    ON bt_signals(as_of_date, signal_id, signal_version);

CREATE INDEX IF NOT EXISTS idx_bt_signals_rank
    ON bt_signals(signal_id, signal_version, as_of_date, signal_rank);

CREATE TABLE IF NOT EXISTS bt_runs (
    run_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    signal_version TEXT NOT NULL,
    alpha_id TEXT,
    alpha_version TEXT,
    config_json TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    top_n INTEGER NOT NULL,
    hold_period_days INTEGER NOT NULL,
    rebalance_freq TEXT DEFAULT 'none',
    transaction_cost_bps REAL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    error TEXT,
    CHECK (status IN ('queued', 'running', 'finished', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_bt_runs_status
    ON bt_runs(status);

CREATE INDEX IF NOT EXISTS idx_bt_runs_asof
    ON bt_runs(as_of_date);

CREATE TABLE IF NOT EXISTS bt_run_metrics (
    run_id TEXT PRIMARY KEY,
    cagr REAL,
    volatility REAL,
    sharpe REAL,
    mdd REAL,
    total_return REAL,
    win_rate REAL,
    turnover REAL
);

CREATE TABLE IF NOT EXISTS bt_run_equity_curve (
    run_id TEXT NOT NULL,
    date TEXT NOT NULL,
    equity REAL,
    returns REAL,
    drawdown REAL,
    PRIMARY KEY (run_id, date)
);

CREATE TABLE IF NOT EXISTS bt_run_positions (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL,
    exit_date TEXT,
    exit_price REAL,
    weight REAL,
    shares REAL,
    PRIMARY KEY (run_id, ticker, entry_date)
);

CREATE TABLE IF NOT EXISTS bt_run_trades (
    run_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    price REAL,
    shares REAL,
    fee REAL,
    PRIMARY KEY (run_id, trade_id)
);
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
    tables = [
        'market_stocks', 'market_prices_daily', 'market_volume_analysis',
        'market_etf_flows', 'market_smart_money_runs', 'market_smart_money_picks',
        'market_documents', 'bt_prices_daily', 'bt_universe_snapshot',
        'bt_signal_definitions', 'bt_signals', 'bt_runs', 'bt_run_metrics'
    ]
    try:
        conn = get_connection(db_path)
        counts = {}
        for table in tables:
            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            except:
                counts[table] = 0
        conn.close()
        return counts
    except Exception as e:
        logger.error(f"Failed to get table counts: {e}")
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🗄️ Initializing Gyuant Market Database...")
    if init_db():
        counts = get_table_counts()
        print(f"📊 Table counts:")
        for table, count in counts.items():
            print(f"   {table}: {count}")
    else:
        print("❌ Database initialization failed")
