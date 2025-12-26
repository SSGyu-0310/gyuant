-- Gyuant SQLite schema (draft)
-- Single source of truth for operational + backtest data.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

BEGIN;

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
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker) REFERENCES market_stocks(ticker) ON UPDATE CASCADE ON DELETE RESTRICT
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
    PRIMARY KEY (ticker, as_of_date),
    FOREIGN KEY (ticker) REFERENCES market_stocks(ticker) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_market_volume_asof
    ON market_volume_analysis(as_of_date);

CREATE INDEX IF NOT EXISTS idx_market_volume_score
    ON market_volume_analysis(as_of_date, supply_demand_score);

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
    UNIQUE (run_id, rank),
    FOREIGN KEY (run_id) REFERENCES market_smart_money_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_market_picks_run
    ON market_smart_money_picks(run_id);

CREATE INDEX IF NOT EXISTS idx_market_picks_ticker
    ON market_smart_money_picks(ticker);

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
    PRIMARY KEY (signal_id, signal_version, as_of_date, ticker),
    FOREIGN KEY (signal_id, signal_version)
        REFERENCES bt_signal_definitions(signal_id, version)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_bt_signals_asof
    ON bt_signals(as_of_date, signal_id, signal_version);

CREATE INDEX IF NOT EXISTS idx_bt_signals_rank
    ON bt_signals(signal_id, signal_version, as_of_date, signal_rank);

CREATE TABLE IF NOT EXISTS bt_fundamentals (
    as_of_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    pe_ratio REAL,
    pb_ratio REAL,
    revenue_growth REAL,
    roe REAL,
    market_cap REAL,
    source TEXT,
    PRIMARY KEY (as_of_date, ticker)
);

CREATE TABLE IF NOT EXISTS bt_alpha_defs (
    alpha_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    CHECK (status IN ('draft', 'active', 'deprecated'))
);

CREATE TABLE IF NOT EXISTS bt_alpha_versions (
    alpha_id TEXT NOT NULL,
    version TEXT NOT NULL,
    expression TEXT NOT NULL,
    params_schema_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (alpha_id, version),
    FOREIGN KEY (alpha_id) REFERENCES bt_alpha_defs(alpha_id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bt_alpha_nl_requests (
    request_id TEXT PRIMARY KEY,
    alpha_id TEXT,
    user_input TEXT NOT NULL,
    llm_output TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    approved_at TEXT,
    error TEXT,
    CHECK (status IN ('pending', 'approved', 'rejected', 'failed')),
    FOREIGN KEY (alpha_id) REFERENCES bt_alpha_defs(alpha_id)
        ON UPDATE CASCADE ON DELETE SET NULL
);

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
    CHECK (status IN ('queued', 'running', 'finished', 'failed')),
    FOREIGN KEY (signal_id, signal_version)
        REFERENCES bt_signal_definitions(signal_id, version)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (alpha_id, alpha_version)
        REFERENCES bt_alpha_versions(alpha_id, version)
        ON UPDATE CASCADE ON DELETE SET NULL
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
    turnover REAL,
    FOREIGN KEY (run_id) REFERENCES bt_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bt_run_equity_curve (
    run_id TEXT NOT NULL,
    date TEXT NOT NULL,
    equity REAL,
    returns REAL,
    drawdown REAL,
    PRIMARY KEY (run_id, date),
    FOREIGN KEY (run_id) REFERENCES bt_runs(run_id) ON DELETE CASCADE
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
    PRIMARY KEY (run_id, ticker, entry_date),
    FOREIGN KEY (run_id) REFERENCES bt_runs(run_id) ON DELETE CASCADE
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
    PRIMARY KEY (run_id, trade_id),
    FOREIGN KEY (run_id) REFERENCES bt_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bt_alpha_runs (
    run_id TEXT PRIMARY KEY,
    alpha_id TEXT NOT NULL,
    alpha_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES bt_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (alpha_id, alpha_version)
        REFERENCES bt_alpha_versions(alpha_id, version)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

COMMIT;
