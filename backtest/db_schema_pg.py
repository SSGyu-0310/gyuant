#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gyuant Market Database Schema for PostgreSQL (Multi-Schema)

 Schema Organization:
 - market: 운영 데이터 (가격, 종목 마스터)
 - backtest: 백테스트 전용 데이터 (PIT 가격, 유니버스 스냅샷, 실행 결과)
 - factors: 팩터 데이터 (볼륨 분석, 스마트 머니)
 - direct_indexing: 다이렉트 인덱싱 데이터 (포트폴리오, 포지션, 세금 로트)
"""

import os
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Date,
    DateTime,
    Numeric,
    BigInteger,
    Integer,
    Boolean,
    Text,
    Float,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint,
    func,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.engine import Engine

from utils.db import get_database_url
from utils.env import load_env

load_env()

logger = logging.getLogger(__name__)

Base = declarative_base()


_engine: Optional[Engine] = None
_SessionLocal = None


def get_engine() -> Engine:
    """Get or create SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        url = get_database_url()
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=os.getenv("SQL_ECHO", "").lower() == "true",
        )
        logger.info(
            f"PostgreSQL engine created: {url.split('@')[1] if '@' in url else url}"
        )
    return _engine


def get_session():
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


# =============================================================================
# MARKET SCHEMA - 운영 데이터
# =============================================================================


class MarketTicker(Base):
    """종목 마스터 테이블 (market.tickers)"""

    __tablename__ = "tickers"
    __table_args__ = {"schema": "market"}

    symbol = Column(String(20), primary_key=True, comment="종목 심볼")
    name = Column(String(255), comment="회사명")
    sector = Column(String(100), comment="섹터")
    industry = Column(String(255), comment="산업")
    exchange = Column(String(50), comment="거래소")
    market_cap = Column(BigInteger, comment="시가총액")
    country = Column(String(50), default="US")
    is_active = Column(Boolean, default=True)
    date_added = Column(Date, comment="지수 편입일")
    date_removed = Column(Date, comment="지수 제외일")
    source = Column(String(50), default="FMP")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class MarketDailyPrice(Base):
    """일봉 가격 테이블 (market.daily_prices)"""

    __tablename__ = "daily_prices"
    __table_args__ = (
        Index("idx_market_prices_date", "date"),
        Index("idx_market_prices_symbol_date", "symbol", "date"),
        {"schema": "market"},
    )

    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)
    open = Column(Numeric(15, 4))
    high = Column(Numeric(15, 4))
    low = Column(Numeric(15, 4))
    close = Column(Numeric(15, 4))
    adj_close = Column(Numeric(15, 4))
    volume = Column(BigInteger)
    change = Column(Numeric(15, 4))
    change_pct = Column(Numeric(10, 4))
    source = Column(String(50), default="FMP")
    ingested_at = Column(DateTime, server_default=func.now())


class MarketETFFlow(Base):
    """ETF 자금 흐름 (market.etf_flows)"""

    __tablename__ = "etf_flows"
    __table_args__ = (Index("idx_etf_flows_date", "as_of_date"), {"schema": "market"})

    ticker = Column(String(20), primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    name = Column(String(255))
    category = Column(String(100))
    current_price = Column(Numeric(15, 4))
    price_1w_pct = Column(Numeric(10, 4))
    price_1m_pct = Column(Numeric(10, 4))
    vol_ratio_5d_20d = Column(Numeric(10, 4))
    obv_change_20d_pct = Column(Numeric(10, 4))
    avg_volume_20d = Column(BigInteger)
    flow_score = Column(Numeric(10, 4))
    flow_status = Column(String(50))
    source = Column(String(50), default="FMP")
    ingested_at = Column(DateTime, server_default=func.now())


# =============================================================================
# BACKTEST SCHEMA - 백테스트 전용 (PIT)
# =============================================================================


class BacktestPrice(Base):
    """백테스트용 가격 데이터 (backtest.prices_daily)"""

    __tablename__ = "prices_daily"
    __table_args__ = (Index("idx_bt_prices_date", "date"), {"schema": "backtest"})

    ticker = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)
    open = Column(Numeric(15, 4))
    high = Column(Numeric(15, 4))
    low = Column(Numeric(15, 4))
    close = Column(Numeric(15, 4))
    volume = Column(BigInteger)
    source = Column(String(50))
    ingested_at = Column(DateTime, server_default=func.now())


class BacktestUniverse(Base):
    """유니버스 스냅샷 (backtest.universe_snapshot)"""

    __tablename__ = "universe_snapshot"
    __table_args__ = (
        Index("idx_bt_universe_asof", "as_of_date"),
        {"schema": "backtest"},
    )

    as_of_date = Column(Date, primary_key=True)
    ticker = Column(String(20), primary_key=True)
    name = Column(String(255))
    sector = Column(String(100))
    market = Column(String(50))
    weight = Column(Numeric(10, 6))
    source = Column(String(50), default="FMP")
    ingested_at = Column(DateTime, server_default=func.now())


class BacktestFinancialStatement(Base):
    """재무제표 (PIT) (backtest.financial_statements)"""

    __tablename__ = "financial_statements"
    __table_args__ = (
        Index("idx_bt_fin_filing", "filing_date"),
        Index("idx_bt_fin_symbol_filing", "symbol", "filing_date"),
        CheckConstraint(
            "filing_date >= period_end_date", name="chk_filing_after_period"
        ),
        {"schema": "backtest"},
    )

    symbol = Column(String(20), primary_key=True)
    period_end_date = Column(Date, primary_key=True)
    statement_type = Column(String(20), primary_key=True)
    filing_date = Column(Date, nullable=False, comment="SEC 공시일 (PIT key)")
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(String(5))
    period_type = Column(String(20), default="quarterly")
    revenue = Column(BigInteger)
    gross_profit = Column(BigInteger)
    operating_income = Column(BigInteger)
    net_income = Column(BigInteger)
    eps = Column(Numeric(10, 4))
    eps_diluted = Column(Numeric(10, 4))
    total_assets = Column(BigInteger)
    total_liabilities = Column(BigInteger)
    total_equity = Column(BigInteger)
    source = Column(String(50), default="FMP")
    ingested_at = Column(DateTime, server_default=func.now())


class BacktestSignalDefinition(Base):
    """백테스트 신호 정의 (backtest.signal_definitions)"""

    __tablename__ = "signal_definitions"
    __table_args__ = {"schema": "backtest"}

    signal_id = Column(String(50), primary_key=True)
    version = Column(String(20), primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    params_schema_json = Column(Text)  # JSON string for parameters schema
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class BacktestSignal(Base):
    """백테스트 신호 데이터 (backtest.signals)"""

    __tablename__ = "signals"
    __table_args__ = (
        Index("idx_bt_signals_asof", "as_of_date"),
        Index("idx_bt_signals_id_asof", "signal_id", "signal_version", "as_of_date"),
        {"schema": "backtest"},
    )

    signal_id = Column(String(50), nullable=False)
    signal_version = Column(String(20), nullable=False)
    as_of_date = Column(Date, nullable=False)
    ticker = Column(String(20), nullable=False)
    signal_value = Column(Numeric(10, 4))
    signal_rank = Column(Integer)
    meta_json = Column(Text)  # Additional metadata as JSON

    __mapper_args__ = {"primary_key": [signal_id, signal_version, as_of_date, ticker]}


class BacktestRun(Base):
    """백테스트 실행 기록 (backtest.runs)"""

    __tablename__ = "runs"
    __table_args__ = (
        Index("idx_bt_runs_status", "status"),
        Index("idx_bt_runs_asof", "as_of_date"),
        {"schema": "backtest"},
    )

    run_id = Column(String(50), primary_key=True)
    strategy_name = Column(String(100))  # Strategy name (required by existing DB)
    signal_id = Column(String(50))
    signal_version = Column(String(20))
    alpha_id = Column(String(50))
    alpha_version = Column(String(20))
    as_of_date = Column(Date, nullable=False, comment="Backtest as-of date (PIT)")
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    universe = Column(String(50))
    config_json = Column(Text)  # Full configuration as JSON
    top_n = Column(Integer)
    hold_period_days = Column(Integer)
    rebalance_freq = Column(String(20), default="quarterly")
    transaction_cost_bps = Column(Numeric(6, 2), default=10)
    status = Column(String(20), default="queued")
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    finished_at = Column(DateTime)

    __table_args__ = (
        Index("idx_bt_runs_status", "status"),
        Index("idx_bt_runs_asof", "as_of_date"),
        CheckConstraint(
            "status IN ('queued', 'running', 'finished', 'failed')",
            name="chk_bt_status",
        ),
        {"schema": "backtest"},
    )


class BacktestRunMetrics(Base):
    """백테스트 실행 성과 지표 (backtest.run_metrics)"""

    __tablename__ = "run_metrics"
    __table_args__ = {"schema": "backtest"}

    run_id = Column(String(50), primary_key=True)
    cagr = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 4))
    sharpe = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    total_return = Column(Numeric(10, 4))
    win_rate = Column(Numeric(6, 4))
    turnover = Column(Numeric(10, 4))
    created_at = Column(DateTime, server_default=func.now())


class BacktestEquityCurve(Base):
    """백테스트 자본 곡선 (backtest.equity_curve)"""

    __tablename__ = "equity_curve"
    __table_args__ = (
        Index("idx_bt_equity_run_date", "run_id", "date"),
        {"schema": "backtest"},
    )

    run_id = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    equity = Column(Numeric(15, 2))
    cash = Column(Numeric(15, 2))
    returns = Column(Numeric(10, 4))
    drawdown = Column(Numeric(10, 4))

    __mapper_args__ = {"primary_key": [run_id, date]}


class BacktestPosition(Base):
    """백테스트 포지션 추적 (backtest.positions)"""

    __tablename__ = "positions"
    __table_args__ = (Index("idx_bt_positions_run", "run_id"), {"schema": "backtest"})

    run_id = Column(String(50), nullable=False)
    ticker = Column(String(20), nullable=False)
    entry_date = Column(Date, nullable=False)
    entry_price = Column(Numeric(15, 4))
    exit_date = Column(Date)
    exit_price = Column(Numeric(15, 4))
    weight = Column(Numeric(10, 6))
    shares = Column(Numeric(15, 4))

    __mapper_args__ = {"primary_key": [run_id, ticker, entry_date]}


class BacktestTrade(Base):
    """백테스트 거래 내역 (backtest.trades)"""

    __tablename__ = "trades"
    __table_args__ = (
        Index("idx_bt_trades_run_date", "run_id", "trade_date"),
        {"schema": "backtest"},
    )

    run_id = Column(String(50), nullable=False)
    trade_id = Column(String(100), primary_key=True)
    ticker = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy', 'sell'
    trade_date = Column(Date, nullable=False)
    price = Column(Numeric(15, 4))
    shares = Column(Numeric(15, 4))
    fee = Column(Numeric(10, 2))


# =============================================================================
# FACTORS SCHEMA - 팩터/다이렉트 인덱싱
# =============================================================================


class FactorVolumeAnalysis(Base):
    """볼륨 분석 팩터 (factors.volume_analysis)"""

    __tablename__ = "volume_analysis"
    __table_args__ = (Index("idx_factor_vol_date", "as_of_date"), {"schema": "factors"})

    ticker = Column(String(20), primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    name = Column(String(255))
    obv = Column(Numeric(20, 4))
    obv_change_20d = Column(Numeric(10, 4))
    ad_line = Column(Numeric(20, 4))
    ad_change_20d = Column(Numeric(10, 4))
    mfi = Column(Numeric(10, 4))
    vol_ratio_5d_20d = Column(Numeric(10, 4))
    surge_count_5d = Column(Integer)
    surge_count_20d = Column(Integer)
    supply_demand_score = Column(Numeric(10, 4))
    supply_demand_stage = Column(String(50))
    source = Column(String(50))
    ingested_at = Column(DateTime, server_default=func.now())


class FactorSmartMoneyPick(Base):
    """스마트 머니 추천 (factors.smart_money_picks)"""

    __tablename__ = "smart_money_picks"
    __table_args__ = (Index("idx_factor_sm_date", "run_date"), {"schema": "factors"})

    run_date = Column(Date, primary_key=True)
    rank = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    name = Column(String(255))
    sector = Column(String(100))
    composite_score = Column(Numeric(10, 4))
    smart_money_score = Column(Numeric(10, 4))
    grade = Column(String(5))
    current_price = Column(Numeric(15, 4))
    target_upside = Column(Numeric(10, 4))
    sd_score = Column(Numeric(10, 4))
    tech_score = Column(Numeric(10, 4))
    fund_score = Column(Numeric(10, 4))
    analyst_score = Column(Numeric(10, 4))
    rs_score = Column(Numeric(10, 4))
    recommendation = Column(String(50))
    source = Column(String(50))
    ingested_at = Column(DateTime, server_default=func.now())


class FactorFundamental(Base):
    """펀더멘탈 팩터 (factors.fundamentals)"""

    __tablename__ = "fundamentals"
    __table_args__ = (
        Index("idx_factor_fund_date", "as_of_date"),
        {"schema": "factors"},
    )

    ticker = Column(String(20), primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    pe_ratio = Column(Numeric(10, 4))
    pb_ratio = Column(Numeric(10, 4))
    ps_ratio = Column(Numeric(10, 4))
    ev_ebitda = Column(Numeric(10, 4))
    dividend_yield = Column(Numeric(10, 4))
    roe = Column(Numeric(10, 4))
    roa = Column(Numeric(10, 4))
    current_ratio = Column(Numeric(10, 4))
    debt_to_equity = Column(Numeric(10, 4))
    revenue_growth = Column(Numeric(10, 4))
    earnings_growth = Column(Numeric(10, 4))
    source = Column(String(50), default="FMP")
    ingested_at = Column(DateTime, server_default=func.now())


# =============================================================================
# DIRECT INDEXING SCHEMA - 다이렉트 인덱싱
# =============================================================================


class DirectIndexingPortfolio(Base):
    """다이렉트 인덱싱 포트폴리오 마스터 (direct_indexing.portfolios)"""

    __tablename__ = "portfolios"
    __table_args__ = {"schema": "direct_indexing"}

    portfolio_id = Column(String(50), primary_key=True)
    name = Column(String(255))
    benchmark = Column(String(20))
    base_universe = Column(String(50))
    weighting_method = Column(String(20))  # 'equal', 'market_cap', 'custom'
    rebalance_freq = Column(String(20), default="quarterly")
    initial_capital = Column(Numeric(15, 2), default=100000)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DirectIndexingPosition(Base):
    """다이렉트 인덱싱 현재 포지션 (direct_indexing.positions)"""

    __tablename__ = "positions"
    __table_args__ = (
        Index("idx_di_positions_portfolio", "portfolio_id"),
        Index("idx_di_positions_date", "portfolio_id", "as_of_date"),
        {"schema": "direct_indexing"},
    )

    portfolio_id = Column(String(50), nullable=False)
    as_of_date = Column(Date, nullable=False)
    ticker = Column(String(20), nullable=False)
    shares = Column(Numeric(15, 4))
    cost_basis = Column(Numeric(15, 2))
    market_value = Column(Numeric(15, 2))
    weight = Column(Numeric(10, 6))

    __mapper_args__ = {"primary_key": [portfolio_id, as_of_date, ticker]}


class DirectIndexingTaxLot(Base):
    """세금 로트 추적 (direct_indexing.tax_lots)"""

    __tablename__ = "tax_lots"
    __table_args__ = (
        Index("idx_di_taxlots_portfolio", "portfolio_id"),
        {"schema": "direct_indexing"},
    )

    lot_id = Column(String(100), primary_key=True)
    portfolio_id = Column(String(50), nullable=False)
    ticker = Column(String(20), nullable=False)
    acquisition_date = Column(Date, nullable=False)
    shares = Column(Numeric(15, 4))
    cost_basis = Column(Numeric(15, 2))
    wash_sale_until = Column(Date)  # Wash sale lock period end date


class DirectIndexingRebalanceLog(Base):
    """리밸런스 로그 (direct_indexing.rebalance_logs)"""

    __tablename__ = "rebalance_logs"
    __table_args__ = (
        Index("idx_di_rebal_portfolio", "portfolio_id"),
        {"schema": "direct_indexing"},
    )

    rebalance_id = Column(String(100), primary_key=True)
    portfolio_id = Column(String(50), nullable=False)
    as_of_date = Column(Date, nullable=False)
    action_json = Column(Text)  # Rebalance actions as JSON
    created_at = Column(DateTime, server_default=func.now())


class DirectIndexingTlhEvent(Base):
    """Tax-loss harvesting 이벤트 (direct_indexing.tlh_events)"""

    __tablename__ = "tlh_events"
    __table_args__ = (
        Index("idx_di_tlh_portfolio", "portfolio_id"),
        {"schema": "direct_indexing"},
    )

    event_id = Column(String(100), primary_key=True)
    portfolio_id = Column(String(50), nullable=False)
    ticker = Column(String(20), nullable=False)
    loss_amount = Column(Numeric(15, 2))
    sale_date = Column(Date, nullable=False)
    replacement_ticker = Column(String(20))
    wash_sale_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())


# =============================================================================
# Database Initialization
# =============================================================================


def create_schemas(engine: Engine):
    """Create all schemas if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS market"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS backtest"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS factors"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS direct_indexing"))
        conn.commit()
    logger.info("✅ Schemas created: market, backtest, factors, direct_indexing")


def ensure_backtest_runs_columns(engine: Engine):
    """Add missing columns to backtest.runs for backward compatibility."""
    statements = [
        "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS signal_id VARCHAR(50);",
        "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS signal_version VARCHAR(20);",
        "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS alpha_id VARCHAR(50);",
        "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS alpha_version VARCHAR(20);",
        "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS top_n INTEGER;",
        "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS hold_period_days INTEGER;",
        "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS transaction_cost_bps NUMERIC(6, 2) DEFAULT 10;",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("✅ Ensured backtest.runs compatibility columns exist")


def init_db(drop_existing: bool = False) -> bool:
    """Initialize all tables in the PostgreSQL database."""
    try:
        engine = get_engine()

        # Create schemas first
        create_schemas(engine)

        if drop_existing:
            logger.warning("⚠️ Dropping all existing tables...")
            Base.metadata.drop_all(engine)

        Base.metadata.create_all(engine)
        ensure_backtest_runs_columns(engine)
        logger.info("✅ PostgreSQL database initialized successfully")

        # Log created tables by schema
        for table in Base.metadata.sorted_tables:
            schema = table.schema or "public"
            logger.info(f"   📦 {schema}.{table.name}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def get_table_counts() -> dict:
    """Get row counts for all tables."""
    engine = get_engine()
    counts: dict = {}

    for table in Base.metadata.sorted_tables:
        try:
            schema = table.schema or "public"
            full_name = f"{schema}.{table.name}"
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {full_name}"))
                counts[full_name] = result.scalar()
        except Exception:
            counts[f"{table.schema or 'public'}.{table.name}"] = 0

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gyuant Database Schema Manager")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DROP all existing tables and recreate them (WARNING: Data loss)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Show table row counts and exit"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🐘 PostgreSQL Multi-Schema Database Manager")
    print("=" * 60)

    try:
        if args.check:
            counts = get_table_counts()
            print("\n📊 Table Row Counts:")
            for table, count in counts.items():
                print(f"   {table}: {count}")
        else:
            if args.reset:
                print("⚠️  WARNING: You are about to DROP ALL TABLES.")
                confirm = input("Are you sure? (yes/no): ")
                if confirm.lower() != "yes":
                    print("❌ Operation cancelled.")
                    exit(1)

            init_db(drop_existing=args.reset)
            
            counts = get_table_counts()
            print("\n📊 Table Row Counts:")
            for table, count in counts.items():
                print(f"   {table}: {count}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

