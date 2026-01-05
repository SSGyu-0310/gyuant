#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gyuant Market Database Schema for PostgreSQL (Multi-Schema)

Schema Organization:
- market: 운영 데이터 (가격, 종목 마스터)
- backtest: 백테스트 전용 데이터 (PIT 가격, 유니버스 스냅샷)
- factors: 팩터/다이렉트 인덱싱 데이터 (볼륨 분석, 스마트 머니)
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

Base = declarative_base()


# =============================================================================
# Database Connection
# =============================================================================

def get_database_url() -> str:
    """Build PostgreSQL connection URL from environment variables."""
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    database = os.getenv("PG_DATABASE", "gyuant_market")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


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
        logger.info(f"PostgreSQL engine created: {url.split('@')[1] if '@' in url else url}")
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
        {"schema": "market"}
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
    __table_args__ = (
        Index("idx_etf_flows_date", "as_of_date"),
        {"schema": "market"}
    )
    
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
    __table_args__ = (
        Index("idx_bt_prices_date", "date"),
        {"schema": "backtest"}
    )
    
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
        {"schema": "backtest"}
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
        CheckConstraint("filing_date >= period_end_date", name="chk_filing_after_period"),
        {"schema": "backtest"}
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


class BacktestRun(Base):
    """백테스트 실행 기록 (backtest.runs)"""
    __tablename__ = "runs"
    __table_args__ = (
        Index("idx_bt_runs_status", "status"),
        {"schema": "backtest"}
    )
    
    run_id = Column(String(50), primary_key=True)
    strategy_name = Column(String(100), nullable=False)
    strategy_version = Column(String(20), default="1.0")
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    universe = Column(String(50))
    initial_capital = Column(Numeric(15, 2), default=100000)
    top_n = Column(Integer)
    rebalance_freq = Column(String(20), default="monthly")
    transaction_cost_bps = Column(Numeric(6, 2), default=10)
    total_return = Column(Numeric(10, 4))
    cagr = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    status = Column(String(20), default="pending")
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    finished_at = Column(DateTime)


# =============================================================================
# FACTORS SCHEMA - 팩터/다이렉트 인덱싱
# =============================================================================

class FactorVolumeAnalysis(Base):
    """볼륨 분석 팩터 (factors.volume_analysis)"""
    __tablename__ = "volume_analysis"
    __table_args__ = (
        Index("idx_factor_vol_date", "as_of_date"),
        {"schema": "factors"}
    )
    
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
    __table_args__ = (
        Index("idx_factor_sm_date", "run_date"),
        {"schema": "factors"}
    )
    
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
        {"schema": "factors"}
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
# Database Initialization
# =============================================================================

def create_schemas(engine: Engine):
    """Create all schemas if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS market"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS backtest"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS factors"))
        conn.commit()
    logger.info("✅ Schemas created: market, backtest, factors")


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
    counts = {}
    
    for table in Base.metadata.sorted_tables:
        try:
            schema = table.schema or "public"
            full_name = f"{schema}.{table.name}"
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {full_name}"))
                counts[full_name] = result.scalar()
        except Exception:
            counts[full_name] = 0
    
    return counts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🐘 PostgreSQL Multi-Schema Database Initialization")
    print("=" * 60)
    
    try:
        init_db()
        counts = get_table_counts()
        print("\n📊 Table Row Counts:")
        for table, count in counts.items():
            print(f"   {table}: {count}")
    except Exception as e:
        print(f"❌ Error: {e}")
