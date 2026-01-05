#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Data Access Layer for Gyuant

Provides abstraction over PostgreSQL and CSV data sources.
Toggle between sources using USE_POSTGRES environment variable.

Usage:
    from utils.data_access import get_prices, get_tickers
    
    prices = get_prices(tickers=['AAPL', 'MSFT'], start_date='2024-01-01')
    tickers = get_tickers()
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Configuration
USE_POSTGRES = os.getenv("USE_POSTGRES", "true").lower() == "true"
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "us_market"))

_PG_ENGINE: Optional[Engine] = None


# =============================================================================
# PostgreSQL Data Access
# =============================================================================

def _normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure both close and current_price exist when one is present."""
    if df.empty:
        return df
    if "current_price" not in df.columns and "close" in df.columns:
        df["current_price"] = df["close"]
    if "close" not in df.columns and "current_price" in df.columns:
        df["close"] = df["current_price"]
    return df

def _pg_get_prices(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: Optional[List[str]] = None,
    lookback_days: int = 500  # Default: load only last 500 days for performance
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Get price data from PostgreSQL market.daily_prices.
    
    Performance optimized:
    - Default lookback_days=500 to avoid loading all 748K rows
    - Uses SQLAlchemy engine with psycopg driver for pandas compatibility
    """
    try:
        engine = _get_pg_engine()
        
        # Default columns for output
        default_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        output_cols = columns if columns else default_cols
        
        # Build SELECT clause - map ticker to symbol
        select_cols = []
        for col in output_cols:
            if col == "ticker":
                select_cols.append("symbol as ticker")
            else:
                select_cols.append(col)
        
        cols_str = ", ".join(select_cols)
        
        # Build WHERE clause with named placeholders for SQLAlchemy
        conditions = []
        params: Dict[str, Any] = {}
        
        if tickers:
            ticker_keys = []
            for idx, ticker in enumerate(tickers):
                key = f"ticker_{idx}"
                ticker_keys.append(f":{key}")
                params[key] = ticker
            conditions.append(f"symbol IN ({', '.join(ticker_keys)})")
        
        # Apply date filter (use provided start_date or default lookback)
        if start_date:
            conditions.append("date >= :start_date")
            params["start_date"] = start_date
        elif lookback_days > 0:
            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            conditions.append("date >= :start_date")
            params["start_date"] = cutoff
        
        if end_date:
            conditions.append("date <= :end_date")
            params["end_date"] = end_date
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"SELECT {cols_str} FROM market.daily_prices WHERE {where_clause} ORDER BY symbol, date"
        
        # Use pandas read_sql with SQLAlchemy engine (officially supported)
        df = pd.read_sql(text(query), engine, params=params if params else None)
        
        logger.debug(f"Loaded {len(df)} rows from PostgreSQL")
        
        return df, None
        
    except Exception as e:
        logger.error(f"PostgreSQL price query failed: {e}")
        return pd.DataFrame(), str(e)


def _get_pg_engine() -> Engine:
    """Create or return cached SQLAlchemy engine for PostgreSQL."""
    global _PG_ENGINE
    if _PG_ENGINE is None:
        pg_host = os.getenv("PG_HOST", "localhost")
        pg_port = os.getenv("PG_PORT", "5432")
        pg_db = os.getenv("PG_DATABASE", "gyuant_market")
        pg_user = os.getenv("PG_USER", "postgres")
        pg_pass = os.getenv("PG_PASSWORD", "")
        url = f"postgresql+psycopg://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        _PG_ENGINE = create_engine(url, pool_pre_ping=True)
    return _PG_ENGINE


def _pg_get_tickers(active_only: bool = True) -> pd.DataFrame:
    """Get ticker list from PostgreSQL market.tickers."""
    try:
        from utils.db_writer_pg import get_db_writer
        
        writer = get_db_writer()
        
        query = "SELECT symbol as ticker, name, sector, industry FROM market.tickers"
        if active_only:
            query += " WHERE is_active = true"
        query += " ORDER BY symbol"
        
        result = writer.execute_raw(query)
        
        if not result:
            return pd.DataFrame()
        
        df = pd.DataFrame(result, columns=["ticker", "name", "sector", "industry"])
        
        return df
        
    except Exception as e:
        logger.error(f"PostgreSQL ticker query failed: {e}")
        return pd.DataFrame()


def _pg_get_latest_prices(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Get latest price for each ticker from PostgreSQL."""
    try:
        from utils.db_writer_pg import get_db_writer
        
        writer = get_db_writer()
        
        if tickers:
            placeholders = ", ".join(["%s"] * len(tickers))
            query = f"""
                SELECT DISTINCT ON (symbol) 
                    symbol as ticker, date, close, volume
                FROM market.daily_prices
                WHERE symbol IN ({placeholders})
                ORDER BY symbol, date DESC
            """
            result = writer.execute_raw(query, tuple(tickers))
        else:
            query = """
                SELECT DISTINCT ON (symbol) 
                    symbol as ticker, date, close, volume
                FROM market.daily_prices
                ORDER BY symbol, date DESC
            """
            result = writer.execute_raw(query)
        
        if not result:
            return pd.DataFrame()
        
        df = pd.DataFrame(result, columns=["ticker", "date", "close", "volume"])
        
        return df
        
    except Exception as e:
        logger.error(f"PostgreSQL latest price query failed: {e}")
        return pd.DataFrame()


def _pg_get_volume_analysis(as_of_date: Optional[str] = None) -> pd.DataFrame:
    """Get volume analysis from PostgreSQL factors.volume_analysis."""
    try:
        from utils.db_writer_pg import get_db_writer
        
        writer = get_db_writer()
        
        if as_of_date:
            query = """
                SELECT ticker, as_of_date, name, supply_demand_score, supply_demand_stage,
                       obv_change_20d, mfi, vol_ratio_5d_20d
                FROM factors.volume_analysis
                WHERE as_of_date = %s
                ORDER BY supply_demand_score DESC
            """
            result = writer.execute_raw(query, (as_of_date,))
        else:
            # Get latest
            query = """
                SELECT DISTINCT ON (ticker) 
                    ticker, as_of_date, name, supply_demand_score, supply_demand_stage,
                    obv_change_20d, mfi, vol_ratio_5d_20d
                FROM factors.volume_analysis
                ORDER BY ticker, as_of_date DESC
            """
            result = writer.execute_raw(query)
        
        if not result:
            return pd.DataFrame()
        
        df = pd.DataFrame(result, columns=[
            "ticker", "as_of_date", "name", "supply_demand_score", "supply_demand_stage",
            "obv_change_20d", "mfi", "vol_ratio_5d_20d"
        ])
        
        return df
        
    except Exception as e:
        logger.error(f"PostgreSQL volume analysis query failed: {e}")
        return pd.DataFrame()


# =============================================================================
# CSV Data Access (Fallback)
# =============================================================================

def _csv_get_prices(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Get price data from us_daily_prices.csv."""
    csv_path = DATA_DIR / "us_daily_prices.csv"
    
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        if columns:
            requested = set(columns)
            if "close" in requested or "current_price" in requested:
                requested.update(["close", "current_price"])
            usecols = list(requested)
        else:
            usecols = ["ticker", "date", "open", "high", "low", "close", "current_price", "volume"]
        df = pd.read_csv(csv_path, usecols=lambda c: c in usecols + ["ticker", "date"])
        
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        
        if start_date:
            df = df[df["date"] >= start_date]
        
        if end_date:
            df = df[df["date"] <= end_date]
        
        return _normalize_price_columns(df).sort_values(["ticker", "date"])
        
    except Exception as e:
        logger.error(f"CSV price read failed: {e}")
        return pd.DataFrame()


def _csv_get_tickers(active_only: bool = True) -> pd.DataFrame:
    """Get ticker list from us_stocks_list.csv."""
    csv_path = DATA_DIR / "us_stocks_list.csv"
    
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"CSV ticker read failed: {e}")
        return pd.DataFrame()


def _csv_get_latest_prices(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Get latest price for each ticker from CSV."""
    df = _csv_get_prices(tickers=tickers, columns=["ticker", "date", "close", "volume"])
    
    if df.empty:
        return df
    
    df = df.sort_values("date").groupby("ticker").last().reset_index()
    return df[["ticker", "date", "close", "volume"]]


def _csv_get_volume_analysis(as_of_date: Optional[str] = None) -> pd.DataFrame:
    """Get volume analysis from us_volume_analysis.csv."""
    csv_path = DATA_DIR / "us_volume_analysis.csv"
    
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"CSV volume analysis read failed: {e}")
        return pd.DataFrame()


# =============================================================================
# Public API
# =============================================================================

def get_prices(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: Optional[List[str]] = None,
    lookback_days: int = 500,
    return_meta: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str, Optional[str]]]:
    """
    Get historical price data.
    
    Args:
        tickers: List of ticker symbols (None = all)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        columns: Columns to return
        lookback_days: Default days to load (0 = all data)
        return_meta: When True, return (df, source, fallback_reason)
    
    Returns:
        DataFrame with price data (or tuple when return_meta=True)
    """
    fallback_reason = None
    if USE_POSTGRES:
        df, pg_error = _pg_get_prices(tickers, start_date, end_date, columns, lookback_days)
        if not df.empty:
            df = _normalize_price_columns(df)
            if return_meta:
                return df, "postgres", None
            return df
        if pg_error:
            fallback_reason = f"PostgreSQL failed: {pg_error}"
            logger.warning(f"{fallback_reason}. Falling back to CSV.")
        else:
            fallback_reason = "PostgreSQL returned 0 rows"
            logger.info("PostgreSQL returned 0 rows, falling back to CSV")
    
    df = _csv_get_prices(tickers, start_date, end_date, columns)
    df = _normalize_price_columns(df)
    if return_meta:
        return df, "csv", fallback_reason
    return df


def get_tickers(active_only: bool = True) -> pd.DataFrame:
    """
    Get ticker/stock list.
    
    Args:
        active_only: Only return active stocks
    
    Returns:
        DataFrame with ticker info
    """
    if USE_POSTGRES:
        df = _pg_get_tickers(active_only)
        if not df.empty:
            return df
        logger.info("PostgreSQL empty, falling back to CSV")
    
    return _csv_get_tickers(active_only)


def get_latest_prices(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get latest price for each ticker.
    
    Args:
        tickers: List of ticker symbols (None = all)
    
    Returns:
        DataFrame with latest prices
    """
    if USE_POSTGRES:
        df = _pg_get_latest_prices(tickers)
        if not df.empty:
            return df
        logger.info("PostgreSQL empty, falling back to CSV")
    
    return _csv_get_latest_prices(tickers)


def get_volume_analysis(as_of_date: Optional[str] = None) -> pd.DataFrame:
    """
    Get volume analysis data.
    
    Args:
        as_of_date: Date for analysis (None = latest)
    
    Returns:
        DataFrame with volume analysis
    """
    if USE_POSTGRES:
        df = _pg_get_volume_analysis(as_of_date)
        if not df.empty:
            return df
        logger.info("PostgreSQL empty, falling back to CSV")
    
    return _csv_get_volume_analysis(as_of_date)


def get_data_source() -> str:
    """Return current data source (postgres or csv)."""
    return "postgres" if USE_POSTGRES else "csv"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print(f"📊 Data Access Layer Test")
    print(f"   Source: {get_data_source()}")
    print("=" * 50)
    
    # Test tickers
    tickers = get_tickers()
    print(f"   Tickers: {len(tickers)} rows")
    
    # Test prices
    prices = get_prices(tickers=["AAPL", "MSFT"], start_date="2024-01-01")
    print(f"   Prices (AAPL, MSFT from 2024): {len(prices)} rows")
    
    # Test latest
    latest = get_latest_prices(tickers=["AAPL", "MSFT"])
    print(f"   Latest prices: {len(latest)} rows")
    if not latest.empty:
        print(latest)
    
    # Test volume analysis
    vol = get_volume_analysis()
    print(f"   Volume analysis: {len(vol)} rows")
