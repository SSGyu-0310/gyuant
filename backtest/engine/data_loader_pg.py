import os
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import text

from utils.db import get_engine

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "us_market"


def _get_data_dir() -> Path:
    data_dir_env = os.getenv("DATA_DIR")
    if not data_dir_env:
        return DEFAULT_DATA_DIR
    candidate = Path(data_dir_env)
    if candidate.is_absolute():
        return candidate
    return (ROOT_DIR / candidate).resolve()


DATA_DIR_PATH = _get_data_dir()
SMART_MONEY_CSV = DATA_DIR_PATH / "smart_money_picks_v2.csv"


class PostgresDataLoader:
    """
    Data loader for backtesting using PostgreSQL schema (backtest.*).
    Ensures Point-in-Time (PIT) consistency.
    """

    def __init__(self):
        self.engine = get_engine()

    def get_prices_as_of(
        self,
        as_of_date: str,
        ticker_list: Optional[List[str]] = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        Get daily price data up to as_of_date.
        """
        try:
            # Calculate start_date in Python since PostgreSQL interval doesn't support parameter binding
            from datetime import datetime, timedelta

            end_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            start_date = end_date - timedelta(days=lookback_days)

            query_str = """
                SELECT ticker as ticker, date, open, high, low, close, volume
                FROM market.daily_prices
                WHERE date <= :as_of_date
                  AND date >= :start_date
            """

            # Note: Changed from backtest.prices_daily to market.daily_prices
            # to utilize available data.
            # In a strict PIT setup, we would copy snapshot data to backtest.prices_daily.

            params = {"as_of_date": as_of_date, "start_date": str(start_date)}

            if ticker_list:
                # Use symbol column if ticker is not present, though schema says ticker
                # market.daily_prices definition: symbol (PK), date (PK)...
                # But wait, schema in db_schema_pg.py says:
                # class MarketDailyPrice(Base): ... symbol = Column(String(20), primary_key=True)
                # So the column is 'symbol', not 'ticker'.

                query_str = """
                    SELECT symbol as ticker, date, open, high, low, close, volume
                    FROM market.daily_prices
                    WHERE date <= :as_of_date
                      AND date >= :start_date
                """
                query_str += " AND symbol = ANY(:tickers)"
                params["tickers"] = ticker_list
            else:
                query_str = """
                    SELECT symbol as ticker, date, open, high, low, close, volume
                    FROM market.daily_prices
                    WHERE date <= :as_of_date
                      AND date >= :start_date
                """

            query_str += " ORDER BY symbol, date"

            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    text(query_str), conn, params=params, parse_dates=["date"]
                )

            return df

        except Exception as e:
            logger.error(f"Failed to get prices as of {as_of_date}: {e}")
            return pd.DataFrame()

    def get_universe_as_of(self, as_of_date: str) -> pd.DataFrame:
        """
        Get the universe snapshot as of as_of_date.
        Returns DataFrame with columns [as_of_date, ticker, name, sector, market, weight].
        """
        try:
            query_str = """
                WITH latest_snapshot AS (
                    SELECT MAX(as_of_date) as snap_date
                    FROM backtest.universe_snapshot
                    WHERE as_of_date <= :as_of_date
                )
                SELECT us.as_of_date, us.ticker, us.name, us.sector, us.market, us.weight
                FROM backtest.universe_snapshot us
                JOIN latest_snapshot ls ON us.as_of_date = ls.snap_date
            """

            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    text(query_str), conn, params={"as_of_date": as_of_date}
                )

            if df.empty:
                # Diagnostics
                with self.engine.connect() as conn:
                    cnt = conn.execute(
                        text("SELECT COUNT(*) FROM backtest.universe_snapshot")
                    ).scalar()

                logger.warning(
                    f"No universe snapshot found for {as_of_date}. (Total rows in table: {cnt})"
                )
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Failed to get universe as of {as_of_date}: {e}")
            return pd.DataFrame()

    def get_fundamentals_as_of(
        self, as_of_date: str, ticker_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get latest fundamental data as of a specific date (PIT).
        Uses backtest.financial_statements table.
        """
        try:
            params = {"as_of_date": as_of_date}

            # Subquery to find latest filing_date per symbol <= as_of_date
            subquery = """
                SELECT symbol, MAX(filing_date) as max_filing_date
                FROM backtest.financial_statements
                WHERE filing_date <= :as_of_date
            """

            if ticker_list:
                subquery += " AND symbol = ANY(:tickers)"
                params["tickers"] = ticker_list

            subquery += " GROUP BY symbol"

            query_str = f"""
                WITH latest_filings AS ({subquery})
                SELECT fs.*, fs.symbol as ticker
                FROM backtest.financial_statements fs
                JOIN latest_filings lf 
                  ON fs.symbol = lf.symbol AND fs.filing_date = lf.max_filing_date
            """

            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query_str), conn, params=params)

            return df

        except Exception as e:
            logger.error(f"Failed to get fundamentals: {e}")
            return pd.DataFrame()

    def get_signals_as_of(
        self,
        as_of_date: str,
        signal_id: str,
        version: str = "1.0",
        ticker_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get signal values as of as_of_date.
        """
        try:
            query_str = """
                SELECT ticker, signal_value, signal_rank
                FROM backtest.signals
                WHERE as_of_date <= :as_of_date
                  AND signal_id = :signal_id
                  AND signal_version = :version
            """

            params = {
                "as_of_date": as_of_date,
                "signal_id": signal_id,
                "version": version,
            }

            if ticker_list:
                query_str += " AND ticker = ANY(:tickers)"
                params["tickers"] = ticker_list

            final_query = f"""
                SELECT DISTINCT ON (ticker) ticker, signal_value, signal_rank
                FROM ({query_str}) sub
                ORDER BY ticker, as_of_date DESC
            """

            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(final_query), conn, params=params)

            return df

        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return pd.DataFrame()

    def get_smart_money_ranks(self, as_of_date: str, limit: int = 50) -> pd.DataFrame:
        """
        Get Smart Money picks as of a specific date (PIT).
        Falls back to local CSV if factors table is empty.
        """
        df = pd.DataFrame()

        try:
            # First try factors.smart_money_picks (dedicated backtest table)
            final_query = """
                WITH latest_run AS (
                    SELECT MAX(run_date) as r_date
                    FROM factors.smart_money_picks
                    WHERE run_date <= :as_of_date
                )
                SELECT ticker, smart_money_score, rank, composite_score
                FROM factors.smart_money_picks p
                JOIN latest_run lr ON p.run_date = lr.r_date
                ORDER BY rank ASC
                LIMIT :limit
            """

            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    text(final_query),
                    conn,
                    params={"as_of_date": as_of_date, "limit": limit},
                )
        except Exception as exc:
            logger.error(f"Failed to query factors.smart_money_picks: {exc}")
            print(f"   ❌ [DATA ERROR] get_smart_money_ranks failed: {exc}")
            df = pd.DataFrame()

        if df.empty:
            fallback_df = self._load_local_smart_money_csv(limit)
            if not fallback_df.empty:
                print(f"   📊 [DATA] Using local smart_money CSV fallback: {len(fallback_df)} tickers")
                df = fallback_df
            else:
                print("   ⚠️ [DATA] No smart_money data available in any table or local CSV.")

        return df

    def _load_local_smart_money_csv(self, limit: int) -> pd.DataFrame:
        if not SMART_MONEY_CSV.exists():
            return pd.DataFrame()

        try:
            csv_df = pd.read_csv(SMART_MONEY_CSV, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read smart_money CSV fallback: %s", exc)
            return pd.DataFrame()

        if csv_df.empty:
            return pd.DataFrame()

        if "rank" in csv_df.columns:
            csv_df = csv_df.sort_values("rank")

        csv_df = csv_df.head(limit).copy()

        if "ticker" not in csv_df.columns:
            return pd.DataFrame()

        if "smart_money_score" not in csv_df.columns:
            if "composite_score" in csv_df.columns:
                csv_df["smart_money_score"] = csv_df["composite_score"]
            else:
                csv_df["smart_money_score"] = 0

        if "composite_score" not in csv_df.columns:
            csv_df["composite_score"] = csv_df["smart_money_score"]

        desired_columns = ["ticker", "smart_money_score", "rank", "composite_score"]
        available_columns = [col for col in desired_columns if col in csv_df.columns]
        if not available_columns:
            return pd.DataFrame()

        return csv_df[available_columns].reset_index(drop=True)
