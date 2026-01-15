#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Data Loader (DAO)
Point-in-Time data access for backtesting without look-ahead bias
Aligned with docs/db/schema.sql - uses bt_* tables
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd

# Add parent directories to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.db_schema import get_connection, get_db_path

logger = logging.getLogger(__name__)


class BacktestDataLoader:
    """
    Data loader for backtesting with Point-in-Time queries.
    Uses bt_* tables to prevent look-ahead bias.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
    
    def get_prices_as_of(
        self,
        as_of_date: str,
        ticker_list: Optional[List[str]] = None,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Get price data as of a specific date (no future data).
        Uses bt_prices_daily table.
        """
        try:
            conn = get_connection(self.db_path)
            
            if ticker_list:
                placeholders = ','.join(['?' for _ in ticker_list])
                query = f"""
                    SELECT ticker, date, open, high, low, close, volume
                    FROM bt_prices_daily
                    WHERE date <= ?
                      AND date >= date(?, '-{lookback_days} days')
                      AND ticker IN ({placeholders})
                    ORDER BY ticker, date
                """
                params = [as_of_date, as_of_date] + ticker_list
            else:
                query = f"""
                    SELECT ticker, date, open, high, low, close, volume
                    FROM bt_prices_daily
                    WHERE date <= ?
                      AND date >= date(?, '-{lookback_days} days')
                    ORDER BY ticker, date
                """
                params = [as_of_date, as_of_date]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            logger.debug(f"Retrieved {len(df)} price records as of {as_of_date}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get prices as of {as_of_date}: {e}")
            return pd.DataFrame()
    
    def get_latest_price(
        self,
        as_of_date: str,
        ticker_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get the latest available price for each ticker as of a date."""
        try:
            conn = get_connection(self.db_path)
            
            if ticker_list:
                placeholders = ','.join(['?' for _ in ticker_list])
                query = f"""
                    SELECT ticker, date, close, volume
                    FROM bt_prices_daily
                    WHERE (ticker, date) IN (
                        SELECT ticker, MAX(date) as date
                        FROM bt_prices_daily
                        WHERE date <= ?
                          AND ticker IN ({placeholders})
                        GROUP BY ticker
                    )
                """
                params = [as_of_date] + ticker_list
            else:
                query = """
                    SELECT ticker, date, close, volume
                    FROM bt_prices_daily
                    WHERE (ticker, date) IN (
                        SELECT ticker, MAX(date) as date
                        FROM bt_prices_daily
                        WHERE date <= ?
                        GROUP BY ticker
                    )
                """
                params = [as_of_date]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Failed to get latest prices: {e}")
            return pd.DataFrame()
    
    def get_universe_as_of(self, as_of_date: str) -> pd.DataFrame:
        """
        Get the universe snapshot as of a specific date.
        Uses bt_universe_snapshot table.
        """
        try:
            conn = get_connection(self.db_path)
            
            # Find closest snapshot date
            query = """
                SELECT MAX(as_of_date) as snapshot_date
                FROM bt_universe_snapshot
                WHERE as_of_date <= ?
            """
            cursor = conn.execute(query, [as_of_date])
            result = cursor.fetchone()
            snapshot_date = result[0] if result else None
            
            if not snapshot_date:
                logger.warning(f"No universe snapshot found for {as_of_date}")
                conn.close()
                return pd.DataFrame()
            
            query = """
                SELECT as_of_date, ticker, name, sector, market
                FROM bt_universe_snapshot
                WHERE as_of_date = ?
                ORDER BY ticker
            """
            df = pd.read_sql_query(query, conn, params=[snapshot_date])
            conn.close()
            
            logger.debug(f"Retrieved universe snapshot: {snapshot_date} ({len(df)} tickers)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get universe as of {as_of_date}: {e}")
            return pd.DataFrame()
    
    def get_signals_as_of(
        self,
        as_of_date: str,
        signal_id: str,
        signal_version: str = "1.0",
        ticker_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get signals as of a specific date from bt_signals."""
        try:
            conn = get_connection(self.db_path)
            
            if ticker_list:
                placeholders = ','.join(['?' for _ in ticker_list])
                query = f"""
                    SELECT as_of_date, ticker, signal_value, signal_rank
                    FROM bt_signals
                    WHERE as_of_date <= ?
                      AND signal_id = ?
                      AND signal_version = ?
                      AND ticker IN ({placeholders})
                    ORDER BY as_of_date DESC, signal_rank ASC
                """
                params = [as_of_date, signal_id, signal_version] + ticker_list
            else:
                query = """
                    SELECT as_of_date, ticker, signal_value, signal_rank
                    FROM bt_signals
                    WHERE as_of_date <= ?
                      AND signal_id = ?
                      AND signal_version = ?
                    ORDER BY as_of_date DESC, signal_rank ASC
                """
                params = [as_of_date, signal_id, signal_version]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return pd.DataFrame()
    
    def get_market_prices(
        self,
        start_date: str,
        end_date: str,
        ticker_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get operational market prices (for Flask API etc)."""
        try:
            conn = get_connection(self.db_path)
            
            if ticker_list:
                placeholders = ','.join(['?' for _ in ticker_list])
                query = f"""
                    SELECT ticker, date, open, high, low, close, volume, change, change_rate
                    FROM market_prices_daily
                    WHERE date BETWEEN ? AND ?
                      AND ticker IN ({placeholders})
                    ORDER BY ticker, date
                """
                params = [start_date, end_date] + ticker_list
            else:
                query = """
                    SELECT ticker, date, open, high, low, close, volume, change, change_rate
                    FROM market_prices_daily
                    WHERE date BETWEEN ? AND ?
                    ORDER BY ticker, date
                """
                params = [start_date, end_date]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market prices: {e}")
            return pd.DataFrame()


# Convenience functions
def get_prices_as_of(date: str, ticker_list: Optional[List[str]] = None) -> pd.DataFrame:
    """Convenience function for getting backtest prices as of a date."""
    loader = BacktestDataLoader()
    return loader.get_prices_as_of(date, ticker_list)


def get_universe_as_of(date: str) -> pd.DataFrame:
    """Convenience function for getting universe as of a date."""
    loader = BacktestDataLoader()
    return loader.get_universe_as_of(date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    loader = BacktestDataLoader()
    
    print("📊 Backtest Data Loader Demo")
    print("=" * 50)
    print(f"Database: {loader.db_path}")
    
    test_date = "2024-12-20"
    print(f"\n🔍 Querying data as of {test_date}...")
    
    universe = loader.get_universe_as_of(test_date)
    print(f"   Universe: {len(universe)} tickers")
    
    if not universe.empty:
        sample_tickers = universe['ticker'].head(5).tolist()
        prices = loader.get_prices_as_of(test_date, sample_tickers, lookback_days=30)
        print(f"   Prices: {len(prices)} records for {sample_tickers}")
