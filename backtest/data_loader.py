#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Data Loader (DAO)
Point-in-Time data access for backtesting without look-ahead bias
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
    sys.path.append(str(ROOT_DIR))

from backtest.db_schema import get_connection, get_db_path

logger = logging.getLogger(__name__)


class BacktestDataLoader:
    """
    Data loader for backtesting with Point-in-Time queries.
    Prevents look-ahead bias by only returning data available as of the query date.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            db_path: Path to SQLite database (uses default if None)
        """
        self.db_path = db_path or get_db_path()
    
    def get_prices_as_of(
        self,
        as_of_date: str,
        ticker_list: Optional[List[str]] = None,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Get price data as of a specific date (no future data).
        
        Args:
            as_of_date: Reference date (YYYY-MM-DD), only data <= this date
            ticker_list: List of tickers to query (all if None)
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with price data
        """
        try:
            conn = get_connection(self.db_path)
            
            # Build query
            if ticker_list:
                placeholders = ','.join(['?' for _ in ticker_list])
                query = f"""
                    SELECT ticker, date, open, high, low, close, volume, change, change_rate
                    FROM daily_prices
                    WHERE date <= ?
                      AND date >= date(?, '-{lookback_days} days')
                      AND ticker IN ({placeholders})
                    ORDER BY ticker, date
                """
                params = [as_of_date, as_of_date] + ticker_list
            else:
                query = f"""
                    SELECT ticker, date, open, high, low, close, volume, change, change_rate
                    FROM daily_prices
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
        """
        Get the latest available price for each ticker as of a date.
        
        Args:
            as_of_date: Reference date
            ticker_list: List of tickers to query
            
        Returns:
            DataFrame with latest price per ticker
        """
        try:
            conn = get_connection(self.db_path)
            
            if ticker_list:
                placeholders = ','.join(['?' for _ in ticker_list])
                query = f"""
                    SELECT ticker, date, close, volume
                    FROM daily_prices
                    WHERE (ticker, date) IN (
                        SELECT ticker, MAX(date) as date
                        FROM daily_prices
                        WHERE date <= ?
                          AND ticker IN ({placeholders})
                        GROUP BY ticker
                    )
                """
                params = [as_of_date] + ticker_list
            else:
                query = """
                    SELECT ticker, date, close, volume
                    FROM daily_prices
                    WHERE (ticker, date) IN (
                        SELECT ticker, MAX(date) as date
                        FROM daily_prices
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
        Returns the closest available snapshot <= as_of_date.
        
        Args:
            as_of_date: Reference date (YYYY-MM-DD)
            
        Returns:
            DataFrame with universe members
        """
        try:
            conn = get_connection(self.db_path)
            
            # Find closest snapshot date
            query = """
                SELECT MAX(date) as snapshot_date
                FROM universe_snapshots
                WHERE date <= ?
            """
            cursor = conn.execute(query, [as_of_date])
            result = cursor.fetchone()
            snapshot_date = result[0] if result else None
            
            if not snapshot_date:
                logger.warning(f"No universe snapshot found for {as_of_date}")
                conn.close()
                return pd.DataFrame()
            
            # Get snapshot data
            query = """
                SELECT date, ticker, name, sector, market
                FROM universe_snapshots
                WHERE date = ?
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
        ticker_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get signals as of a specific date.
        
        Args:
            as_of_date: Reference date
            signal_id: Signal identifier
            ticker_list: Optional ticker filter
            
        Returns:
            DataFrame with signals
        """
        try:
            conn = get_connection(self.db_path)
            
            if ticker_list:
                placeholders = ','.join(['?' for _ in ticker_list])
                query = f"""
                    SELECT as_of_date, ticker, signal_value, signal_version
                    FROM signals
                    WHERE as_of_date <= ?
                      AND signal_id = ?
                      AND ticker IN ({placeholders})
                    ORDER BY as_of_date DESC, ticker
                """
                params = [as_of_date, signal_id] + ticker_list
            else:
                query = """
                    SELECT as_of_date, ticker, signal_value, signal_version
                    FROM signals
                    WHERE as_of_date <= ?
                      AND signal_id = ?
                    ORDER BY as_of_date DESC, ticker
                """
                params = [as_of_date, signal_id]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return pd.DataFrame()
    
    def save_signal(
        self,
        as_of_date: str,
        ticker: str,
        signal_id: str,
        signal_value: float,
        signal_version: str = "1.0"
    ) -> bool:
        """
        Save a signal to the database.
        
        Args:
            as_of_date: Signal calculation date
            ticker: Stock ticker
            signal_id: Signal identifier
            signal_value: Signal value
            signal_version: Version string
            
        Returns:
            True if successful
        """
        try:
            conn = get_connection(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO signals
                (as_of_date, ticker, signal_id, signal_value, signal_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (as_of_date, ticker, signal_id, signal_value, signal_version, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return False


# Convenience functions
def get_prices_as_of(date: str, ticker_list: Optional[List[str]] = None) -> pd.DataFrame:
    """Convenience function for getting prices as of a date."""
    loader = BacktestDataLoader()
    return loader.get_prices_as_of(date, ticker_list)


def get_universe_as_of(date: str) -> pd.DataFrame:
    """Convenience function for getting universe as of a date."""
    loader = BacktestDataLoader()
    return loader.get_universe_as_of(date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Demo usage
    loader = BacktestDataLoader()
    
    print("📊 Backtest Data Loader Demo")
    print("=" * 50)
    
    # Test point-in-time query
    test_date = "2024-12-20"
    print(f"\n🔍 Querying data as of {test_date}...")
    
    # Get universe
    universe = loader.get_universe_as_of(test_date)
    print(f"   Universe: {len(universe)} tickers")
    
    # Get prices for a few tickers
    if not universe.empty:
        sample_tickers = universe['ticker'].head(5).tolist()
        prices = loader.get_prices_as_of(test_date, sample_tickers, lookback_days=30)
        print(f"   Prices: {len(prices)} records for {sample_tickers}")
