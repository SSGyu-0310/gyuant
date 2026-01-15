#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL Database Initialization Script

This script:
1. Creates all tables defined in db_schema_pg.py
2. Loads current S&P 500 tickers from FMP API
3. Loads sample price data for validation
4. Verifies data integrity by comparing counts
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.db_schema_pg import init_db, get_engine, get_session, Ticker, DailyPrice
from utils.db_writer_pg import get_db_writer
from utils.env import load_env
from utils.fmp_client import get_fmp_client

load_env()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_tables() -> bool:
    """Create all database tables."""
    print("\n" + "=" * 60)
    print("📦 Step 1: Creating PostgreSQL Tables")
    print("=" * 60)
    
    try:
        init_db(drop_existing=False)
        print("✅ All tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False


def load_sp500_tickers() -> int:
    """Load current S&P 500 tickers from FMP API."""
    print("\n" + "=" * 60)
    print("📊 Step 2: Loading S&P 500 Tickers")
    print("=" * 60)
    
    try:
        fmp = get_fmp_client()
        writer = get_db_writer()
        
        # Fetch S&P 500 constituents
        constituents = fmp.get_sp500_constituents()
        
        if not constituents:
            print("⚠️ No S&P 500 constituents found. Check FMP_API_KEY.")
            return 0
        
        print(f"   Fetched {len(constituents)} S&P 500 constituents from FMP")
        
        # Transform to ticker records
        ticker_records = []
        for c in constituents:
            ticker_records.append({
                "symbol": c.get("symbol"),
                "name": c.get("name"),
                "sector": c.get("sector"),
                "industry": c.get("subSector"),
                "exchange": c.get("exchange", ""),
                "date_added": c.get("dateFirstAdded"),
                "is_active": True,
                "source": "FMP",
            })
        
        # Bulk insert
        affected = writer.save_tickers(ticker_records, on_conflict="update")
        print(f"✅ Saved {affected} tickers to database")
        
        return affected
        
    except Exception as e:
        print(f"❌ Failed to load tickers: {e}")
        logger.exception("Ticker loading error")
        return 0


def load_sample_prices(sample_tickers: list = None, days: int = 365) -> int:
    """
    Load sample price data for a few tickers.
    
    Args:
        sample_tickers: List of ticker symbols to load. Defaults to top 5.
        days: Number of days of historical data to load.
    
    Returns:
        Number of price records saved.
    """
    print("\n" + "=" * 60)
    print("📈 Step 3: Loading Sample Price Data")
    print("=" * 60)
    
    if sample_tickers is None:
        sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    try:
        fmp = get_fmp_client()
        writer = get_db_writer()
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        total_records = 0
        api_counts = {}
        
        for symbol in sample_tickers:
            print(f"   Fetching {symbol}...", end=" ")
            
            # Fetch historical prices
            data = fmp.historical_price_full(symbol, from_date=start_date, to_date=end_date)
            historical = data.get("historical", [])
            
            if not historical:
                print(f"⚠️ No data")
                continue
            
            # Transform to price records
            price_records = []
            for h in historical:
                price_records.append({
                    "symbol": symbol,
                    "date": h.get("date"),
                    "open": h.get("open"),
                    "high": h.get("high"),
                    "low": h.get("low"),
                    "close": h.get("close"),
                    "adj_close": h.get("adjClose"),
                    "volume": h.get("volume"),
                    "change": h.get("change"),
                    "change_pct": h.get("changePercent"),
                    "source": "FMP",
                })
            
            # Bulk insert
            affected = writer.save_bulk_prices(price_records, on_conflict="update")
            api_counts[symbol] = len(historical)
            total_records += affected
            print(f"✅ {len(historical)} records")
        
        print(f"\n✅ Total: {total_records} price records saved")
        
        return total_records, api_counts
        
    except Exception as e:
        print(f"❌ Failed to load prices: {e}")
        logger.exception("Price loading error")
        return 0, {}


def verify_data_integrity(api_counts: dict) -> bool:
    """
    Verify data integrity by comparing API counts with DB counts.
    
    Args:
        api_counts: Dict of {symbol: count} from API responses
    
    Returns:
        True if all counts match
    """
    print("\n" + "=" * 60)
    print("🔍 Step 4: Verifying Data Integrity")
    print("=" * 60)
    
    try:
        writer = get_db_writer()
        
        # Check ticker count
        ticker_count = writer.get_row_count("tickers")
        print(f"   Tickers in DB: {ticker_count}")
        
        # Check price counts per symbol
        all_match = True
        total_db_prices = 0
        
        for symbol, api_count in api_counts.items():
            # Query DB count for this symbol
            result = writer.execute_raw(
                "SELECT COUNT(*) FROM daily_prices WHERE symbol = %s",
                (symbol,)
            )
            db_count = result[0][0] if result else 0
            total_db_prices += db_count
            
            match = "✅" if db_count == api_count else "⚠️"
            if db_count != api_count:
                all_match = False
            
            print(f"   {symbol}: API={api_count}, DB={db_count} {match}")
        
        # Total price count
        total_price_count = writer.get_row_count("daily_prices")
        print(f"\n   Total prices in DB: {total_price_count}")
        
        if all_match:
            print("\n✅ Data integrity verified - all counts match!")
        else:
            print("\n⚠️ Some counts don't match - please investigate")
        
        return all_match
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        logger.exception("Verification error")
        return False


def print_summary():
    """Print final summary of database contents."""
    print("\n" + "=" * 60)
    print("📋 Database Summary")
    print("=" * 60)
    
    try:
        writer = get_db_writer()
        
        tables = [
            ("tickers", "종목 마스터"),
            ("daily_prices", "일봉 가격"),
            ("financial_statements", "재무제표"),
            ("universe_snapshots", "유니버스 스냅샷"),
            ("bt_runs", "백테스트 실행"),
        ]
        
        for table, desc in tables:
            try:
                count = writer.get_row_count(table)
                print(f"   {table:25} ({desc}): {count:,} rows")
            except Exception:
                print(f"   {table:25} ({desc}): N/A")
        
        writer.close()
        
    except Exception as e:
        print(f"❌ Failed to get summary: {e}")


def main():
    """Main entry point."""
    print("\n")
    print("🐘 PostgreSQL Database Initialization")
    print("=" * 60)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Database: {os.getenv('PG_DATABASE', 'gyuant_market')}")
    print(f"   Host: {os.getenv('PG_HOST', 'localhost')}:{os.getenv('PG_PORT', '5432')}")
    
    # Step 1: Create tables
    if not create_tables():
        print("\n❌ Initialization failed at Step 1")
        return 1
    
    # Step 2: Load S&P 500 tickers
    ticker_count = load_sp500_tickers()
    if ticker_count == 0:
        print("\n⚠️ No tickers loaded - continuing with sample data")
    
    # Step 3: Load sample prices
    price_count, api_counts = load_sample_prices()
    
    # Step 4: Verify integrity
    if api_counts:
        verify_data_integrity(api_counts)
    
    # Print summary
    print_summary()
    
    print("\n" + "=" * 60)
    print("🎉 PostgreSQL initialization complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
