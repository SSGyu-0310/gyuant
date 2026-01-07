#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to PostgreSQL Migration Script

Migrates local CSV data to PostgreSQL multi-schema database:
- us_stocks_list.csv → market.tickers
- us_daily_prices.csv → market.daily_prices
- us_volume_analysis.csv → factors.volume_analysis
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.db_writer_pg import get_db_writer
from utils.env import load_env

load_env()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_stocks_list(data_dir: Path) -> int:
    """Migrate us_stocks_list.csv to market.tickers"""
    csv_path = data_dir / "us_stocks_list.csv"
    
    if not csv_path.exists():
        logger.warning(f"⚠️ {csv_path} not found, skipping")
        return 0
    
    print(f"\n📊 Migrating {csv_path.name}...")
    
    df = pd.read_csv(csv_path)
    logger.info(f"   Read {len(df)} rows from CSV")
    
    # Transform to ticker records
    records = []
    for _, row in df.iterrows():
        records.append({
            "symbol": row.get("ticker"),
            "name": row.get("name"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "exchange": row.get("exchange"),
            "market_cap": row.get("market_cap") or row.get("marketCap"),
            "is_active": True,
            "source": "CSV",
        })
    
    writer = get_db_writer()
    
    # Use raw SQL for schema-qualified table
    query = """
        INSERT INTO market.tickers (symbol, name, sector, industry, exchange, market_cap, is_active, source, updated_at)
        VALUES (%(symbol)s, %(name)s, %(sector)s, %(industry)s, %(exchange)s, %(market_cap)s, %(is_active)s, %(source)s, NOW())
        ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            exchange = EXCLUDED.exchange,
            market_cap = EXCLUDED.market_cap,
            is_active = EXCLUDED.is_active,
            source = EXCLUDED.source,
            updated_at = NOW()
    """
    
    affected = writer._execute_batch(query, records)
    logger.info(f"   ✅ Saved {affected} tickers to market.tickers")
    
    return affected


def migrate_daily_prices(data_dir: Path, batch_size: int = 50000) -> int:
    """Migrate us_daily_prices.csv to market.daily_prices"""
    csv_path = data_dir / "us_daily_prices.csv"
    
    if not csv_path.exists():
        logger.warning(f"⚠️ {csv_path} not found, skipping")
        return 0
    
    print(f"\n📈 Migrating {csv_path.name} (this may take a few minutes)...")
    
    # Read CSV in chunks for large files
    total_rows = 0
    writer = get_db_writer()
    
    query = """
        INSERT INTO market.daily_prices (symbol, date, open, high, low, close, adj_close, volume, change, change_pct, source, ingested_at)
        VALUES (%(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(adj_close)s, %(volume)s, %(change)s, %(change_pct)s, %(source)s, NOW())
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            change = EXCLUDED.change,
            change_pct = EXCLUDED.change_pct,
            ingested_at = NOW()
    """
    
    chunk_num = 0
    for chunk in pd.read_csv(csv_path, chunksize=batch_size):
        chunk_num += 1
        
        records = []
        for _, row in chunk.iterrows():
            # Handle date
            date_val = row.get("date")
            if pd.isna(date_val):
                continue
            
            records.append({
                "symbol": row.get("ticker"),
                "date": str(date_val)[:10],  # YYYY-MM-DD
                "open": row.get("open") if pd.notna(row.get("open")) else None,
                "high": row.get("high") if pd.notna(row.get("high")) else None,
                "low": row.get("low") if pd.notna(row.get("low")) else None,
                # CSV uses 'current_price' column, map to 'close'
                "close": row.get("current_price") if pd.notna(row.get("current_price")) else None,
                "adj_close": row.get("current_price") if pd.notna(row.get("current_price")) else None,
                "volume": int(row.get("volume")) if pd.notna(row.get("volume")) else None,
                "change": row.get("change") if pd.notna(row.get("change")) else None,
                "change_pct": row.get("change_rate") if pd.notna(row.get("change_rate")) else None,
                "source": "CSV",
            })
        
        if records:
            affected = writer._execute_batch(query, records)
            total_rows += affected
            print(f"   Chunk {chunk_num}: {affected} rows (total: {total_rows:,})", end="\r")
    
    print(f"\n   ✅ Saved {total_rows:,} rows to market.daily_prices")
    
    return total_rows


def migrate_volume_analysis(data_dir: Path) -> int:
    """Migrate us_volume_analysis.csv to factors.volume_analysis"""
    csv_path = data_dir / "us_volume_analysis.csv"
    
    if not csv_path.exists():
        logger.warning(f"⚠️ {csv_path} not found, skipping")
        return 0
    
    print(f"\n📊 Migrating {csv_path.name}...")
    
    df = pd.read_csv(csv_path)
    logger.info(f"   Read {len(df)} rows from CSV")
    
    # Get latest date from data or use today
    today = datetime.now().strftime("%Y-%m-%d")
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "ticker": row.get("ticker"),
            "as_of_date": today,
            "name": row.get("name"),
            "obv": row.get("obv") if pd.notna(row.get("obv")) else None,
            "obv_change_20d": row.get("obv_change_20d") if pd.notna(row.get("obv_change_20d")) else None,
            "ad_line": row.get("ad_line") if pd.notna(row.get("ad_line")) else None,
            "ad_change_20d": row.get("ad_change_20d") if pd.notna(row.get("ad_change_20d")) else None,
            "mfi": row.get("mfi") if pd.notna(row.get("mfi")) else None,
            "vol_ratio_5d_20d": row.get("vol_ratio_5d_20d") if pd.notna(row.get("vol_ratio_5d_20d")) else None,
            "surge_count_5d": int(row.get("surge_count_5d")) if pd.notna(row.get("surge_count_5d")) else None,
            "surge_count_20d": int(row.get("surge_count_20d")) if pd.notna(row.get("surge_count_20d")) else None,
            "supply_demand_score": row.get("supply_demand_score") if pd.notna(row.get("supply_demand_score")) else None,
            "supply_demand_stage": row.get("supply_demand_stage"),
            "source": "CSV",
        })
    
    writer = get_db_writer()
    
    query = """
        INSERT INTO factors.volume_analysis (
            ticker, as_of_date, name, obv, obv_change_20d, ad_line, ad_change_20d,
            mfi, vol_ratio_5d_20d, surge_count_5d, surge_count_20d,
            supply_demand_score, supply_demand_stage, source, ingested_at
        ) VALUES (
            %(ticker)s, %(as_of_date)s, %(name)s, %(obv)s, %(obv_change_20d)s, %(ad_line)s, %(ad_change_20d)s,
            %(mfi)s, %(vol_ratio_5d_20d)s, %(surge_count_5d)s, %(surge_count_20d)s,
            %(supply_demand_score)s, %(supply_demand_stage)s, %(source)s, NOW()
        )
        ON CONFLICT (ticker, as_of_date) DO UPDATE SET
            name = EXCLUDED.name,
            obv = EXCLUDED.obv,
            obv_change_20d = EXCLUDED.obv_change_20d,
            ad_line = EXCLUDED.ad_line,
            ad_change_20d = EXCLUDED.ad_change_20d,
            mfi = EXCLUDED.mfi,
            vol_ratio_5d_20d = EXCLUDED.vol_ratio_5d_20d,
            surge_count_5d = EXCLUDED.surge_count_5d,
            surge_count_20d = EXCLUDED.surge_count_20d,
            supply_demand_score = EXCLUDED.supply_demand_score,
            supply_demand_stage = EXCLUDED.supply_demand_stage,
            ingested_at = NOW()
    """
    
    affected = writer._execute_batch(query, records)
    logger.info(f"   ✅ Saved {affected} rows to factors.volume_analysis")
    
    return affected


def verify_migration(writer) -> dict:
    """Verify row counts after migration."""
    print("\n" + "=" * 60)
    print("🔍 Verifying Migration")
    print("=" * 60)
    
    tables = [
        ("market.tickers", "종목 마스터"),
        ("market.daily_prices", "일봉 가격"),
        ("factors.volume_analysis", "볼륨 분석"),
    ]
    
    counts = {}
    for table, desc in tables:
        try:
            result = writer.execute_raw(f"SELECT COUNT(*) FROM {table}")
            count = result[0][0] if result else 0
            counts[table] = count
            print(f"   {table:30} ({desc}): {count:,} rows")
        except Exception as e:
            print(f"   {table:30}: Error - {e}")
            counts[table] = 0
    
    return counts


def main():
    """Main entry point."""
    print("\n")
    print("🐘 CSV to PostgreSQL Migration")
    print("=" * 60)
    
    data_dir = Path(os.getenv("DATA_DIR", ROOT_DIR / "us_market"))
    print(f"   Data directory: {data_dir}")
    
    start_time = datetime.now()
    
    # Migrate each data source
    ticker_count = migrate_stocks_list(data_dir)
    price_count = migrate_daily_prices(data_dir)
    volume_count = migrate_volume_analysis(data_dir)
    
    # Verify
    writer = get_db_writer()
    verify_migration(writer)
    writer.close()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print(f"🎉 Migration complete! (Elapsed: {elapsed:.1f} seconds)")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
