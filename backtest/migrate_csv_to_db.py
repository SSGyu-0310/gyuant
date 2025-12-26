#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to SQLite Migration Script
One-time migration of existing CSV data to backtest database
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add parent directories to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backtest.db_schema import get_connection, init_db, get_db_path, get_table_counts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_daily_prices(data_dir: Path, batch_size: int = 10000) -> int:
    """
    Migrate us_daily_prices.csv to daily_prices table.
    
    Args:
        data_dir: Directory containing the CSV file
        batch_size: Number of rows to insert per batch
        
    Returns:
        Number of rows inserted
    """
    csv_path = data_dir / 'us_daily_prices.csv'
    if not csv_path.exists():
        logger.warning(f"⚠️ Price file not found: {csv_path}")
        return 0
    
    logger.info(f"📂 Loading prices from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loaded {len(df):,} rows from CSV")
        
        # Rename columns to match schema
        column_map = {
            'current_price': 'close',
        }
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"❌ Missing required column: {col}")
                return 0
        
        # Add metadata columns if missing
        if 'source' not in df.columns:
            df['source'] = 'FMP'
        if 'as_of' not in df.columns:
            df['as_of'] = datetime.now().isoformat()
        
        # Insert into database in batches
        conn = get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            
            for _, row in batch.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO daily_prices 
                        (ticker, date, open, high, low, close, volume, change, change_rate, name, market, source, as_of)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('ticker'),
                        row.get('date'),
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('close'),
                        row.get('volume'),
                        row.get('change'),
                        row.get('change_rate'),
                        row.get('name'),
                        row.get('market'),
                        row.get('source'),
                        row.get('as_of'),
                    ))
                    inserted += 1
                except Exception as e:
                    logger.debug(f"Insert failed for row: {e}")
            
            conn.commit()
            logger.info(f"   Processed {min(i + batch_size, len(df)):,} / {len(df):,} rows...")
        
        conn.close()
        logger.info(f"✅ Inserted {inserted:,} rows into daily_prices")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_stock_list(data_dir: Path) -> int:
    """
    Migrate us_stocks_list.csv to universe_snapshots table (today's date).
    
    Args:
        data_dir: Directory containing the CSV file
        
    Returns:
        Number of rows inserted
    """
    csv_path = data_dir / 'us_stocks_list.csv'
    if not csv_path.exists():
        logger.warning(f"⚠️ Stock list file not found: {csv_path}")
        return 0
    
    logger.info(f"📂 Loading stock list from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        today = datetime.now().strftime('%Y-%m-%d')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO universe_snapshots 
                    (date, ticker, name, sector, market)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    today,
                    row.get('ticker'),
                    row.get('name'),
                    row.get('sector', 'N/A'),
                    row.get('market', 'S&P500'),
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Insert failed for row: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Inserted {inserted} rows into universe_snapshots (date: {today})")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_existing_snapshots(data_dir: Path) -> int:
    """
    Migrate existing CSV snapshots from universe_snapshots directory.
    
    Args:
        data_dir: Directory containing backtest/universe_snapshots/
        
    Returns:
        Number of rows inserted
    """
    snapshot_dir = data_dir / 'backtest' / 'universe_snapshots'
    if not snapshot_dir.exists():
        logger.info(f"📁 No snapshot directory found: {snapshot_dir}")
        return 0
    
    csv_files = list(snapshot_dir.glob('*.csv'))
    if not csv_files:
        logger.info(f"📁 No snapshot files found in {snapshot_dir}")
        return 0
    
    logger.info(f"📂 Found {len(csv_files)} snapshot files to migrate...")
    
    conn = get_connection()
    cursor = conn.cursor()
    total_inserted = 0
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            date_str = csv_path.stem  # e.g., "2024-12-26"
            
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO universe_snapshots 
                        (date, ticker, name, sector, market)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        row.get('date', date_str),
                        row.get('ticker'),
                        row.get('name'),
                        row.get('sector', 'N/A'),
                        row.get('market', 'S&P500'),
                    ))
                    total_inserted += 1
                except Exception as e:
                    logger.debug(f"Insert failed: {e}")
            
            conn.commit()
            logger.info(f"   Migrated {csv_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to migrate {csv_path.name}: {e}")
    
    conn.close()
    logger.info(f"✅ Migrated {total_inserted} snapshot records")
    return total_inserted


def run_migration(data_dir: str = None):
    """
    Run full CSV to SQLite migration.
    
    Args:
        data_dir: Data directory path (uses DATA_DIR env if not provided)
    """
    if data_dir is None:
        data_dir = os.getenv('DATA_DIR', str(ROOT_DIR / 'us_market'))
    data_dir = Path(data_dir)
    
    logger.info("🚀 Starting CSV to SQLite Migration...")
    logger.info(f"📁 Data directory: {data_dir}")
    
    # Initialize database
    if not init_db():
        logger.error("❌ Failed to initialize database")
        return False
    
    # Run migrations
    prices_count = migrate_daily_prices(data_dir)
    stocks_count = migrate_stock_list(data_dir)
    snapshots_count = migrate_existing_snapshots(data_dir)
    
    # Summary
    logger.info("\n📊 Migration Summary:")
    logger.info(f"   daily_prices: {prices_count:,} rows")
    logger.info(f"   universe_snapshots: {stocks_count + snapshots_count:,} rows")
    
    # Verify
    counts = get_table_counts()
    logger.info(f"\n📈 Final table counts: {counts}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate CSV data to SQLite')
    parser.add_argument('--dir', default=None, help='Data directory')
    args = parser.parse_args()
    
    success = run_migration(args.dir)
    if success:
        print("\n🎉 Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)
