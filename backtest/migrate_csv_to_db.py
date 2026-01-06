#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV/JSON to SQLite Migration Script
Aligned with docs/db/migration-checklist.md
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add parent directories to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.db_schema import get_connection, init_db, get_db_path, get_table_counts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_market_stocks(data_dir: Path) -> int:
    """Migrate us_stocks_list.csv to market_stocks table"""
    csv_path = data_dir / 'us_stocks_list.csv'
    if not csv_path.exists():
        logger.warning(f"⚠️ Stock list file not found: {csv_path}")
        return 0
    
    logger.info(f"📂 Migrating stock list to market_stocks...")
    
    try:
        df = pd.read_csv(csv_path)
        conn = get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO market_stocks 
                    (ticker, name, sector, industry, market, is_active, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, 'FMP', ?)
                """, (
                    row.get('ticker'),
                    row.get('name'),
                    row.get('sector', 'N/A'),
                    row.get('industry'),
                    row.get('market', 'S&P500'),
                    datetime.now().isoformat(),
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Insert failed: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Inserted {inserted} rows into market_stocks")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_market_prices(data_dir: Path, batch_size: int = 10000) -> int:
    """Migrate us_daily_prices.csv to market_prices_daily table"""
    csv_path = data_dir / 'us_daily_prices.csv'
    if not csv_path.exists():
        logger.warning(f"⚠️ Price file not found: {csv_path}")
        return 0
    
    logger.info(f"📂 Migrating prices to market_prices_daily...")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loaded {len(df):,} rows from CSV")
        
        conn = get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            
            for _, row in batch.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO market_prices_daily 
                        (ticker, date, open, high, low, close, volume, change, change_rate, source, ingested_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'FMP', ?)
                    """, (
                        row.get('ticker'),
                        row.get('date'),
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('current_price'),  # Maps to 'close'
                        row.get('volume'),
                        row.get('change'),
                        row.get('change_rate'),
                        datetime.now().isoformat(),
                    ))
                    inserted += 1
                except Exception as e:
                    logger.debug(f"Insert failed: {e}")
            
            conn.commit()
            logger.info(f"   Processed {min(i + batch_size, len(df)):,} / {len(df):,} rows...")
        
        conn.close()
        logger.info(f"✅ Inserted {inserted:,} rows into market_prices_daily")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_bt_prices(data_dir: Path, batch_size: int = 10000) -> int:
    """Copy prices to bt_prices_daily for backtesting (separate from operational)"""
    csv_path = data_dir / 'us_daily_prices.csv'
    if not csv_path.exists():
        return 0
    
    logger.info(f"📂 Migrating prices to bt_prices_daily (backtest)...")
    
    try:
        df = pd.read_csv(csv_path)
        conn = get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            
            for _, row in batch.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO bt_prices_daily 
                        (ticker, date, open, high, low, close, volume, source, ingested_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'FMP', ?)
                    """, (
                        row.get('ticker'),
                        row.get('date'),
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('current_price'),
                        row.get('volume'),
                        datetime.now().isoformat(),
                    ))
                    inserted += 1
                except Exception as e:
                    logger.debug(f"Insert failed: {e}")
            
            conn.commit()
            if (i + batch_size) % 100000 == 0:
                logger.info(f"   bt_prices: {min(i + batch_size, len(df)):,} / {len(df):,}")
        
        conn.close()
        logger.info(f"✅ Inserted {inserted:,} rows into bt_prices_daily")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_bt_universe(data_dir: Path) -> int:
    """Migrate stock list to bt_universe_snapshot"""
    csv_path = data_dir / 'us_stocks_list.csv'
    if not csv_path.exists():
        return 0
    
    logger.info(f"📂 Migrating to bt_universe_snapshot...")
    
    try:
        df = pd.read_csv(csv_path)
        today = datetime.now().strftime('%Y-%m-%d')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO bt_universe_snapshot 
                    (as_of_date, ticker, name, sector, market, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, 'FMP', ?)
                """, (
                    today,
                    row.get('ticker'),
                    row.get('name'),
                    row.get('sector', 'N/A'),
                    row.get('market', 'S&P500'),
                    datetime.now().isoformat(),
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Insert failed: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Inserted {inserted} rows into bt_universe_snapshot (date: {today})")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_existing_snapshots(data_dir: Path) -> int:
    """Migrate existing CSV snapshots from universe_snapshots directory"""
    snapshot_dir = data_dir / 'backtest' / 'universe_snapshots'
    if not snapshot_dir.exists():
        return 0
    
    csv_files = list(snapshot_dir.glob('*.csv'))
    if not csv_files:
        return 0
    
    logger.info(f"📂 Found {len(csv_files)} snapshot files to migrate...")
    
    conn = get_connection()
    cursor = conn.cursor()
    total_inserted = 0
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            date_str = csv_path.stem
            
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO bt_universe_snapshot 
                        (as_of_date, ticker, name, sector, market, source, ingested_at)
                        VALUES (?, ?, ?, ?, ?, 'FMP', ?)
                    """, (
                        row.get('date', date_str),
                        row.get('ticker'),
                        row.get('name'),
                        row.get('sector', 'N/A'),
                        row.get('market', 'S&P500'),
                        datetime.now().isoformat(),
                    ))
                    total_inserted += 1
                except Exception as e:
                    logger.debug(f"Insert failed: {e}")
            
            conn.commit()
            
        except Exception as e:
            logger.warning(f"Failed to migrate {csv_path.name}: {e}")
    
    conn.close()
    logger.info(f"✅ Migrated {total_inserted} snapshot records from CSV files")
    return total_inserted


def migrate_volume_analysis(data_dir: Path) -> int:
    """Migrate us_volume_analysis.csv to market_volume_analysis"""
    csv_path = data_dir / 'us_volume_analysis.csv'
    if not csv_path.exists():
        logger.info(f"📁 No volume analysis file found")
        return 0
    
    logger.info(f"📂 Migrating volume analysis...")
    
    try:
        df = pd.read_csv(csv_path)
        today = datetime.now().strftime('%Y-%m-%d')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO market_volume_analysis 
                    (ticker, as_of_date, name, obv, obv_change_20d, ad_line, ad_change_20d, 
                     mfi, vol_ratio_5d_20d, surge_count_5d, surge_count_20d,
                     supply_demand_score, supply_demand_stage, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'script', ?)
                """, (
                    row.get('ticker'),
                    row.get('date', today),
                    row.get('name'),
                    row.get('obv'),
                    row.get('obv_change_20d'),
                    row.get('ad_line'),
                    row.get('ad_change_20d'),
                    row.get('mfi'),
                    row.get('vol_ratio_5d_20d'),
                    row.get('surge_count_5d'),
                    row.get('surge_count_20d'),
                    row.get('supply_demand_score'),
                    row.get('supply_demand_stage'),
                    datetime.now().isoformat(),
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Insert failed: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Inserted {inserted} rows into market_volume_analysis")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def run_migration(data_dir: str = None):
    """Run full CSV to SQLite migration"""
    if data_dir is None:
        data_dir = os.getenv('DATA_DIR', str(ROOT_DIR / 'us_market'))
    data_dir = Path(data_dir)
    
    logger.info("🚀 Starting Full Migration to gyuant_market.db...")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"🗄️ Database: {get_db_path()}")
    
    # Initialize database
    if not init_db():
        logger.error("❌ Failed to initialize database")
        return False
    
    # Run migrations in order (per migration-checklist.md)
    results = {}
    results['market_stocks'] = migrate_market_stocks(data_dir)
    results['market_prices_daily'] = migrate_market_prices(data_dir)
    results['market_volume_analysis'] = migrate_volume_analysis(data_dir)
    results['bt_prices_daily'] = migrate_bt_prices(data_dir)
    results['bt_universe_snapshot'] = migrate_bt_universe(data_dir) + migrate_existing_snapshots(data_dir)
    
    # Summary
    logger.info("\n📊 Migration Summary:")
    for table, count in results.items():
        logger.info(f"   {table}: {count:,} rows")
    
    # Verify
    counts = get_table_counts()
    logger.info(f"\n📈 Final table counts:")
    for table, count in counts.items():
        if count > 0:
            logger.info(f"   {table}: {count:,}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate CSV/JSON to SQLite')
    parser.add_argument('--dir', default=None, help='Data directory')
    args = parser.parse_args()
    
    success = run_migration(args.dir)
    if success:
        print("\n🎉 Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)
