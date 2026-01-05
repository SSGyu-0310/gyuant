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
    sys.path.append(str(ROOT_DIR))

from backtest.db_schema import get_connection, init_db, get_db_path, get_table_counts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _file_date(path: Path) -> str:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return mtime.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _extract_date(payload, fallback: str) -> str:
    if isinstance(payload, dict):
        for key in ("as_of_date", "analysis_date", "date", "updated", "timestamp", "week_start"):
            val = payload.get(key)
            if not val:
                continue
            return str(val)[:10]
    return fallback


def _safe_run_id(value: str, fallback: str) -> str:
    raw = value or fallback
    keep = "".join(ch for ch in str(raw) if ch.isalnum())
    if not keep:
        keep = "".join(ch for ch in fallback if ch.isalnum())
    return f"sm_{keep}"


def _load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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


def migrate_market_etf_flows(data_dir: Path) -> int:
    """Migrate us_etf_flows.csv to market_etf_flows table"""
    csv_path = data_dir / "us_etf_flows.csv"
    if not csv_path.exists():
        logger.warning(f"⚠️ ETF flows file not found: {csv_path}")
        return 0

    logger.info("📂 Migrating ETF flows to market_etf_flows...")
    as_of_date = _file_date(csv_path)

    try:
        df = pd.read_csv(csv_path)
        conn = get_connection()
        cursor = conn.cursor()

        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO market_etf_flows
                    (ticker, as_of_date, name, category, current_price, price_1w_pct, price_1m_pct,
                     vol_ratio_5d_20d, obv_change_20d_pct, avg_volume_20d, flow_score, flow_status,
                     source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'FMP', ?)
                    """,
                    (
                        row.get("ticker"),
                        as_of_date,
                        row.get("name"),
                        row.get("category"),
                        row.get("current_price"),
                        row.get("price_1w_pct"),
                        row.get("price_1m_pct"),
                        row.get("vol_ratio_5d_20d"),
                        row.get("obv_change_20d_pct"),
                        row.get("avg_volume_20d"),
                        row.get("flow_score"),
                        row.get("flow_status"),
                        datetime.now().isoformat(),
                    ),
                )
                inserted += 1
            except Exception as e:
                logger.debug(f"Insert failed: {e}")

        conn.commit()
        conn.close()
        logger.info(f"✅ Inserted {inserted} rows into market_etf_flows")
        return inserted

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_smart_money(data_dir: Path) -> int:
    """Migrate smart money outputs to market_smart_money_runs/picks"""
    current_path = data_dir / "smart_money_current.json"
    picks_path = data_dir / "smart_money_picks_v2.csv"
    if not current_path.exists() and not picks_path.exists():
        logger.warning("⚠️ Smart money files not found")
        return 0

    logger.info("📂 Migrating smart money runs/picks...")
    snapshot = _load_json(current_path) if current_path.exists() else None
    fallback_date = _file_date(current_path) if current_path.exists() else _file_date(picks_path)
    analysis_date = _extract_date(snapshot, fallback_date)
    analysis_ts = None
    if isinstance(snapshot, dict):
        analysis_ts = snapshot.get("analysis_timestamp") or snapshot.get("updated") or snapshot.get("timestamp")
    run_id = _safe_run_id(analysis_ts, analysis_date)
    summary_json = None
    if isinstance(snapshot, dict):
        summary_json = json.dumps(snapshot.get("summary", {}), ensure_ascii=False)

    inserted = 0
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO market_smart_money_runs
            (run_id, analysis_date, analysis_timestamp, summary_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                analysis_date,
                analysis_ts,
                summary_json,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

        if picks_path.exists():
            df = pd.read_csv(picks_path)
            for _, row in df.iterrows():
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO market_smart_money_picks
                        (run_id, rank, ticker, name, sector, composite_score, grade, current_price,
                         price_at_rec, change_since_rec, target_upside, sd_score, tech_score, fund_score,
                         analyst_score, rs_score, recommendation, rsi, ma_signal, pe_ratio, market_cap_b,
                         size, rs_20d)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            row.get("rank"),
                            row.get("ticker"),
                            row.get("name"),
                            row.get("sector"),
                            row.get("composite_score"),
                            row.get("grade"),
                            row.get("current_price"),
                            row.get("current_price"),
                            0,
                            row.get("target_upside"),
                            row.get("sd_score"),
                            row.get("tech_score"),
                            row.get("fund_score"),
                            row.get("analyst_score"),
                            row.get("rs_score"),
                            row.get("recommendation"),
                            row.get("rsi"),
                            row.get("ma_signal"),
                            row.get("pe_ratio"),
                            row.get("market_cap_b"),
                            row.get("size"),
                            row.get("rs_20d"),
                        ),
                    )
                    inserted += 1
                except Exception as e:
                    logger.debug(f"Insert failed: {e}")
            conn.commit()

        conn.close()
        logger.info(f"✅ Inserted {inserted} rows into market_smart_money_picks")
        return inserted
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 0


def migrate_market_documents(data_dir: Path) -> int:
    """Migrate JSON documents to market_documents table"""
    docs = [
        ("macro_analysis.json", "macro_analysis", "ko", "gemini"),
        ("macro_analysis_en.json", "macro_analysis", "en", "gemini"),
        ("macro_analysis_gpt.json", "macro_analysis", "ko", "gpt"),
        ("macro_analysis_gpt_en.json", "macro_analysis", "en", "gpt"),
        ("sector_heatmap.json", "sector_heatmap", "na", "na"),
        ("options_flow.json", "options_flow", "na", "na"),
        ("insider_moves.json", "insider_moves", "na", "na"),
        ("portfolio_risk.json", "portfolio_risk", "na", "na"),
        ("ai_summaries.json", "ai_summaries", "na", "na"),
        ("weekly_calendar.json", "calendar", "na", "na"),
        ("etf_flow_analysis.json", "etf_flow_analysis", "na", "na"),
        ("final_top10_report.json", "final_top10_report", "na", "na"),
        ("smart_money_current.json", "smart_money_current", "na", "na"),
    ]

    inserted = 0
    conn = get_connection()
    cursor = conn.cursor()
    for filename, doc_type, lang, model in docs:
        path = data_dir / filename
        if not path.exists():
            continue
        payload = _load_json(path)
        if payload is None:
            logger.warning(f"⚠️ Failed to read {path}")
            continue
        as_of_date = _extract_date(payload, _file_date(path))
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO market_documents
                (doc_type, as_of_date, lang, model, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_type,
                    as_of_date,
                    lang,
                    model,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"Insert failed: {e}")

    conn.commit()
    conn.close()
    if inserted:
        logger.info(f"✅ Inserted {inserted} rows into market_documents")
    return inserted


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
    
    logger.info(f"🚀 Starting Full Migration to {get_db_path().name}...")
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
    results['market_etf_flows'] = migrate_market_etf_flows(data_dir)
    results['market_smart_money_picks'] = migrate_smart_money(data_dir)
    results['market_documents'] = migrate_market_documents(data_dir)
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
