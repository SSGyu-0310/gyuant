#!/usr/bin/env python3
"""
Build Universe Snapshots
------------------------
Generates monthly universe snapshots from historical price data.
Populates the 'backtest.universe_snapshot' table.

Usage:
    python scripts/build_universe_snapshots.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, date
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backtest.db_schema_pg import get_session, BacktestUniverse, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_FILE = ROOT_DIR / "us_market" / "us_daily_prices.csv"


def build_snapshots():
    if not DATA_FILE.exists():
        logger.error(f"Data file not found: {DATA_FILE}")
        sys.exit(1)

    logger.info(f"Loading price data from {DATA_FILE}...")
    # Load only necessary columns to save memory
    df = pd.read_csv(
        DATA_FILE,
        usecols=["ticker", "date", "current_price", "volume", "name", "market"],
        parse_dates=["date"],
    )

    logger.info(f"Loaded {len(df)} records. Range: {df.date.min()} to {df.date.max()}")

    # Determine snapshot dates (Month Ends)
    # We take the last available date for each month
    df["year_month"] = df["date"].dt.to_period("M")

    # Get last trading day per month
    snapshot_dates = df.groupby("year_month")["date"].max().unique()
    snapshot_dates = sorted(snapshot_dates)

    logger.info(f"Found {len(snapshot_dates)} monthly snapshot dates.")

    session = get_session()

    try:
        # Initialize DB if needed (create schemas/tables)
        # init_db() # Assuming already initialized, but safe to run if idempotent

        for snap_date in tqdm(snapshot_dates, desc="Generating Snapshots"):
            snap_date_date = pd.to_datetime(snap_date).date()

            # Get tickers active on this date (or just present in the file for this date)
            # MVP: Tickers with volume > 0 on this date
            day_data = df[df["date"] == snap_date]

            # If strictly MVP, we just take all tickers present on that day
            # Ideally we want liquid stocks. Let's say volume > 0.
            active_tickers = day_data[day_data["volume"] > 0]

            if active_tickers.empty:
                logger.warning(
                    f"No active tickers found for {snap_date_date}. Skipping."
                )
                continue

            # Clear existing snapshot for this date to ensure idempotency
            session.execute(
                text("DELETE FROM backtest.universe_snapshot WHERE as_of_date = :date"),
                {"date": snap_date_date},
            )

            # Prepare bulk insert objects
            snapshot_objects = []
            for _, row in active_tickers.iterrows():
                # Clean sector/market data if available, else default
                # us_daily_prices.csv has 'market' column (e.g. S&P500)
                # It doesn't seem to have sector in the sample I saw, checking 'name'

                snapshot_objects.append(
                    BacktestUniverse(
                        as_of_date=snap_date_date,
                        ticker=row["ticker"],
                        name=str(row["name"]) if pd.notna(row["name"]) else None,
                        sector=None,  # CSV sample didn't show sector, maybe join with us_sectors.csv if needed
                        market=str(row["market"]) if pd.notna(row["market"]) else "US",
                        weight=None,  # Equal weight implied if null, or calculated later
                        source="US_DAILY_PRICES_CSV",
                    )
                )

            session.bulk_save_objects(snapshot_objects)
            session.commit()

        logger.info("Successfully built universe snapshots.")

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to build snapshots: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    build_snapshots()
