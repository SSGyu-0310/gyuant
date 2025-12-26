#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Stock Daily Prices Collection Script
Collects daily price data for NASDAQ and S&P 500 stocks using FMP
Supports dual-write to CSV and SQLite for backtesting
"""

import os
import sys
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.fmp_client import get_fmp_client
from utils.symbols import to_fmp_symbol

# SQLite toggle - dual-write enabled by default for DB transition
USE_SQLITE = os.getenv('USE_SQLITE', 'true').lower() == 'true'

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class USStockDailyPricesCreator:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.getenv('DATA_DIR', '.')
        self.output_dir = self.data_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data file paths
        self.prices_file = os.path.join(self.output_dir, 'us_daily_prices.csv')
        self.stocks_list_file = os.path.join(self.output_dir, 'us_stocks_list.csv')
        
        # Backtest paths
        self.snapshot_dir = os.path.join(self.data_dir, 'backtest', 'universe_snapshots')
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # SQLite database path
        self.db_path = os.path.join(self.data_dir, 'gyuant_market.db')
        
        # Start date for historical data
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime.now()
        
    def get_sp500_tickers(self) -> List[Dict]:
        """Get full S&P 500 tickers list"""
        logger.info("📊 Loading full S&P 500 stocks...")
        
        # Full S&P 500 tickers (as of late 2024)
        sp500_tickers = [
            "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
            "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG",
            "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
            "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH",
            "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
            "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG",
            "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B",
            "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT",
            "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF",
            "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME",
            "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST",
            "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX", "CTAS",
            "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY",
            "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR",
            "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
            "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV",
            "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS",
            "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F",
            "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS",
            "FITB", "FLT", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD",
            "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM",
            "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
            "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE",
            "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE",
            "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG",
            "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL",
            "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC",
            "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L",
            "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW",
            "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR",
            "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM",
            "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS",
            "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB",
            "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI",
            "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR",
            "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY",
            "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP",
            "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC",
            "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC",
            "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL",
            "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX",
            "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG",
            "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF",
            "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC",
            "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV",
            "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER",
            "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI",
            "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ",
            "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB",
            "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM",
            "ZBH", "ZBRA", "ZTS"
        ]
        
        stocks = []
        for ticker in sp500_tickers:
            stocks.append({
                'ticker': ticker,
                'name': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'market': 'S&P500'
            })
        
        logger.info(f"✅ Loaded {len(stocks)} S&P 500 stocks")
        return stocks
    
    def get_nasdaq100_tickers(self) -> List[Dict]:
        """Skip NASDAQ - already covered in S&P 500"""
        logger.info("📊 Skipping NASDAQ 100 (covered in S&P 500)...")
        return []
    
    def load_or_create_stock_list(self) -> pd.DataFrame:
        """Load existing stock list or create new one"""
        if os.path.exists(self.stocks_list_file):
            logger.info(f"📂 Loading existing stock list: {self.stocks_list_file}")
            return pd.read_csv(self.stocks_list_file)
        
        # Create new stock list
        logger.info("📝 Creating new US stock list...")
        
        sp500_stocks = self.get_sp500_tickers()
        nasdaq_stocks = self.get_nasdaq100_tickers()
        
        # Combine and remove duplicates
        all_stocks = sp500_stocks + nasdaq_stocks
        stocks_df = pd.DataFrame(all_stocks)
        stocks_df = stocks_df.drop_duplicates(subset=['ticker'], keep='first')
        
        # Save stock list
        stocks_df.to_csv(self.stocks_list_file, index=False)
        logger.info(f"✅ Saved {len(stocks_df)} stocks to {self.stocks_list_file}")
        
        return stocks_df
    
    def load_existing_prices(self) -> pd.DataFrame:
        """Load existing price data"""
        if os.path.exists(self.prices_file):
            logger.info(f"📂 Loading existing prices: {self.prices_file}")
            df = pd.read_csv(self.prices_file)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            invalid_dates = df['date'].isna().sum()
            if invalid_dates:
                logger.warning(f"Dropped {invalid_dates} rows with invalid dates in existing file")
                df = df.dropna(subset=['date'])
            return df
        return pd.DataFrame()
    
    def get_latest_dates(self, df: pd.DataFrame) -> Dict[str, datetime]:
        """Get latest date for each ticker"""
        if df.empty:
            return {}
        return df.groupby('ticker')['date'].max().to_dict()
    
    def download_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Download daily price data for a single stock"""
        try:
            fmp = get_fmp_client()
            fmp_symbol = to_fmp_symbol(ticker)
            data = fmp.historical_price_full(
                fmp_symbol,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
            )

            hist_list = data.get("historical", []) if isinstance(data, dict) else []
            if not hist_list:
                return pd.DataFrame()

            hist = pd.DataFrame(hist_list)
            hist = hist.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'current_price',
                'volume': 'volume'
            })
            hist['ticker'] = ticker
            hist['date'] = pd.to_datetime(hist['date'])
            hist = hist.sort_values('date')

            # Calculate change and change_rate
            hist['change'] = hist['current_price'].diff()
            hist['change_rate'] = hist['current_price'].pct_change() * 100
            hist['date'] = hist['date'].dt.strftime('%Y-%m-%d')

            cols = ['ticker', 'date', 'open', 'high', 'low', 'current_price', 'volume', 'change', 'change_rate']
            hist = hist[cols]

            return hist
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to download {ticker}: {e}")
            return pd.DataFrame()
    
    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """Get SQLite connection with proper settings (returns None if not using SQLite)"""
        if not USE_SQLITE:
            return None
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            return conn
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to SQLite: {e}")
            return None
    
    def _save_to_db(self, df: pd.DataFrame) -> int:
        """
        Save price data to SQLite database (dual-write).
        Falls back gracefully if DB is unavailable.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Number of rows inserted (0 if failed or disabled)
        """
        if not USE_SQLITE or df.empty:
            return 0
        
        conn = self._get_db_connection()
        if conn is None:
            return 0
        
        try:
            cursor = conn.cursor()
            inserted = 0
            
            for _, row in df.iterrows():
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
                        row.get('current_price'),  # Maps to 'close' in DB
                        row.get('volume'),
                        row.get('change'),
                        row.get('change_rate'),
                        datetime.now().isoformat(),
                    ))
                    inserted += cursor.rowcount
                except Exception as e:
                    logger.debug(f"DB insert failed: {e}")
            
            conn.commit()
            conn.close()
            logger.info(f"🗄️ SQLite: Inserted {inserted} rows into market_prices_daily")
            return inserted
            
        except Exception as e:
            logger.warning(f"⚠️ SQLite save failed (CSV still saved): {e}")
            try:
                conn.close()
            except:
                pass
            return 0
    
    def _save_universe_to_db(self, stocks_df: pd.DataFrame, date_str: str) -> int:
        """
        Save universe snapshot to SQLite database.
        
        Args:
            stocks_df: DataFrame with stock list
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            Number of rows inserted
        """
        if not USE_SQLITE or stocks_df.empty:
            return 0
        
        conn = self._get_db_connection()
        if conn is None:
            return 0
        
        try:
            cursor = conn.cursor()
            inserted = 0
            
            for _, row in stocks_df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO bt_universe_snapshot 
                        (as_of_date, ticker, name, sector, market, source, ingested_at)
                        VALUES (?, ?, ?, ?, ?, 'FMP', ?)
                    """, (
                        date_str,
                        row.get('ticker'),
                        row.get('name'),
                        row.get('sector', 'N/A'),
                        row.get('market', 'S&P500'),
                        datetime.now().isoformat(),
                    ))
                    inserted += cursor.rowcount
                except Exception as e:
                    logger.debug(f"DB insert failed: {e}")
            
            conn.commit()
            conn.close()
            if inserted > 0:
                logger.info(f"🗄️ SQLite: Inserted {inserted} rows into bt_universe_snapshot")
            return inserted
            
        except Exception as e:
            logger.warning(f"⚠️ SQLite universe save failed: {e}")
            try:
                conn.close()
            except:
                pass
            return 0
    
    def save_universe_snapshot(self, stocks_df: pd.DataFrame, date_str: str) -> bool:
        """Save current universe as a snapshot for backtest survivorship bias prevention
        
        Args:
            stocks_df: DataFrame with stock list (ticker, name, sector columns)
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            True if snapshot was saved, False otherwise
        """
        snapshot_path = os.path.join(self.snapshot_dir, f"{date_str}.csv")
        if os.path.exists(snapshot_path):
            logger.info(f"📸 Snapshot already exists: {snapshot_path}")
            return True
        
        try:
            # Prepare snapshot with required columns
            snapshot_df = stocks_df[['ticker', 'name']].copy()
            snapshot_df.insert(0, 'date', date_str)
            
            # Add sector column (use N/A if not present)
            if 'sector' in stocks_df.columns:
                snapshot_df['sector'] = stocks_df['sector']
            else:
                snapshot_df['sector'] = 'N/A'
            
            # Ensure column order: date, ticker, name, sector
            snapshot_df = snapshot_df[['date', 'ticker', 'name', 'sector']]
            
            snapshot_df.to_csv(snapshot_path, index=False)
            logger.info(f"📸 Saved universe snapshot: {snapshot_path} ({len(snapshot_df)} stocks)")
            return True
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")
            return False
    
    def run(self, full_refresh: bool = False) -> bool:
        """Run data collection (incremental by default)"""
        logger.info("🚀 US Stock Daily Prices Collection Started...")
        if not get_fmp_client().api_key:
            logger.warning("⚠️ FMP_API_KEY is not set. Price data may be empty.")
        
        try:
            # 1. Load stock list
            stocks_df = self.load_or_create_stock_list()
            if stocks_df.empty:
                logger.error("❌ No stocks to process")
                return False
            
            # 1.5 Save universe snapshot for today (prevents survivorship bias)
            today_str = datetime.now().strftime('%Y-%m-%d')
            self.save_universe_snapshot(stocks_df, today_str)
            self._save_universe_to_db(stocks_df, today_str)  # Dual-write to SQLite
            
            # 2. Load existing data
            existing_df = pd.DataFrame() if full_refresh else self.load_existing_prices()
            latest_dates = self.get_latest_dates(existing_df)
            
            # 3. Determine target end date
            now = datetime.now()
            target_end_date = now
            
            # 4. Collect data
            all_new_data = []
            failed_tickers = []
            
            for idx, row in tqdm(stocks_df.iterrows(), desc="Downloading US stocks", total=len(stocks_df)):
                ticker = row['ticker']
                
                # Determine start date
                if ticker in latest_dates:
                    start_date = latest_dates[ticker] + timedelta(days=1)
                else:
                    start_date = self.start_date
                
                # Skip if already up to date
                if start_date >= target_end_date:
                    continue
                
                # Download data
                new_data = self.download_stock_data(ticker, start_date, target_end_date)
                
                if not new_data.empty:
                    # Add name from stock list
                    new_data['name'] = row['name']
                    new_data['market'] = row['market']
                    all_new_data.append(new_data)
                else:
                    failed_tickers.append(ticker)
            
            # 5. Combine and save
            if all_new_data:
                new_df = pd.concat(all_new_data, ignore_index=True)
                
                if not existing_df.empty:
                    final_df = pd.concat([existing_df, new_df])
                    final_df = final_df.drop_duplicates(subset=['ticker', 'date'], keep='last')
                else:
                    final_df = new_df
                
                # Normalize dates, sort, and save
                final_df['date'] = pd.to_datetime(final_df['date'], errors='coerce')
                invalid_dates = final_df['date'].isna().sum()
                if invalid_dates:
                    logger.warning(f"Dropped {invalid_dates} rows with invalid dates before saving")
                    final_df = final_df.dropna(subset=['date'])

                final_df = final_df.sort_values(['ticker', 'date']).reset_index(drop=True)
                final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
                final_df.to_csv(self.prices_file, index=False)
                
                # Dual-write to SQLite if enabled
                self._save_to_db(new_df)
                
                logger.info(f"✅ Saved {len(new_df)} new records to {self.prices_file}")
                logger.info(f"📊 Total records: {len(final_df)}")
            else:
                if not os.path.exists(self.prices_file):
                    cols = ['ticker', 'date', 'open', 'high', 'low', 'current_price', 'volume', 'change', 'change_rate']
                    pd.DataFrame(columns=cols).to_csv(self.prices_file, index=False)
                    logger.warning(f"⚠️ No data collected; created empty file at {self.prices_file}")
                else:
                    logger.info("✨ All data is up to date!")
            
            # 6. Summary
            logger.info(f"\n📊 Collection Summary:")
            logger.info(f"   Total stocks: {len(stocks_df)}")
            logger.info(f"   Success: {len(stocks_df) - len(failed_tickers)}")
            logger.info(f"   Failed: {len(failed_tickers)}")
            
            if failed_tickers[:10]:
                logger.warning(f"   Failed samples: {failed_tickers[:10]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during collection: {e}")
            return False


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='US Stock Daily Prices Collector')
    parser.add_argument('--dir', default=os.getenv('DATA_DIR', '.'), help='Data directory')
    parser.add_argument('--full', action='store_true', help='Full refresh (ignore existing data)')
    args = parser.parse_args()
    
    creator = USStockDailyPricesCreator(data_dir=args.dir)
    success = creator.run(full_refresh=args.full)
    
    if success:
        print("\n🎉 US Stock Daily Prices collection completed!")
        print(f"📁 File location: ./us_daily_prices.csv")
    else:
        print("\n❌ Collection failed.")


if __name__ == "__main__":
    main()
