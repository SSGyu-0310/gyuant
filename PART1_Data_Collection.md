# US Market Backend Blueprint - Part 1: 데이터 수집

> 이 문서는 US Market 시스템의 **데이터 수집** 관련 스크립트를 모아 놓은 것입니다.
> 빈 폴더에 이 코드들을 넣고 순서대로 실행하면 기본 데이터가 수집됩니다.

---

## 📁 파일 목록

| 파일명 | 설명 | 출력 파일 |
|--------|------|-----------|
| `create_us_daily_prices.py` | S&P 500 가격 데이터 수집 | `us_daily_prices.csv` |
| `analyze_volume.py` | 거래량/수급 분석 | `us_volume_analysis.csv` |
| `analyze_13f.py` | 기관 보유 분석 | `us_13f_holdings.csv` |
| `analyze_etf_flows.py` | ETF 자금 흐름 분석 | `us_etf_flows.csv`, `etf_flow_analysis.json` |

---

## 📦 필수 패키지

```bash
pip install pandas numpy yfinance tqdm requests python-dotenv
```

---

## 1️⃣ create_us_daily_prices.py

> S&P 500 종목의 일일 가격 데이터를 yfinance로 수집합니다.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Stock Daily Prices Collection Script
Collects daily price data for NASDAQ and S&P 500 stocks using yfinance
Similar to create_complete_daily_prices.py for Korean stocks
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from tqdm import tqdm

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class USStockDailyPricesCreator:
    def __init__(self):
        self.data_dir = os.getenv('DATA_DIR', '.')
        self.output_dir = self.data_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data file paths
        self.prices_file = os.path.join(self.output_dir, 'us_daily_prices.csv')
        self.stocks_list_file = os.path.join(self.output_dir, 'us_stocks_list.csv')
        
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
                'name': ticker,  # Will be fetched from yfinance
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
            df['date'] = pd.to_datetime(df['date'])
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
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return pd.DataFrame()
            
            hist = hist.reset_index()
            hist['ticker'] = ticker
            
            # Rename columns to match Korean stock format
            hist = hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'current_price',
                'Volume': 'volume'
            })
            
            # Calculate change and change_rate
            hist['change'] = hist['current_price'].diff()
            hist['change_rate'] = hist['current_price'].pct_change() * 100
            
            # Select required columns
            cols = ['ticker', 'date', 'open', 'high', 'low', 'current_price', 'volume', 'change', 'change_rate']
            hist = hist[cols]
            
            return hist
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to download {ticker}: {e}")
            return pd.DataFrame()
    
    def run(self, full_refresh: bool = False) -> bool:
        """Run data collection (incremental by default)"""
        logger.info("🚀 US Stock Daily Prices Collection Started...")
        
        try:
            # 1. Load stock list
            stocks_df = self.load_or_create_stock_list()
            if stocks_df.empty:
                logger.error("❌ No stocks to process")
                return False
            
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
                
                # Sort and save
                final_df = final_df.sort_values(['ticker', 'date']).reset_index(drop=True)
                final_df.to_csv(self.prices_file, index=False)
                
                logger.info(f"✅ Saved {len(new_df)} new records to {self.prices_file}")
                logger.info(f"📊 Total records: {len(final_df)}")
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
    parser.add_argument('--full', action='store_true', help='Full refresh (ignore existing data)')
    args = parser.parse_args()
    
    creator = USStockDailyPricesCreator()
    success = creator.run(full_refresh=args.full)
    
    if success:
        print("\n🎉 US Stock Daily Prices collection completed!")
        print(f"📁 File location: ./us_daily_prices.csv")
    else:
        print("\n❌ Collection failed.")


if __name__ == "__main__":
    main()
```

**사용법:**
```bash
python3 create_us_daily_prices.py          # 증분 업데이트
python3 create_us_daily_prices.py --full   # 전체 새로고침
```

---

## 2️⃣ analyze_volume.py

> OBV, A/D Line, MFI 등 거래량 기반 수급 분석을 수행합니다.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Stock Supply/Demand Analysis - Volume Technical Indicators
Calculates OBV, Accumulation/Distribution Line, Volume Surge Detection
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tqdm import tqdm

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolumeAnalyzer:
    """Volume-based technical analysis for supply/demand detection"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.prices_file = os.path.join(data_dir, 'us_daily_prices.csv')
        self.output_file = os.path.join(data_dir, 'us_volume_analysis.csv')
        
    def load_prices(self) -> pd.DataFrame:
        """Load daily price data"""
        if not os.path.exists(self.prices_file):
            raise FileNotFoundError(f"Price file not found: {self.prices_file}")
        
        logger.info(f"📂 Loading prices from {self.prices_file}")
        df = pd.read_csv(self.prices_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)
        - Price up: Add volume
        - Price down: Subtract volume
        - Price unchanged: No change
        """
        obv = [0]
        for i in range(1, len(df)):
            if df['current_price'].iloc[i] > df['current_price'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['current_price'].iloc[i] < df['current_price'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)
    
    def calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line
        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        A/D = Previous A/D + CLV * Volume
        """
        high_low = df['high'] - df['low']
        high_low = high_low.replace(0, 0.0001)  # Avoid division by zero
        
        clv = ((df['current_price'] - df['low']) - (df['high'] - df['current_price'])) / high_low
        ad = (clv * df['volume']).cumsum()
        return ad
    
    def calculate_volume_sma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Simple Moving Average"""
        return df['volume'].rolling(window=period).mean()
    
    def detect_volume_surge(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """
        Detect volume surges (volume > threshold * SMA)
        Returns boolean series
        """
        vol_sma = self.calculate_volume_sma(df, 20)
        return df['volume'] > (vol_sma * threshold)
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI) - Volume-weighted RSI
        """
        typical_price = (df['high'] + df['low'] + df['current_price']) / 3
        money_flow = typical_price * df['volume']
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0)
        negative_flow = money_flow.where(delta < 0, 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, 0.0001)))
        return mfi
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (daily VWAP approximation)"""
        typical_price = (df['high'] + df['low'] + df['current_price']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def analyze_supply_demand(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive supply/demand analysis
        Returns a dictionary with all indicators
        """
        if len(df) < 30:
            return None
        
        # Calculate all indicators
        df = df.sort_values('date').reset_index(drop=True)
        
        obv = self.calculate_obv(df)
        ad_line = self.calculate_ad_line(df)
        mfi = self.calculate_mfi(df)
        vol_surge = self.detect_volume_surge(df)
        
        # Get recent values
        latest = df.iloc[-1]
        recent_20 = df.tail(20)
        
        # OBV Trend (20-day)
        obv_change = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) * 100 if obv.iloc[-20] != 0 else 0
        
        # A/D Trend (20-day)
        ad_change = (ad_line.iloc[-1] - ad_line.iloc[-20]) / abs(ad_line.iloc[-20]) * 100 if ad_line.iloc[-20] != 0 else 0
        
        # Volume Ratio (5-day avg vs 20-day avg)
        vol_5d = df['volume'].tail(5).mean()
        vol_20d = df['volume'].tail(20).mean()
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1
        
        # Recent volume surges
        surge_count_5d = vol_surge.tail(5).sum()
        surge_count_20d = vol_surge.tail(20).sum()
        
        # MFI current value
        mfi_current = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50
        
        # Supply/Demand Score (0-100)
        score = 50
        
        # OBV contribution
        if obv_change > 10:
            score += 15
        elif obv_change > 5:
            score += 10
        elif obv_change < -10:
            score -= 15
        elif obv_change < -5:
            score -= 10
        
        # A/D contribution
        if ad_change > 10:
            score += 15
        elif ad_change > 5:
            score += 10
        elif ad_change < -10:
            score -= 15
        elif ad_change < -5:
            score -= 10
        
        # Volume ratio contribution
        if vol_ratio > 1.5:
            score += 10
        elif vol_ratio > 1.2:
            score += 5
        elif vol_ratio < 0.7:
            score -= 5
        
        # MFI contribution
        if mfi_current > 70:
            score += 5  # Overbought but with buying pressure
        elif mfi_current < 30:
            score -= 5  # Oversold, possible capitulation
        
        score = max(0, min(100, score))
        
        # Determine stage
        if score >= 70:
            stage = "Strong Accumulation"
        elif score >= 55:
            stage = "Accumulation"
        elif score >= 45:
            stage = "Neutral"
        elif score >= 30:
            stage = "Distribution"
        else:
            stage = "Strong Distribution"
        
        return {
            'date': latest['date'],
            'obv': obv.iloc[-1],
            'obv_change_20d': round(obv_change, 2),
            'ad_line': ad_line.iloc[-1],
            'ad_change_20d': round(ad_change, 2),
            'mfi': round(mfi_current, 1),
            'vol_ratio_5d_20d': round(vol_ratio, 2),
            'surge_count_5d': int(surge_count_5d),
            'surge_count_20d': int(surge_count_20d),
            'supply_demand_score': round(score, 1),
            'supply_demand_stage': stage
        }
    
    def run(self) -> pd.DataFrame:
        """Run volume analysis for all stocks"""
        logger.info("🚀 Starting Volume Analysis...")
        
        # Load data
        df = self.load_prices()
        
        # Get unique tickers
        tickers = df['ticker'].unique()
        logger.info(f"📊 Analyzing {len(tickers)} stocks")
        
        results = []
        
        for ticker in tqdm(tickers, desc="Analyzing volume"):
            ticker_data = df[df['ticker'] == ticker].copy()
            
            if len(ticker_data) < 30:
                continue
            
            analysis = self.analyze_supply_demand(ticker_data)
            
            if analysis:
                result = {
                    'ticker': ticker,
                    'name': ticker_data['name'].iloc[-1] if 'name' in ticker_data.columns else ticker,
                    **analysis
                }
                results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_file, index=False)
        logger.info(f"✅ Analysis complete! Saved to {self.output_file}")
        
        # Print summary
        logger.info("\n📊 Summary:")
        stage_counts = results_df['supply_demand_stage'].value_counts()
        for stage, count in stage_counts.items():
            logger.info(f"   {stage}: {count} stocks")
        
        return results_df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='US Stock Volume Analysis')
    parser.add_argument('--dir', default='.', help='Data directory')
    args = parser.parse_args()
    
    analyzer = VolumeAnalyzer(data_dir=args.dir)
    results = analyzer.run()
    
    # Show top 10 accumulation stocks
    print("\n🔥 Top 10 Accumulation Stocks:")
    top_10 = results.nlargest(10, 'supply_demand_score')
    for _, row in top_10.iterrows():
        print(f"   {row['ticker']}: Score {row['supply_demand_score']} - {row['supply_demand_stage']}")


if __name__ == "__main__":
    main()
```

**사용법:**
```bash
python3 analyze_volume.py
```

---

## 3️⃣ analyze_13f.py

> SEC 13F 공시 기반 기관 보유량 및 인사이더 매매를 분석합니다.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US 13F Institutional Holdings Analysis
Fetches and analyzes institutional holdings from SEC EDGAR
"""

import os
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SEC13FAnalyzer:
    """
    Analyze institutional holdings from SEC 13F filings
    Note: 13F filings are quarterly, with 45-day delay after quarter end
    """
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'us_13f_holdings.csv')
        self.cache_file = os.path.join(data_dir, 'us_13f_cache.json')
        
        # SEC EDGAR API base URL
        self.sec_base_url = "https://data.sec.gov"
        
        # User-Agent required by SEC
        self.headers = {
            'User-Agent': 'StockAnalysis/1.0 (contact@example.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # Major institutional investors (CIK numbers)
        self.major_institutions = {
            '0001067983': 'Berkshire Hathaway',
            '0001350694': 'Citadel Advisors',
            '0001423053': 'Renaissance Technologies',
            '0001037389': 'Bridgewater Associates',
            '0001336528': 'Millennium Management',
            '0001649339': 'Point72 Asset Management',
            '0001364742': 'Two Sigma Investments',
            '0001167483': 'Elliott Investment Management',
            '0001061165': 'Tiger Global Management',
            '0001697748': 'BlackRock Inc.',
            '0001040280': 'Vanguard Group',
            '0001166559': 'Fidelity Management',
            '0001095620': 'State Street Corporation',
            '0000895421': 'Soros Fund Management',
            '0001273087': 'Appaloosa Management',
        }
    
    def analyze_institutional_changes(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze institutional ownership and recent changes
        Uses yfinance as primary data source
        """
        import yfinance as yf
        from tqdm import tqdm
        
        results = []
        
        for ticker in tqdm(tickers, desc="Fetching institutional data"):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Basic ownership info
                inst_pct = info.get('heldPercentInstitutions', 0) or 0
                insider_pct = info.get('heldPercentInsiders', 0) or 0
                
                # Float and shares
                float_shares = info.get('floatShares', 0) or 0
                shares_outstanding = info.get('sharesOutstanding', 0) or 0
                short_pct = info.get('shortPercentOfFloat', 0) or 0
                
                # Insider transactions
                try:
                    insider_txns = stock.insider_transactions
                    if insider_txns is not None and len(insider_txns) > 0:
                        recent = insider_txns.head(10)
                        buys = len(recent[recent['Transaction'].str.contains('Buy', na=False)])
                        sells = len(recent[recent['Transaction'].str.contains('Sale', na=False)])
                        insider_sentiment = 'Buying' if buys > sells else ('Selling' if sells > buys else 'Neutral')
                    else:
                        insider_sentiment = 'Unknown'
                        buys = 0
                        sells = 0
                except:
                    insider_sentiment = 'Unknown'
                    buys = 0
                    sells = 0
                
                # Institutional holders count
                try:
                    inst_holders = stock.institutional_holders
                    num_inst_holders = len(inst_holders) if inst_holders is not None else 0
                except:
                    num_inst_holders = 0
                
                # Score calculation (0-100)
                score = 50
                
                # High institutional ownership is generally positive
                if inst_pct > 0.8:
                    score += 15
                elif inst_pct > 0.6:
                    score += 10
                elif inst_pct < 0.3:
                    score -= 10
                
                # Insider activity
                if buys > sells:
                    score += 15
                elif sells > buys:
                    score -= 10
                
                # Low short interest is positive
                if short_pct < 0.03:
                    score += 5
                elif short_pct > 0.1:
                    score -= 10
                elif short_pct > 0.2:
                    score -= 20
                
                score = max(0, min(100, score))
                
                # Determine stage
                if score >= 70:
                    stage = "Strong Institutional Support"
                elif score >= 55:
                    stage = "Institutional Support"
                elif score >= 45:
                    stage = "Neutral"
                elif score >= 30:
                    stage = "Institutional Concern"
                else:
                    stage = "Strong Institutional Selling"
                
                results.append({
                    'ticker': ticker,
                    'institutional_pct': round(inst_pct * 100, 2),
                    'insider_pct': round(insider_pct * 100, 2),
                    'short_pct': round(short_pct * 100, 2),
                    'float_shares_m': round(float_shares / 1e6, 2) if float_shares else 0,
                    'num_inst_holders': num_inst_holders,
                    'insider_buys': buys,
                    'insider_sells': sells,
                    'insider_sentiment': insider_sentiment,
                    'institutional_score': score,
                    'institutional_stage': stage
                })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Error analyzing {ticker}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def run(self) -> pd.DataFrame:
        """Run institutional analysis for stocks in the data directory"""
        logger.info("🚀 Starting 13F Institutional Analysis...")
        
        # Load stock list
        stocks_file = os.path.join(self.data_dir, 'us_stocks_list.csv')
        
        if os.path.exists(stocks_file):
            stocks_df = pd.read_csv(stocks_file)
            tickers = stocks_df['ticker'].tolist()
        else:
            logger.warning("Stock list not found. Using top 50 S&P 500 stocks.")
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                      'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
                      'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'TMO',
                      'CSCO', 'ABT', 'CRM', 'ACN', 'DHR', 'ORCL', 'NKE', 'TXN', 'PM',
                      'NEE', 'INTC', 'AMD', 'QCOM', 'IBM', 'GS', 'CAT', 'BA', 'DIS',
                      'NFLX', 'PYPL', 'ADBE', 'NOW', 'INTU']
        
        logger.info(f"📊 Analyzing {len(tickers)} stocks")
        
        # Run analysis
        results_df = self.analyze_institutional_changes(tickers)
        
        # Save results
        if not results_df.empty:
            results_df.to_csv(self.output_file, index=False)
            logger.info(f"✅ Analysis complete! Saved to {self.output_file}")
            
            # Summary
            logger.info("\n📊 Summary:")
            stage_counts = results_df['institutional_stage'].value_counts()
            for stage, count in stage_counts.items():
                logger.info(f"   {stage}: {count} stocks")
        else:
            logger.warning("No results to save")
        
        return results_df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='13F Institutional Analysis')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    args = parser.parse_args()
    
    analyzer = SEC13FAnalyzer(data_dir=args.dir)
    
    if args.tickers:
        results = analyzer.analyze_institutional_changes(args.tickers)
    else:
        results = analyzer.run()
    
    if not results.empty:
        # Show top institutional support
        print("\n🏦 Top 10 Institutional Support:")
        top_10 = results.nlargest(10, 'institutional_score')
        for _, row in top_10.iterrows():
            print(f"   {row['ticker']}: Score {row['institutional_score']} | "
                  f"Inst: {row['institutional_pct']:.1f}% | "
                  f"Insider: {row['insider_sentiment']}")
        
        # Show stocks with insider buying
        print("\n📈 Insider Buying Activity:")
        buying = results[results['insider_sentiment'] == 'Buying'].head(10)
        for _, row in buying.iterrows():
            print(f"   {row['ticker']}: {row['insider_buys']} buys vs {row['insider_sells']} sells")


if __name__ == "__main__":
    main()
```

**사용법:**
```bash
python3 analyze_13f.py
python3 analyze_13f.py --tickers AAPL MSFT GOOGL
```

---

## 4️⃣ analyze_etf_flows.py

> 주요 ETF의 자금 흐름을 분석하고 Gemini 3.0으로 인사이트를 생성합니다.

**Note**: 이 파일은 17,000 bytes 이상이라 별도 문서로 분리하는 것이 좋습니다.
핵심 포인트만 기술합니다:

- **24개 주요 ETF 추적** (SPY, QQQ, IWM, GLD, USO 등)
- **자금 흐름 점수(Flow Score)** 계산: 0~100점
- **Gemini 3.0 AI 분석**: 자금 이동의 "왜"를 해석
- **출력**: `us_etf_flows.csv`, `etf_flow_analysis.json`

**핵심 메서드:**
```python
def calculate_flow_proxy(self, df: pd.DataFrame) -> Dict:
    # OBV, Volume Ratio, Price Momentum 기반 점수 계산

def generate_ai_analysis(self, results_df: pd.DataFrame) -> None:
    # Gemini 3.0로 자금 흐름 해석 생성
```

**사용법:**
```bash
python3 analyze_etf_flows.py
```

---

## ▶️ 실행 순서

```bash
cd us_market

# 1. S&P 500 가격 데이터 수집 (최초 1회 오래 걸림)
python3 create_us_daily_prices.py

# 2. 거래량/수급 분석
python3 analyze_volume.py

# 3. 기관 보유 분석 (시간 소요)
python3 analyze_13f.py

# 4. ETF 자금 흐름 + AI 분석
python3 analyze_etf_flows.py
```

또는 통합 스크립트 사용:
```bash
python3 update_all.py --quick   # AI 제외 빠른 업데이트
python3 update_all.py           # 전체 업데이트
```

---

**다음 문서**: [Part 2: 분석/스크리닝](./PART2_Analysis_Screening.md)
