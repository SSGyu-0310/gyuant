#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Smart Money Screener v2.0
Comprehensive analysis combining:
- Volume/Accumulation Analysis
- Technical Analysis (RSI, MACD, MA)
- Fundamental Analysis (P/E, P/B, Growth)
- Analyst Ratings
- Relative Strength vs S&P 500
"""

import os
import sys
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env

load_env()

from utils.fmp_client import FMPClient
from utils.symbols import to_fmp_symbol
from utils.db_writer import get_db_connection

# PostgreSQL toggle
USE_POSTGRES = os.getenv('USE_POSTGRES', 'true').lower() == 'true'

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PriceStore:
    """One-time loader for local price data with helper accessors."""

    def __init__(self, data_dir: str, lookback_days: int = 400):
        self.data_dir = Path(data_dir)
        self.lookback_days = lookback_days
        self.df: Optional[pd.DataFrame] = None
        self.loaded = False
        self.has_spy = False
        self._load()

    def _load(self) -> None:
        # Try PostgreSQL first
        if USE_POSTGRES:
            try:
                from utils.data_access import get_prices
                logger.info("PriceStore: loading from PostgreSQL...")
                df = get_prices()
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # Map 'close' to 'current_price' for compatibility
                    if 'close' in df.columns and 'current_price' not in df.columns:
                        df['current_price'] = df['close']
                    df = df.dropna(subset=['date', 'ticker'])
                    df['ticker'] = df['ticker'].astype('category')
                    if 'name' in df.columns:
                        df['name'] = df['name'].astype('category')
                    # Keep only recent window
                    cutoff = pd.Timestamp(datetime.utcnow().date() - timedelta(days=self.lookback_days + 10))
                    df = df[df['date'] >= cutoff]
                    df = df.sort_values(['ticker', 'date'])
                    self.has_spy = (df['ticker'] == 'SPY').any()
                    self.df = df
                    self.loaded = True
                    logger.info("PriceStore: loaded %d rows (%d tickers) from PostgreSQL", len(df), df['ticker'].nunique())
                    return
                logger.info("PriceStore: PostgreSQL empty, falling back to CSV")
            except Exception as exc:
                logger.warning("PriceStore: PostgreSQL load failed: %s, falling back to CSV", exc)
        
        # CSV Fallback
        price_file = self.data_dir / "us_daily_prices.csv"
        if not price_file.exists():
            logger.warning("PriceStore: local price file not found: %s", price_file)
            return
        try:
            cutoff = (datetime.utcnow().date() - timedelta(days=self.lookback_days + 10)).isoformat()
            query = """
                SELECT
                    p.ticker,
                    p.date,
                    p.open,
                    p.high,
                    p.low,
                    p.close AS current_price,
                    p.volume,
                    s.name
                FROM market_prices_daily p
                LEFT JOIN market_stocks s ON p.ticker = s.ticker
                WHERE p.date >= ?
            """
            df = pd.read_sql_query(query, conn, params=[cutoff])
            if df.empty:
                logger.warning("PriceStore: market_prices_daily is empty")
                return
            if "date" not in df.columns or "ticker" not in df.columns:
                logger.warning("PriceStore: required columns missing in market_prices_daily")
                return
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "ticker"])
            df["ticker"] = df["ticker"].astype("category")
            if "name" in df.columns:
                df["name"] = df["name"].astype("category")
            dtype = {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "current_price": "float64",
                "volume": "Int64",
            }
            for col in ("open", "high", "low", "current_price", "volume"):
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype.get(col, "float64"))
                    except Exception:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values(["ticker", "date"])
            self.has_spy = (df["ticker"] == "SPY").any()
            self.df = df
            self.loaded = True
            logger.info(
                "PriceStore: loaded %d rows (%d tickers, SPY present=%s) from SQLite",
                len(df),
                df["ticker"].nunique(),
                self.has_spy,
            )
        except Exception as exc:
            logger.warning("PriceStore: failed to load from SQLite: %s", exc)
            self.df = None
            self.loaded = False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def has_data(self, ticker: str) -> bool:
        return bool(self.loaded and self.df is not None and not self.df[self.df["ticker"] == ticker].empty)

    def get_history(self, ticker: str, days: int) -> pd.DataFrame:
        if not self.has_data(ticker):
            return pd.DataFrame()
        df = self.df[self.df["ticker"] == ticker]
        cutoff = pd.Timestamp(datetime.utcnow().date() - timedelta(days=days + 5))
        df = df[df["date"] >= cutoff]
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df = df.rename(columns={"date": "Date", "current_price": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
        # Ensure required columns exist
        for col in ("Open", "High", "Low", "Volume"):
            if col not in df.columns:
                df[col] = np.nan
        df = df[["Date", "Close", "Open", "High", "Low", "Volume"]]
        df = df.sort_values("Date")
        return df

    def get_latest_price(self, ticker: str) -> Optional[float]:
        if not self.has_data(ticker):
            return None
        series = self.df.loc[self.df["ticker"] == ticker, "current_price"]
        if series.empty:
            return None
        val = series.iloc[-1]
        return float(val) if pd.notna(val) else None

    def get_name(self, ticker: str) -> Optional[str]:
        if not self.has_data(ticker) or "name" not in self.df.columns:
            return None
        series = self.df.loc[self.df["ticker"] == ticker, "name"]
        if series.empty:
            return None
        val = series.iloc[-1]
        return str(val) if pd.notna(val) else None


class EnhancedSmartMoneyScreener:
    """
    Enhanced screener with comprehensive analysis:
    1. Supply/Demand (volume analysis)
    2. Technical Analysis (RSI, MACD, MA)
    3. Fundamentals (valuation, growth)
    4. Analyst Ratings
    5. Relative Strength
    """
    
    TA_LOOKBACK_DAYS = int(os.getenv("SMART_MONEY_TA_DAYS", "400"))

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.getenv("DATA_DIR", ".")
        self.output_file = os.path.join(self.data_dir, 'smart_money_picks_v2.csv')
        self.price_store = PriceStore(self.data_dir, lookback_days=self.TA_LOOKBACK_DAYS)
        self._tech_table: Optional[pd.DataFrame] = None
        
        # Load analysis data
        self.volume_df = None
        self.etf_df = None
        self.prices_df = None
        
        self._client_local = threading.local()
        self._cache_lock = threading.Lock()
        self.profile_cache = {}
        self.quote_cache = {}
        self.metrics_cache = {}
        self.ratios_cache = {}
        self.history_cache = {}
        self.api_counts = defaultdict(int)
        
        # S&P 500 benchmark data
        self.spy_data = None
        
    def load_data(self) -> bool:
        """Load all analysis results"""
        try:
            conn = get_db_connection()
            if conn is None:
                logger.warning("⚠️ SQLite connection unavailable")
                return False

            try:
                # Volume Analysis (latest snapshot)
                volume_query = """
                    SELECT *
                    FROM market_volume_analysis
                    WHERE as_of_date = (SELECT MAX(as_of_date) FROM market_volume_analysis)
                """
                self.volume_df = pd.read_sql_query(volume_query, conn)
                if self.volume_df.empty:
                    logger.warning("⚠️ Volume analysis not found in SQLite")
                    return False
                logger.info("✅ Loaded volume analysis from SQLite: %d stocks", len(self.volume_df))

                # ETF Flows (latest snapshot)
                etf_query = """
                    SELECT *
                    FROM market_etf_flows
                    WHERE as_of_date = (SELECT MAX(as_of_date) FROM market_etf_flows)
                """
                self.etf_df = pd.read_sql_query(etf_query, conn)
                if self.etf_df.empty:
                    logger.warning("⚠️ ETF flows not found in SQLite")
                    self.etf_df = None
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            
            # Load SPY for relative strength
            logger.info("📈 Loading SPY benchmark data...")
            self.spy_data = self._get_history("SPY", days=self.TA_LOOKBACK_DAYS)
            # Pre-compute technical metrics from local prices (vectorized)
            self._precompute_technical_table()
            if not self.price_store.has_spy:
                logger.warning("PriceStore: SPY not found locally; SPY history may fallback to API once.")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False

    def _get_client(self) -> FMPClient:
        client = getattr(self._client_local, "client", None)
        if client is None:
            client = FMPClient()
            self._client_local.client = client
        return client

    def _get_cached(self, cache: Dict[Any, Any], key: Any, loader):
        with self._cache_lock:
            if key in cache:
                return cache[key]
        value = loader()
        with self._cache_lock:
            cache[key] = value
        return value

    def _count_api(self, name: str) -> None:
        self.api_counts[name] += 1

    def _get_profile(self, ticker: str) -> Dict:
        return self._get_cached(
            self.profile_cache,
            ticker,
            lambda: self._count_api("profile") or self._get_client().profile(to_fmp_symbol(ticker)),
        )

    def _get_quote(self, ticker: str) -> Dict:
        def loader():
            self._count_api("quote")
            quotes = self._get_client().quote([to_fmp_symbol(ticker)])
            return quotes[0] if quotes else {}

        return self._get_cached(self.quote_cache, ticker, loader)

    def _get_key_metrics(self, ticker: str) -> Dict:
        return self._get_cached(
            self.metrics_cache,
            ticker,
            lambda: self._count_api("key_metrics") or self._get_client().key_metrics_ttm(to_fmp_symbol(ticker)),
        )

    def _get_ratios(self, ticker: str) -> Dict:
        return self._get_cached(
            self.ratios_cache,
            ticker,
            lambda: self._count_api("ratios") or self._get_client().ratios_ttm(to_fmp_symbol(ticker)),
        )

    def _get_local_meta(self, ticker: str) -> Dict[str, Any]:
        """Return lightweight local metadata: latest close and name if available."""
        name = self.price_store.get_name(ticker)
        if name is None and self.volume_df is not None and not self.volume_df.empty and "name" in self.volume_df.columns:
            name_series = self.volume_df.loc[self.volume_df["ticker"] == ticker, "name"]
            if not name_series.empty:
                name = name_series.iloc[0]
        price = self.price_store.get_latest_price(ticker)
        if price is None and self._tech_table is not None and ticker in self._tech_table.index:
            val = self._tech_table.loc[ticker].get("last_close")
            if val is not None and not pd.isna(val):
                price = float(val)
        return {"name": name, "price": price}

    def _ensure_local_prices(self) -> Optional[pd.DataFrame]:
        return self.price_store.df

    def _get_local_history(self, ticker: str, days: int) -> pd.DataFrame:
        return self.price_store.get_history(ticker, days)

    def _precompute_technical_table(self) -> None:
        """
        Vectorized technical computation for all tickers using local prices.
        Falls back silently if local data is missing.
        """
        df = self._ensure_local_prices()
        if df is None or df.empty:
            return
        price_col = None
        for cand in ("Close", "close", "current_price"):
            if cand in df.columns:
                price_col = cand
                break
        if price_col is None or "ticker" not in df.columns or "date" not in df.columns:
            return

        df = df.rename(columns={price_col: "Close", "date": "Date"})
        df = df[["ticker", "Date", "Close"]].dropna()
        if df.empty:
            return
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["ticker", "Date"])

        records: List[Dict[str, Any]] = []

        def calc_metrics(group: pd.DataFrame) -> Optional[Dict[str, Any]]:
            close = group["Close"]
            if close.empty:
                return None
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal

            ma20 = close.rolling(20, min_periods=20).mean()
            ma50 = close.rolling(50, min_periods=50).mean()
            ma200 = close.rolling(200, min_periods=200).mean()

            ret_20 = np.nan
            ret_60 = np.nan
            if len(close) >= 21:
                ret_20 = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100
            if len(close) >= 61:
                ret_60 = ((close.iloc[-1] / close.iloc[-61]) - 1) * 100

            if len(close) < 50:
                tech_score = 50
                ma_signal = "Neutral"
                cross_signal = "None"
            else:
                current_price = close.iloc[-1]
                m20 = ma20.iloc[-1]
                m50 = ma50.iloc[-1]
                m200 = ma200.iloc[-1] if len(close) >= 200 else np.nan
                ma_signal = "Neutral"
                if pd.notna(m20) and pd.notna(m50):
                    if current_price > m20 > m50:
                        ma_signal = "Bullish"
                    elif current_price < m20 < m50:
                        ma_signal = "Bearish"
                ma50_prev = ma50.iloc[-5] if len(ma50) >= 5 else np.nan
                ma200_prev = ma200.iloc[-5] if len(ma200) >= 5 else np.nan
                if (
                    pd.notna(m50) and pd.notna(m200) and pd.notna(ma50_prev) and pd.notna(ma200_prev)
                    and m50 > m200 and ma50_prev <= ma200_prev
                ):
                    cross_signal = "Golden Cross"
                elif (
                    pd.notna(m50) and pd.notna(m200) and pd.notna(ma50_prev) and pd.notna(ma200_prev)
                    and m50 < m200 and ma50_prev >= ma200_prev
                ):
                    cross_signal = "Death Cross"
                else:
                    cross_signal = "None"

                rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else 50
                macd_hist_current = macd_hist.iloc[-1]
                tech_score = 50
                if 40 <= rsi_val <= 60:
                    tech_score += 10
                elif rsi_val < 30:
                    tech_score += 15
                elif rsi_val > 70:
                    tech_score -= 5
                if macd_hist_current > 0 and len(macd_hist) > 1 and macd_hist.iloc[-2] < 0:
                    tech_score += 15
                elif macd_hist_current > 0:
                    tech_score += 8
                elif macd_hist_current < 0:
                    tech_score -= 5
                if ma_signal == "Bullish":
                    tech_score += 15
                elif ma_signal == "Bearish":
                    tech_score -= 10
                if cross_signal == "Golden Cross":
                    tech_score += 10
                elif cross_signal == "Death Cross":
                    tech_score -= 15
                tech_score = max(0, min(100, tech_score))

            return {
                "ticker": group["ticker"].iloc[0],
                "rsi": float(rsi_series.iloc[-1]) if not rsi_series.empty else np.nan,
                "macd": float(macd.iloc[-1]),
                "macd_signal": float(signal.iloc[-1]),
                "macd_histogram": float(macd_hist.iloc[-1]),
                "ma20": float(ma20.iloc[-1]) if not ma20.empty else np.nan,
                "ma50": float(ma50.iloc[-1]) if not ma50.empty else np.nan,
                "ma200": float(ma200.iloc[-1]) if not ma200.empty else np.nan,
                "ma_signal": ma_signal,
                "cross_signal": cross_signal,
                "technical_score": tech_score,
                "ret_20d": ret_20,
                "ret_60d": ret_60,
                "last_close": float(close.iloc[-1]),
            }

        for _, group in df.groupby("ticker"):
            metrics = calc_metrics(group)
            if metrics:
                records.append(metrics)

        if not records:
            return

        tech_df = pd.DataFrame(records)
        spy_row = tech_df[tech_df["ticker"] == "SPY"]
        if not spy_row.empty:
            spy20 = float(spy_row["ret_20d"].iloc[0]) if not spy_row["ret_20d"].isna().all() else np.nan
            spy60 = float(spy_row["ret_60d"].iloc[0]) if not spy_row["ret_60d"].isna().all() else np.nan
            tech_df["rs_20d"] = tech_df["ret_20d"] - spy20 if not np.isnan(spy20) else np.nan
            tech_df["rs_60d"] = tech_df["ret_60d"] - spy60 if not np.isnan(spy60) else np.nan
        else:
            tech_df["rs_20d"] = np.nan
            tech_df["rs_60d"] = np.nan

        self._tech_table = tech_df.set_index("ticker")

    def _get_history(self, ticker: str, days: int = 180) -> pd.DataFrame:
        cache_key = (ticker, days)

        def loader():
            local_df = self._get_local_history(ticker, days)
            if not local_df.empty:
                self._count_api("history_local")
                return local_df
            end_date = datetime.utcnow().date()
            from_date = (end_date - timedelta(days=days)).isoformat()
            to_date = end_date.isoformat()
            self._count_api("history")
            data = self._get_client().historical_price_full(
                to_fmp_symbol(ticker),
                from_date=from_date,
                to_date=to_date,
            )
            hist_list = data.get("historical", []) if isinstance(data, dict) else []
            if not hist_list:
                return pd.DataFrame()
            df = pd.DataFrame(hist_list)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.sort_values('Date')
            df = df.rename(columns={'close': 'Close'})
            return df

        return self._get_cached(self.history_cache, cache_key, loader)
    
    def get_technical_analysis(self, ticker: str) -> Dict:
        """Calculate technical indicators"""
        try:
            if self._tech_table is not None and ticker in self._tech_table.index:
                row = self._tech_table.loc[ticker]
                return {
                    'rsi': float(row.get("rsi", 50)) if not pd.isna(row.get("rsi", np.nan)) else 50,
                    'macd': float(row.get("macd", 0)),
                    'macd_signal': float(row.get("macd_signal", 0)),
                    'macd_histogram': float(row.get("macd_histogram", 0)),
                    'ma20': float(row.get("ma20", 0)) if not pd.isna(row.get("ma20", np.nan)) else 0,
                    'ma50': float(row.get("ma50", 0)) if not pd.isna(row.get("ma50", np.nan)) else 0,
                    'ma_signal': row.get("ma_signal", "Neutral"),
                    'cross_signal': row.get("cross_signal", "None"),
                    'technical_score': float(row.get("technical_score", 50)),
                }

            hist = self._get_history(ticker, days=self.TA_LOOKBACK_DAYS)
            
            if len(hist) < 50:
                return self._default_technical()
            
            close = hist['Close']
            
            # RSI (14-day)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            
            macd_current = macd.iloc[-1]
            signal_current = signal.iloc[-1]
            macd_hist_current = macd_histogram.iloc[-1]
            
            # Moving Averages
            ma20 = close.rolling(20, min_periods=20).mean().iloc[-1]
            ma50_series = close.rolling(50, min_periods=50).mean()
            ma50 = ma50_series.iloc[-1]
            ma200_series = close.rolling(200, min_periods=200).mean()
            ma200 = ma200_series.iloc[-1]
            current_price = close.iloc[-1]
            
            # MA Arrangement
            if pd.notna(ma20) and pd.notna(ma50) and current_price > ma20 > ma50:
                ma_signal = "Bullish"
            elif pd.notna(ma20) and pd.notna(ma50) and current_price < ma20 < ma50:
                ma_signal = "Bearish"
            else:
                ma_signal = "Neutral"
            
            # Golden/Death Cross
            ma50_prev = ma50_series.iloc[-5] if len(ma50_series) >= 5 else np.nan
            ma200_prev = ma200_series.iloc[-5] if len(ma200_series) >= 5 else np.nan
            
            if pd.notna(ma50) and pd.notna(ma200) and pd.notna(ma50_prev) and pd.notna(ma200_prev) and ma50 > ma200 and ma50_prev <= ma200_prev:
                cross_signal = "Golden Cross"
            elif pd.notna(ma50) and pd.notna(ma200) and pd.notna(ma50_prev) and pd.notna(ma200_prev) and ma50 < ma200 and ma50_prev >= ma200_prev:
                cross_signal = "Death Cross"
            else:
                cross_signal = "None"
            
            # Technical Score (0-100)
            tech_score = 50
            
            # RSI contribution
            if 40 <= current_rsi <= 60:
                tech_score += 10  # Neutral zone - room to move
            elif current_rsi < 30:
                tech_score += 15  # Oversold - potential bounce
            elif current_rsi > 70:
                tech_score -= 5   # Overbought
            
            # MACD contribution
            if macd_hist_current > 0 and macd_histogram.iloc[-2] < 0:
                tech_score += 15  # Bullish crossover
            elif macd_hist_current > 0:
                tech_score += 8
            elif macd_hist_current < 0:
                tech_score -= 5
            
            # MA contribution
            if ma_signal == "Bullish":
                tech_score += 15
            elif ma_signal == "Bearish":
                tech_score -= 10
            
            if cross_signal == "Golden Cross":
                tech_score += 10
            elif cross_signal == "Death Cross":
                tech_score -= 15
            
            tech_score = max(0, min(100, tech_score))
            
            return {
                'rsi': round(current_rsi, 1),
                'macd': round(macd_current, 3),
                'macd_signal': round(signal_current, 3),
                'macd_histogram': round(macd_hist_current, 3),
                'ma20': round(ma20, 2),
                'ma50': round(ma50, 2),
                'ma_signal': ma_signal,
                'cross_signal': cross_signal,
                'technical_score': tech_score
            }
            
        except Exception as e:
            return self._default_technical()
    
    def _default_technical(self) -> Dict:
        return {
            'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_histogram': 0,
            'ma20': 0, 'ma50': 0, 'ma_signal': 'Unknown', 'cross_signal': 'None',
            'technical_score': 50
        }
    
    def get_fundamental_analysis(self, ticker: str) -> Dict:
        """Get fundamental/valuation metrics"""
        try:
            profile = self._get_profile(ticker)
            metrics = self._get_key_metrics(ticker)
            ratios = self._get_ratios(ticker)
            # Keep internal representation numeric (np.nan instead of 'N/A')
            def _num(val: Any) -> float:
                try:
                    if val is None or (isinstance(val, (float, int)) and np.isnan(val)):
                        return np.nan
                    return float(val)
                except Exception:
                    return np.nan
            
            # Valuation
            pe_ratio_raw = metrics.get('peRatioTTM') or ratios.get('priceEarningsRatioTTM')
            forward_pe_raw = profile.get('pe') or ratios.get('priceEarningsRatioTTM')
            pb_ratio_raw = ratios.get('priceToBookRatioTTM') or metrics.get('pbRatioTTM')
            
            # Growth (fallback to 0 if not available)
            revenue_growth = profile.get('revenueGrowth') or 0
            earnings_growth = profile.get('earningsGrowth') or 0
            
            # Profitability
            profit_margin = ratios.get('netProfitMarginTTM') or metrics.get('netProfitMarginTTM') or 0
            roe = metrics.get('roeTTM') or ratios.get('returnOnEquityTTM') or 0
            
            # Market Cap
            market_cap = profile.get('mktCap') or profile.get('marketCap')
            
            # Dividend
            dividend_yield = ratios.get('dividendYieldTTM')
            
            # Fundamental Score (0-100)
            fund_score = 50
            
            # P/E contribution (lower is better, but not too low)
            pe_ratio = _num(pe_ratio_raw)
            forward_pe = _num(forward_pe_raw)
            pb_ratio = _num(pb_ratio_raw)
            if 0 < pe_ratio < 15:
                fund_score += 15
            elif 15 <= pe_ratio < 25:
                fund_score += 10
            elif pe_ratio > 40:
                fund_score -= 10
            elif pe_ratio < 0:  # Negative earnings
                fund_score -= 15
            
            # Growth contribution
            if revenue_growth > 0.2:
                fund_score += 15
            elif revenue_growth > 0.1:
                fund_score += 10
            elif revenue_growth > 0:
                fund_score += 5
            elif revenue_growth < 0:
                fund_score -= 10
            
            # ROE contribution
            if roe > 0.2:
                fund_score += 10
            elif roe > 0.1:
                fund_score += 5
            elif roe < 0:
                fund_score -= 10
            
            fund_score = max(0, min(100, fund_score))
            
            # Size category
            if market_cap > 200e9:
                size = "Mega Cap"
            elif market_cap > 10e9:
                size = "Large Cap"
            elif market_cap > 2e9:
                size = "Mid Cap"
            elif market_cap > 300e6:
                size = "Small Cap"
            else:
                size = "Micro Cap"
            
            return {
                'pe_ratio': round(pe_ratio, 2) if not np.isnan(pe_ratio) else np.nan,
                'forward_pe': round(forward_pe, 2) if not np.isnan(forward_pe) else np.nan,
                'pb_ratio': round(pb_ratio, 2) if not np.isnan(pb_ratio) else np.nan,
                'revenue_growth': round(revenue_growth * 100, 1) if revenue_growth else 0,
                'earnings_growth': round(earnings_growth * 100, 1) if earnings_growth else 0,
                'profit_margin': round(profit_margin * 100, 1) if profit_margin else 0,
                'roe': round(roe * 100, 1) if roe else 0,
                'market_cap_b': round(market_cap / 1e9, 1) if market_cap else np.nan,
                'size': size,
                'dividend_yield': round(dividend_yield * 100, 2) if dividend_yield else np.nan,
                'fundamental_score': fund_score
            }
            
        except Exception as e:
            return self._default_fundamental()
    
    def _default_fundamental(self) -> Dict:
        return {
            'pe_ratio': np.nan, 'forward_pe': np.nan, 'pb_ratio': np.nan,
            'revenue_growth': 0, 'earnings_growth': 0, 'profit_margin': 0,
            'roe': 0, 'market_cap_b': np.nan, 'size': 'Unknown', 'dividend_yield': np.nan,
            'fundamental_score': 50
        }
    
    def get_analyst_ratings(self, ticker: str) -> Dict:
        """Get analyst consensus and target price"""
        try:
            local_meta = self._get_local_meta(ticker)
            profile = None
            quote = None
            client = self._get_client()
            self._count_api("ratings_snapshot")
            ratings = client.ratings_snapshot(to_fmp_symbol(ticker))
            self._count_api("price_target_consensus")
            consensus = client.price_target_consensus(to_fmp_symbol(ticker))
            
            # Prefer local name/price; fallback to profile/quote only if missing
            company_name = local_meta.get("name")
            current_price = local_meta.get("price")
            if company_name is None or current_price is None:
                profile = self._get_profile(ticker)
                quote = self._get_quote(ticker)
            company_name = company_name or (profile.get('companyName') if profile else None) or (profile.get('name') if profile else None) or ticker
            current_price = (
                current_price
                or (quote.get('price') if quote else None)
                or (profile.get('price') if profile else None)
                or 0
            )
            if not current_price or pd.isna(current_price):
                logger.warning("Current price missing for %s; skipping upside calc fallback to 0", ticker)
                current_price = 0
            target_price = (
                consensus.get('targetConsensus')
                or consensus.get('targetMedian')
                or consensus.get('targetMean')
                or 0
            )
            
            rec_raw = ratings.get('ratingRecommendation') or ratings.get('rating') or ''
            rec_norm = str(rec_raw).lower()
            if 'strong buy' in rec_norm:
                recommendation = 'strongBuy'
            elif 'buy' in rec_norm:
                recommendation = 'buy'
            elif 'hold' in rec_norm:
                recommendation = 'hold'
            elif 'strong sell' in rec_norm:
                recommendation = 'strongSell'
            elif 'sell' in rec_norm:
                recommendation = 'sell'
            else:
                recommendation = 'none'
            
            # Upside potential
            if current_price > 0 and target_price > 0:
                upside = ((target_price / current_price) - 1) * 100
            else:
                upside = 0
            
            # Analyst Score (0-100)
            analyst_score = 50
            
            # Recommendation contribution
            rec_map = {
                'strongBuy': 25, 'buy': 20, 'hold': 0,
                'sell': -15, 'strongSell': -25
            }
            analyst_score += rec_map.get(recommendation, 0)
            
            # Upside contribution
            if upside > 30: analyst_score += 20
            elif upside > 20: analyst_score += 15
            elif upside > 10: analyst_score += 10
            elif upside > 0: analyst_score += 5
            elif upside < -10: analyst_score -= 15
            
            analyst_score = max(0, min(100, analyst_score))
            
            return {
                'company_name': company_name,
                'current_price': round(current_price, 2),
                'target_price': round(target_price, 2) if target_price else 'N/A',
                'upside_pct': round(upside, 1),
                'recommendation': recommendation,
                'analyst_score': analyst_score
            }
            
        except Exception as e:
            return self._default_analyst()
            
    def _default_analyst(self) -> Dict:
        return {
            'company_name': '', 'current_price': 0, 'target_price': 'N/A',
            'upside_pct': 0, 'recommendation': 'none', 'analyst_score': 50
        }
    
    def get_relative_strength(self, ticker: str) -> Dict:
        """Calculate relative strength vs S&P 500"""
        try:
            if self._tech_table is not None and ticker in self._tech_table.index:
                row = self._tech_table.loc[ticker]
                spy_row = self._tech_table.loc["SPY"] if "SPY" in self._tech_table.index else None
                rs_20d = float(row.get("rs_20d", 0)) if not pd.isna(row.get("rs_20d", np.nan)) else 0
                rs_60d = float(row.get("rs_60d", 0)) if not pd.isna(row.get("rs_60d", np.nan)) else 0
                rs_score = 50 if spy_row is None else self._compute_rs_score(rs_20d, rs_60d)
                return {'rs_20d': round(rs_20d, 1), 'rs_60d': round(rs_60d, 1), 'rs_score': rs_score}

            if self.spy_data is None or len(self.spy_data) < 61:
                return {'rs_20d': 0, 'rs_60d': 0, 'rs_score': 50}
            
            hist = self._get_history(ticker, days=140)
            
            if len(hist) < 61:
                return {'rs_20d': 0, 'rs_60d': 0, 'rs_score': 50}
            
            # Calculate returns using trading-day indices
            stock_return_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
            stock_return_60d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-61] - 1) * 100
            
            spy_return_20d = (self.spy_data['Close'].iloc[-1] / self.spy_data['Close'].iloc[-21] - 1) * 100 if len(self.spy_data) >= 21 else 0
            spy_return_60d = (self.spy_data['Close'].iloc[-1] / self.spy_data['Close'].iloc[-61] - 1) * 100
            
            rs_20d = stock_return_20d - spy_return_20d
            rs_60d = stock_return_60d - spy_return_60d
            
            rs_score = self._compute_rs_score(rs_20d, rs_60d)
            
            return {
                'rs_20d': round(rs_20d, 1),
                'rs_60d': round(rs_60d, 1),
                'rs_score': rs_score
            }
            
        except Exception as e:
            return {'rs_20d': 0, 'rs_60d': 0, 'rs_score': 50}

    def _compute_rs_score(self, rs_20d: float, rs_60d: float) -> int:
        """Shared RS scoring logic."""
        rs_score = 50
        if rs_20d > 10: rs_score += 25
        elif rs_20d > 5: rs_score += 15
        elif rs_20d > 0: rs_score += 8
        elif rs_20d < -10: rs_score -= 20
        elif rs_20d < -5: rs_score -= 10

        if rs_60d > 15: rs_score += 15
        elif rs_60d > 5: rs_score += 8
        elif rs_60d < -15: rs_score -= 15

        return int(max(0, min(100, rs_score)))

    def _log_api_stats(self, total_candidates: int) -> None:
        """Log API usage summary."""
        if total_candidates <= 0:
            total_candidates = 1
        if not self.api_counts:
            return
        stats = {k: v for k, v in self.api_counts.items()}
        avg_per_ticker = {k: round(v / total_candidates, 3) for k, v in stats.items()}
        logger.info("API usage counts: %s", stats)
        logger.info("API usage avg per analyzed ticker: %s", avg_per_ticker)
    
    def calculate_composite_score(self, row: Mapping[str, Any], tech: Dict, fund: Dict, analyst: Dict, rs: Dict) -> Tuple[float, str]:
        """Calculate final composite score"""
        # Weighted composite score (renormalized weights)
        composite = (
            row.get('supply_demand_score', 50) * 0.3125 +
            tech.get('technical_score', 50) * 0.25 +
            fund.get('fundamental_score', 50) * 0.1875 +
            analyst.get('analyst_score', 50) * 0.125 +
            rs.get('rs_score', 50) * 0.125
        )
        
        # Determine grade
        if composite >= 80: grade = "🔥 S급 (즉시 매수)"
        elif composite >= 70: grade = "🌟 A급 (적극 매수)"
        elif composite >= 60: grade = "📈 B급 (매수 고려)"
        elif composite >= 50: grade = "📊 C급 (관망)"
        elif composite >= 40: grade = "⚠️ D급 (주의)"
        else: grade = "🚫 F급 (회피)"
        
        return round(composite, 1), grade
    
    def _resolve_workers(self, workers: Optional[int]) -> int:
        if workers is not None:
            return max(1, min(12, int(workers)))
        env_workers = os.getenv("SMART_MONEY_WORKERS")
        if env_workers:
            try:
                return max(1, min(12, int(env_workers)))
            except ValueError:
                logger.warning("SMART_MONEY_WORKERS must be an integer; using single-thread")
                return 1
        enable_threads = str(os.getenv("PERF_ENABLE_THREADS", "0")).lower() in ("1", "true", "yes")
        if enable_threads:
            try:
                return max(1, min(12, int(os.getenv("PERF_MAX_WORKERS", "6"))))
            except ValueError:
                return 6
        # Default safe parallelism for I/O-bound API
        return 6

    def _analyze_row(self, row: Mapping[str, Any]) -> Optional[Dict]:
        ticker = row.get('ticker')
        if not ticker:
            return None
        try:
            tech = self.get_technical_analysis(ticker)
            fund = self.get_fundamental_analysis(ticker)
            analyst = self.get_analyst_ratings(ticker)
            rs = self.get_relative_strength(ticker)

            composite_score, grade = self.calculate_composite_score(row, tech, fund, analyst, rs)

            return {
                'ticker': ticker,
                'name': analyst.get('company_name', ticker),
                'composite_score': composite_score,
                'grade': grade,
                'sd_score': row.get('supply_demand_score', 50),
                'tech_score': tech['technical_score'],
                'fund_score': fund['fundamental_score'],
                'analyst_score': analyst['analyst_score'],
                'rs_score': rs['rs_score'],
                'current_price': analyst['current_price'],
                'target_upside': analyst['upside_pct'],
                'rsi': tech['rsi'],
                'ma_signal': tech['ma_signal'],
                'pe_ratio': fund['pe_ratio'],
                'market_cap_b': fund['market_cap_b'],
                'size': fund['size'],
                'recommendation': analyst['recommendation'],
                'rs_20d': rs['rs_20d']
            }
        except Exception as exc:
            logger.warning("⚠️ Failed to analyze %s: %s", ticker, exc)
            return None

    def _apply_stage2_preselection(self, rows: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """
        Two-step screening:
        1) Rank by local-only signals (supply/demand + technical + RS).
        2) Run expensive API-based analysis only for top-K tickers.
        """
        if self._tech_table is None or self._tech_table.empty or not rows:
            return rows

        try:
            limit = int(os.getenv("SMART_MONEY_STAGE2_LIMIT", "180"))
        except ValueError:
            limit = 180

        if limit <= 0 or len(rows) <= limit:
            return rows

        try:
            w_sd = float(os.getenv("SMART_MONEY_STAGE2_W_SD", "0.4"))
            w_tech = float(os.getenv("SMART_MONEY_STAGE2_W_TECH", "0.4"))
            w_rs = float(os.getenv("SMART_MONEY_STAGE2_W_RS", "0.2"))
        except ValueError:
            w_sd, w_tech, w_rs = 0.4, 0.4, 0.2

        ranked: List[Tuple[float, Mapping[str, Any]]] = []
        for row in rows:
            ticker = row.get("ticker")
            sd = row.get("supply_demand_score", 50) or 0
            tech_score = 50.0
            rs_score = 50.0
            if ticker in self._tech_table.index:
                trow = self._tech_table.loc[ticker]
                tech_val = trow.get("technical_score", np.nan)
                tech_score = float(tech_val) if not pd.isna(tech_val) else 50.0
                rs20 = trow.get("rs_20d", np.nan)
                rs60 = trow.get("rs_60d", np.nan)
                if not pd.isna(rs20) and not pd.isna(rs60):
                    rs_score = float(self._compute_rs_score(float(rs20), float(rs60)))
            pre_score = sd * w_sd + tech_score * w_tech + rs_score * w_rs
            ranked.append((pre_score, row))

        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = [row for _, row in ranked[:limit]]
        logger.info("Two-step screening enabled: %d -> %d tickers for API phase", len(rows), len(selected))
        return selected

    def run_screening(
        self,
        top_n: int = 50,
        max_tickers: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run enhanced screening"""
        logger.info("🔍 Running Enhanced Smart Money Screening...")
        
        if self.volume_df is None or self.volume_df.empty:
            logger.warning("⚠️ Volume analysis data not loaded")
            return pd.DataFrame()

        # Pre-filter: Focus on accumulation candidates
        filtered = self.volume_df[self.volume_df['supply_demand_score'] >= 50]
        if max_tickers and max_tickers > 0:
            filtered = filtered.sort_values('supply_demand_score', ascending=False).head(max_tickers)
            logger.info("🔧 Limiting screening universe to top %d by supply/demand score", max_tickers)
        
        logger.info(f"📊 Pre-filtered to {len(filtered)} candidates")
        
        results = []
        rows = filtered.to_dict(orient='records')
        rows = self._apply_stage2_preselection(rows)
        worker_count = self._resolve_workers(workers)
        if worker_count > 1:
            logger.info("⚡ Parallel screening enabled: %d workers", worker_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(self._analyze_row, row) for row in rows]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Enhanced Screening"):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for row in tqdm(rows, total=len(rows), desc="Enhanced Screening"):
                result = self._analyze_row(row)
                if result:
                    results.append(result)
        
        # Create DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('composite_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return results_df
    
    def run(self, top_n: int = 50, max_tickers: Optional[int] = None, workers: Optional[int] = None) -> pd.DataFrame:
        """Main execution"""
        logger.info("🚀 Starting Enhanced Smart Money Screener v2.0...")
        
        if not self.load_data():
            logger.error("❌ Failed to load data")
            return pd.DataFrame()
        
        results_df = self.run_screening(top_n, max_tickers=max_tickers, workers=workers)
        self._log_api_stats(total_candidates=len(results_df))
        
        # Save results
        results_df.to_csv(self.output_file, index=False)
        logger.info(f"✅ Saved to {self.output_file}")
        self._save_to_db(results_df)
        
        # Print summary
        logger.info("\n📊 Grade Distribution:")
        for grade in results_df['grade'].unique():
            count = len(results_df[results_df['grade'] == grade])
            logger.info(f"   {grade}: {count} stocks")
        
        return results_df

    def _save_to_db(self, results_df: pd.DataFrame) -> int:
        if results_df.empty:
            return 0
        conn = get_db_connection()
        if conn is None:
            return 0
        inserted = 0
        now = datetime.utcnow().isoformat()
        analysis_date = now[:10]
        run_id = "sm_" + "".join(ch for ch in now if ch.isalnum())
        summary = {
            "total_analyzed": len(results_df),
            "avg_score": round(float(results_df["composite_score"].mean()), 2),
        }
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO market_smart_money_runs
                (run_id, analysis_date, analysis_timestamp, summary_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, analysis_date, now, json.dumps(summary, ensure_ascii=False), now),
            )

            for _, row in results_df.iterrows():
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
                except Exception as exc:
                    logger.debug("DB insert failed: %s", exc)
            conn.commit()
            logger.info("🗄️ SQLite: Inserted %d rows into market_smart_money_picks", inserted)
        except Exception as exc:
            logger.warning("⚠️ SQLite save failed (CSV still saved): %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return inserted


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Smart Money Screener')
    parser.add_argument('--dir', default=os.getenv("DATA_DIR", "."), help='Data directory')
    parser.add_argument('--top', type=int, default=20, help='Top N to display')
    parser.add_argument('--limit', type=int, default=None, help='Limit screening universe for faster runs')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers (default: env)')
    args = parser.parse_args()

    max_tickers = args.limit
    if max_tickers is None:
        env_limit = os.getenv("SMART_MONEY_LIMIT")
        if env_limit:
            try:
                max_tickers = int(env_limit)
            except ValueError:
                logger.warning("SMART_MONEY_LIMIT must be an integer; ignoring")
                max_tickers = None
    if max_tickers is not None and max_tickers <= 0:
        max_tickers = None

    screener = EnhancedSmartMoneyScreener(data_dir=args.dir)
    results = screener.run(top_n=args.top, max_tickers=max_tickers, workers=args.workers)
    
    if not results.empty:
        print(f"\n🔥 TOP {args.top} ENHANCED SMART MONEY PICKS")
        print("=" * 80)
        display_cols = ['rank', 'ticker', 'name', 'grade', 'composite_score', 'current_price']
        print(results[display_cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
