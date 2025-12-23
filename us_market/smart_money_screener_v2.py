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
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    sys.path.append(str(ROOT_DIR))

from utils.fmp_client import FMPClient
from utils.symbols import to_fmp_symbol

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSmartMoneyScreener:
    """
    Enhanced screener with comprehensive analysis:
    1. Supply/Demand (volume analysis)
    2. Technical Analysis (RSI, MACD, MA)
    3. Fundamentals (valuation, growth)
    4. Analyst Ratings
    5. Relative Strength
    """
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'smart_money_picks_v2.csv')
        
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
        
        # S&P 500 benchmark data
        self.spy_data = None
        
    def load_data(self) -> bool:
        """Load all analysis results"""
        try:
            # Volume Analysis
            vol_file = os.path.join(self.data_dir, 'us_volume_analysis.csv')
            if os.path.exists(vol_file):
                self.volume_df = pd.read_csv(vol_file)
                logger.info(f"✅ Loaded volume analysis: {len(self.volume_df)} stocks")
            else:
                logger.warning("⚠️ Volume analysis not found")
                return False
            
            # ETF Flows
            etf_file = os.path.join(self.data_dir, 'us_etf_flows.csv')
            if os.path.exists(etf_file):
                self.etf_df = pd.read_csv(etf_file)
            
            # Load SPY for relative strength
            logger.info("📈 Loading SPY benchmark data...")
            self.spy_data = self._get_history("SPY", days=90)
            
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

    def _get_profile(self, ticker: str) -> Dict:
        return self._get_cached(
            self.profile_cache,
            ticker,
            lambda: self._get_client().profile(to_fmp_symbol(ticker)),
        )

    def _get_quote(self, ticker: str) -> Dict:
        def loader():
            quotes = self._get_client().quote([to_fmp_symbol(ticker)])
            return quotes[0] if quotes else {}

        return self._get_cached(self.quote_cache, ticker, loader)

    def _get_key_metrics(self, ticker: str) -> Dict:
        return self._get_cached(
            self.metrics_cache,
            ticker,
            lambda: self._get_client().key_metrics_ttm(to_fmp_symbol(ticker)),
        )

    def _get_ratios(self, ticker: str) -> Dict:
        return self._get_cached(
            self.ratios_cache,
            ticker,
            lambda: self._get_client().ratios_ttm(to_fmp_symbol(ticker)),
        )

    def _get_history(self, ticker: str, days: int = 180) -> pd.DataFrame:
        cache_key = (ticker, days)

        def loader():
            end_date = datetime.utcnow().date()
            from_date = (end_date - timedelta(days=days)).isoformat()
            to_date = end_date.isoformat()
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
            hist = self._get_history(ticker, days=180)
            
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
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50
            current_price = close.iloc[-1]
            
            # MA Arrangement
            if current_price > ma20 > ma50:
                ma_signal = "Bullish"
            elif current_price < ma20 < ma50:
                ma_signal = "Bearish"
            else:
                ma_signal = "Neutral"
            
            # Golden/Death Cross
            ma50_prev = close.rolling(50).mean().iloc[-5]
            ma200_prev = close.rolling(200).mean().iloc[-5] if len(close) >= 200 else ma50_prev
            
            if ma50 > ma200 and ma50_prev <= ma200_prev:
                cross_signal = "Golden Cross"
            elif ma50 < ma200 and ma50_prev >= ma200_prev:
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
            
            # Valuation
            pe_ratio = metrics.get('peRatioTTM') or ratios.get('priceEarningsRatioTTM') or 0
            forward_pe = profile.get('pe') or ratios.get('priceEarningsRatioTTM') or 0
            pb_ratio = ratios.get('priceToBookRatioTTM') or metrics.get('pbRatioTTM') or 0
            
            # Growth (fallback to 0 if not available)
            revenue_growth = profile.get('revenueGrowth') or 0
            earnings_growth = profile.get('earningsGrowth') or 0
            
            # Profitability
            profit_margin = ratios.get('netProfitMarginTTM') or metrics.get('netProfitMarginTTM') or 0
            roe = metrics.get('roeTTM') or ratios.get('returnOnEquityTTM') or 0
            
            # Market Cap
            market_cap = profile.get('mktCap') or profile.get('marketCap') or 0
            
            # Dividend
            dividend_yield = ratios.get('dividendYieldTTM') or 0
            
            # Fundamental Score (0-100)
            fund_score = 50
            
            # P/E contribution (lower is better, but not too low)
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
                'pe_ratio': round(pe_ratio, 2) if pe_ratio else 'N/A',
                'forward_pe': round(forward_pe, 2) if forward_pe else 'N/A',
                'pb_ratio': round(pb_ratio, 2) if pb_ratio else 'N/A',
                'revenue_growth': round(revenue_growth * 100, 1) if revenue_growth else 0,
                'earnings_growth': round(earnings_growth * 100, 1) if earnings_growth else 0,
                'profit_margin': round(profit_margin * 100, 1) if profit_margin else 0,
                'roe': round(roe * 100, 1) if roe else 0,
                'market_cap_b': round(market_cap / 1e9, 1),
                'size': size,
                'dividend_yield': round(dividend_yield * 100, 2) if dividend_yield else 0,
                'fundamental_score': fund_score
            }
            
        except Exception as e:
            return self._default_fundamental()
    
    def _default_fundamental(self) -> Dict:
        return {
            'pe_ratio': 'N/A', 'forward_pe': 'N/A', 'pb_ratio': 'N/A',
            'revenue_growth': 0, 'earnings_growth': 0, 'profit_margin': 0,
            'roe': 0, 'market_cap_b': 0, 'size': 'Unknown', 'dividend_yield': 0,
            'fundamental_score': 50
        }
    
    def get_analyst_ratings(self, ticker: str) -> Dict:
        """Get analyst consensus and target price"""
        try:
            profile = self._get_profile(ticker)
            quote = self._get_quote(ticker)
            client = self._get_client()
            ratings = client.ratings_snapshot(to_fmp_symbol(ticker))
            consensus = client.price_target_consensus(to_fmp_symbol(ticker))
            
            company_name = profile.get('companyName') or profile.get('name') or ticker
            
            current_price = quote.get('price') or profile.get('price') or 0
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
            if self.spy_data is None or len(self.spy_data) < 20:
                return {'rs_20d': 0, 'rs_60d': 0, 'rs_score': 50}
            
            hist = self._get_history(ticker, days=90)
            
            if len(hist) < 20:
                return {'rs_20d': 0, 'rs_60d': 0, 'rs_score': 50}
            
            # Calculate returns
            stock_return_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
            stock_return_60d = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
            
            spy_return_20d = (self.spy_data['Close'].iloc[-1] / self.spy_data['Close'].iloc[-21] - 1) * 100 if len(self.spy_data) >= 21 else 0
            spy_return_60d = (self.spy_data['Close'].iloc[-1] / self.spy_data['Close'].iloc[0] - 1) * 100
            
            rs_20d = stock_return_20d - spy_return_20d
            rs_60d = stock_return_60d - spy_return_60d
            
            # RS Score (0-100)
            rs_score = 50
            if rs_20d > 10: rs_score += 25
            elif rs_20d > 5: rs_score += 15
            elif rs_20d > 0: rs_score += 8
            elif rs_20d < -10: rs_score -= 20
            elif rs_20d < -5: rs_score -= 10
            
            if rs_60d > 15: rs_score += 15
            elif rs_60d > 5: rs_score += 8
            elif rs_60d < -15: rs_score -= 15
            
            rs_score = max(0, min(100, rs_score))
            
            return {
                'rs_20d': round(rs_20d, 1),
                'rs_60d': round(rs_60d, 1),
                'rs_score': rs_score
            }
            
        except Exception as e:
            return {'rs_20d': 0, 'rs_60d': 0, 'rs_score': 50}
    
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
            return max(1, min(8, int(workers)))
        env_workers = os.getenv("SMART_MONEY_WORKERS")
        if env_workers:
            try:
                return max(1, min(8, int(env_workers)))
            except ValueError:
                logger.warning("SMART_MONEY_WORKERS must be an integer; using single-thread")
                return 1
        enable_threads = str(os.getenv("PERF_ENABLE_THREADS", "0")).lower() in ("1", "true", "yes")
        if enable_threads:
            try:
                return max(1, min(8, int(os.getenv("PERF_MAX_WORKERS", "4"))))
            except ValueError:
                return 4
        return 1

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
        
        # Save results
        results_df.to_csv(self.output_file, index=False)
        logger.info(f"✅ Saved to {self.output_file}")
        
        # Print summary
        logger.info("\n📊 Grade Distribution:")
        for grade in results_df['grade'].unique():
            count = len(results_df[results_df['grade'] == grade])
            logger.info(f"   {grade}: {count} stocks")
        
        return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Smart Money Screener')
    parser.add_argument('--dir', default='.', help='Data directory')
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
