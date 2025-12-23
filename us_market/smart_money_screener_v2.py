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
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from utils.fmp_client import get_fmp_client

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

        self.client = get_fmp_client()
        
        # Load analysis data
        self.volume_df = None
        self.holdings_df = None
        self.etf_df = None
        self.prices_df = None
        
        # Cache for FMP data
        self.fmp_cache = {
            "history": {},
            "profile": {},
            "quote": {},
            "metrics": {},
            "ratios": {},
            "ratings": {},
            "targets": {},
        }
        
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
            
            # 13F Holdings
            holdings_file = os.path.join(self.data_dir, 'us_13f_holdings.csv')
            if os.path.exists(holdings_file):
                self.holdings_df = pd.read_csv(holdings_file)
                logger.info(f"✅ Loaded 13F holdings: {len(self.holdings_df)} stocks")
            else:
                logger.warning("⚠️ 13F holdings not found")
                return False
            
            # ETF Flows
            etf_file = os.path.join(self.data_dir, 'us_etf_flows.csv')
            if os.path.exists(etf_file):
                self.etf_df = pd.read_csv(etf_file)
            
            # Load SPY for relative strength
            logger.info("📈 Loading SPY benchmark data...")
            self.spy_data = self._get_history("SPY", days=120)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False

    def _get_history(self, ticker: str, days: int = 180) -> pd.DataFrame:
        cache_key = (ticker, days)
        cached = self.fmp_cache["history"].get(cache_key)
        if cached is not None:
            return cached
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        payload = self.client.historical_price_full(
            ticker,
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
        )
        hist = pd.DataFrame(payload.get("historical") or [])
        if not hist.empty:
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist = hist.dropna(subset=["date"]).sort_values("date")
        self.fmp_cache["history"][cache_key] = hist
        return hist

    def _get_profile(self, ticker: str) -> Dict:
        cached = self.fmp_cache["profile"].get(ticker)
        if cached is not None:
            return cached
        profiles = self.client.profile(ticker)
        profile = profiles[0] if profiles else {}
        self.fmp_cache["profile"][ticker] = profile
        return profile

    def _get_quote(self, ticker: str) -> Dict:
        cached = self.fmp_cache["quote"].get(ticker)
        if cached is not None:
            return cached
        quotes = self.client.quote([ticker])
        quote = quotes[0] if quotes else {}
        self.fmp_cache["quote"][ticker] = quote
        return quote

    def _get_metrics(self, ticker: str) -> Dict:
        cached = self.fmp_cache["metrics"].get(ticker)
        if cached is not None:
            return cached
        metrics = self.client.key_metrics_ttm(ticker)
        item = metrics[0] if metrics else {}
        self.fmp_cache["metrics"][ticker] = item
        return item

    def _get_ratios(self, ticker: str) -> Dict:
        cached = self.fmp_cache["ratios"].get(ticker)
        if cached is not None:
            return cached
        ratios = self.client.ratios_ttm(ticker)
        item = ratios[0] if ratios else {}
        self.fmp_cache["ratios"][ticker] = item
        return item

    def _get_ratings(self, ticker: str) -> Dict:
        cached = self.fmp_cache["ratings"].get(ticker)
        if cached is not None:
            return cached
        ratings = self.client.ratings_snapshot(ticker)
        item = ratings[0] if ratings else {}
        self.fmp_cache["ratings"][ticker] = item
        return item

    def _get_targets(self, ticker: str) -> Dict:
        cached = self.fmp_cache["targets"].get(ticker)
        if cached is not None:
            return cached
        targets = self.client.price_target_consensus(ticker)
        item = targets[0] if targets else {}
        self.fmp_cache["targets"][ticker] = item
        return item
    
    def get_technical_analysis(self, ticker: str) -> Dict:
        """Calculate technical indicators"""
        try:
            hist = self._get_history(ticker, days=200)
            if hist.empty or len(hist) < 50:
                return self._default_technical()

            close = hist['close']
            
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
            metrics = self._get_metrics(ticker)
            ratios = self._get_ratios(ticker)

            # Valuation
            pe_ratio = ratios.get('peRatioTTM') or ratios.get('priceEarningsRatioTTM') or profile.get('pe') or 0
            forward_pe = profile.get('forwardPE') or ratios.get('forwardPERatioTTM') or 0
            pb_ratio = ratios.get('pbRatioTTM') or ratios.get('priceToBookRatioTTM') or profile.get('priceToBook') or 0

            # Growth
            revenue_growth = metrics.get('revenueGrowthTTM') or metrics.get('revenueGrowth') or 0
            earnings_growth = metrics.get('netIncomeGrowthTTM') or metrics.get('netIncomeGrowth') or 0

            # Profitability
            profit_margin = ratios.get('netProfitMarginTTM') or ratios.get('profitMarginTTM') or metrics.get('netProfitMarginTTM') or 0
            roe = ratios.get('roeTTM') or ratios.get('returnOnEquityTTM') or 0

            # Market Cap
            market_cap = profile.get('mktCap') or profile.get('marketCap') or 0

            # Dividend
            dividend_yield = profile.get('dividendYield') or ratios.get('dividendYieldTTM') or 0
            
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
            ratings = self._get_ratings(ticker)
            targets = self._get_targets(ticker)

            # Get company name
            company_name = profile.get('companyName', '') or profile.get('name', '') or ticker

            current_price = quote.get('price') or profile.get('price') or 0
            target_price = (
                targets.get('targetMean')
                or targets.get('targetConsensus')
                or targets.get('targetMedian')
                or 0
            )

            # Recommendation
            raw_rec = ratings.get('rating') or ratings.get('ratingRecommendation') or ratings.get('recommendation') or 'none'
            recommendation = str(raw_rec).strip()
            
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
            rec_key = recommendation.lower().replace(" ", "")
            if "strongbuy" in rec_key:
                norm_rec = "strongBuy"
            elif "strongsell" in rec_key:
                norm_rec = "strongSell"
            elif "buy" in rec_key:
                norm_rec = "buy"
            elif "sell" in rec_key:
                norm_rec = "sell"
            elif "hold" in rec_key or "neutral" in rec_key:
                norm_rec = "hold"
            else:
                norm_rec = "hold"
            analyst_score += rec_map.get(norm_rec, 0)
            
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
                'recommendation': norm_rec,
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

            hist = self._get_history(ticker, days=120)
            if hist.empty or len(hist) < 20:
                return {'rs_20d': 0, 'rs_60d': 0, 'rs_score': 50}
            
            # Calculate returns
            stock_return_20d = (hist['close'].iloc[-1] / hist['close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
            stock_return_60d = (hist['close'].iloc[-1] / hist['close'].iloc[0] - 1) * 100

            spy_return_20d = (self.spy_data['close'].iloc[-1] / self.spy_data['close'].iloc[-21] - 1) * 100 if len(self.spy_data) >= 21 else 0
            spy_return_60d = (self.spy_data['close'].iloc[-1] / self.spy_data['close'].iloc[0] - 1) * 100
            
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
    
    def calculate_composite_score(self, row: pd.Series, tech: Dict, fund: Dict, analyst: Dict, rs: Dict) -> Tuple[float, str]:
        """Calculate final composite score"""
        # Weighted composite
        composite = (
            row.get('supply_demand_score', 50) * 0.25 +
            row.get('institutional_score', 50) * 0.20 +
            tech.get('technical_score', 50) * 0.20 +
            fund.get('fundamental_score', 50) * 0.15 +
            analyst.get('analyst_score', 50) * 0.10 +
            rs.get('rs_score', 50) * 0.10
        )
        
        # Determine grade
        if composite >= 80: grade = "🔥 S급 (즉시 매수)"
        elif composite >= 70: grade = "🌟 A급 (적극 매수)"
        elif composite >= 60: grade = "📈 B급 (매수 고려)"
        elif composite >= 50: grade = "📊 C급 (관망)"
        elif composite >= 40: grade = "⚠️ D급 (주의)"
        else: grade = "🚫 F급 (회피)"
        
        return round(composite, 1), grade
    
    def run_screening(self, top_n: int = 50) -> pd.DataFrame:
        """Run enhanced screening"""
        logger.info("🔍 Running Enhanced Smart Money Screening...")
        
        # Merge volume and holdings data
        merged_df = pd.merge(
            self.volume_df,
            self.holdings_df,
            on='ticker',
            how='inner',
            suffixes=('_vol', '_inst')
        )
        
        # Pre-filter: Focus on accumulation candidates
        filtered = merged_df[merged_df['supply_demand_score'] >= 50]
        
        logger.info(f"📊 Pre-filtered to {len(filtered)} candidates")
        
        results = []
        
        for idx, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Enhanced Screening"):
            ticker = row['ticker']
            
            # Get all analyses
            tech = self.get_technical_analysis(ticker)
            fund = self.get_fundamental_analysis(ticker)
            analyst = self.get_analyst_ratings(ticker)
            rs = self.get_relative_strength(ticker)
            
            # Calculate composite score
            composite_score, grade = self.calculate_composite_score(row, tech, fund, analyst, rs)
            
            result = {
                'ticker': ticker,
                'name': analyst.get('company_name', ticker),
                'composite_score': composite_score,
                'grade': grade,
                'sd_score': row.get('supply_demand_score', 50),
                'inst_score': row.get('institutional_score', 50),
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
            results.append(result)
        
        # Create DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('composite_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return results_df
    
    def run(self, top_n: int = 50) -> pd.DataFrame:
        """Main execution"""
        logger.info("🚀 Starting Enhanced Smart Money Screener v2.0...")
        
        if not self.load_data():
            logger.error("❌ Failed to load data")
            return pd.DataFrame()
        
        results_df = self.run_screening(top_n)
        
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
    args = parser.parse_args()
    
    screener = EnhancedSmartMoneyScreener(data_dir=args.dir)
    results = screener.run(top_n=args.top)
    
    if not results.empty:
        print(f"\n🔥 TOP {args.top} ENHANCED SMART MONEY PICKS")
        print("=" * 80)
        display_cols = ['rank', 'ticker', 'name', 'grade', 'composite_score', 'current_price']
        print(results[display_cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
