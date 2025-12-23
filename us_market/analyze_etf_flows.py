#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US ETF Flows Analysis
Analyzes fund flows for major ETFs using volume and price data
Optionally generates AI-powered insights using Gemini
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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


class ETFFlowAnalyzer:
    """Analyze ETF fund flows using volume and price indicators"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_csv = os.path.join(data_dir, 'us_etf_flows.csv')
        self.output_json = os.path.join(data_dir, 'etf_flow_analysis.json')
        self.client = get_fmp_client()
        
        # Major ETFs to track (24 ETFs covering different sectors/asset classes)
        self.etf_list = {
            # Broad Market
            'SPY': {'name': 'SPDR S&P 500 ETF', 'category': 'Large Cap'},
            'QQQ': {'name': 'Invesco QQQ (NASDAQ-100)', 'category': 'Tech/Growth'},
            'IWM': {'name': 'iShares Russell 2000', 'category': 'Small Cap'},
            'DIA': {'name': 'SPDR Dow Jones', 'category': 'Blue Chip'},
            'VTI': {'name': 'Vanguard Total Stock Market', 'category': 'Total Market'},
            
            # Sector ETFs
            'XLK': {'name': 'Technology Select Sector', 'category': 'Technology'},
            'XLF': {'name': 'Financial Select Sector', 'category': 'Financials'},
            'XLV': {'name': 'Health Care Select Sector', 'category': 'Healthcare'},
            'XLE': {'name': 'Energy Select Sector', 'category': 'Energy'},
            'XLI': {'name': 'Industrial Select Sector', 'category': 'Industrials'},
            'XLY': {'name': 'Consumer Discretionary', 'category': 'Consumer'},
            'XLP': {'name': 'Consumer Staples', 'category': 'Consumer Staples'},
            'XLU': {'name': 'Utilities Select Sector', 'category': 'Utilities'},
            'XLRE': {'name': 'Real Estate Select Sector', 'category': 'Real Estate'},
            
            # Themed/Factor ETFs
            'ARKK': {'name': 'ARK Innovation ETF', 'category': 'Innovation'},
            'SOXX': {'name': 'iShares Semiconductor', 'category': 'Semiconductors'},
            'IBB': {'name': 'iShares Biotechnology', 'category': 'Biotech'},
            
            # Commodities
            'GLD': {'name': 'SPDR Gold Trust', 'category': 'Gold'},
            'SLV': {'name': 'iShares Silver Trust', 'category': 'Silver'},
            'USO': {'name': 'United States Oil Fund', 'category': 'Oil'},
            
            # Bonds/Fixed Income
            'TLT': {'name': 'iShares 20+ Year Treasury', 'category': 'Long-Term Bonds'},
            'HYG': {'name': 'iShares High Yield Corporate', 'category': 'High Yield Bonds'},
            
            # International
            'EEM': {'name': 'iShares Emerging Markets', 'category': 'Emerging Markets'},
            'EFA': {'name': 'iShares MSCI EAFE', 'category': 'Developed Markets'},
        }
        
        # Date range
        self.lookback_days = 90
    
    def download_etf_data(self, ticker: str) -> pd.DataFrame:
        """Download ETF historical data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            payload = self.client.historical_price_full(
                ticker,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
            )
            hist = pd.DataFrame(payload.get("historical") or [])
            if hist.empty:
                return pd.DataFrame()

            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist = hist.dropna(subset=["date"]).sort_values("date")
            hist['ticker'] = ticker
            hist = hist.rename(columns={"date": "Date", "close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
            return hist
            
        except Exception as e:
            logger.debug(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_flow_proxy(self, df: pd.DataFrame) -> Dict:
        """
        Calculate flow proxy indicators
        Since we don't have actual AUM data, we use volume and price patterns
        """
        if len(df) < 20:
            return None
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # Volume trends
        vol_5d = df['Volume'].tail(5).mean()
        vol_20d = df['Volume'].tail(20).mean()
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1
        
        # Price momentum
        price_1w = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        price_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
        
        # OBV trend (20-day)
        obv_change = (obv[-1] - obv[-20]) / abs(obv[-20]) * 100 if obv[-20] != 0 else 0
        
        # Flow Score (0-100)
        score = 50
        
        # Volume ratio contribution
        if vol_ratio > 1.5:
            score += 15
        elif vol_ratio > 1.2:
            score += 10
        elif vol_ratio < 0.8:
            score -= 10
        
        # Price momentum contribution
        if price_1w > 3:
            score += 10
        elif price_1w > 1:
            score += 5
        elif price_1w < -3:
            score -= 10
        elif price_1w < -1:
            score -= 5
        
        # OBV trend contribution
        if obv_change > 10:
            score += 15
        elif obv_change > 5:
            score += 10
        elif obv_change < -10:
            score -= 15
        elif obv_change < -5:
            score -= 10
        
        score = max(0, min(100, score))
        
        # Determine flow status
        if score >= 70:
            flow_status = "Strong Inflow"
        elif score >= 55:
            flow_status = "Inflow"
        elif score >= 45:
            flow_status = "Neutral"
        elif score >= 30:
            flow_status = "Outflow"
        else:
            flow_status = "Strong Outflow"
        
        return {
            'current_price': round(df['Close'].iloc[-1], 2),
            'price_1w_pct': round(price_1w, 2),
            'price_1m_pct': round(price_1m, 2),
            'vol_ratio_5d_20d': round(vol_ratio, 2),
            'obv_change_20d_pct': round(obv_change, 2),
            'avg_volume_20d': int(vol_20d),
            'flow_score': round(score, 1),
            'flow_status': flow_status
        }
    
    def generate_ai_analysis(self, results_df: pd.DataFrame) -> Optional[Dict]:
        """Generate AI analysis using Gemini (optional)"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.info("⚠️ GEMINI_API_KEY not set, skipping AI analysis")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Prepare data summary
            inflows = results_df[results_df['flow_status'].str.contains('Inflow')]
            outflows = results_df[results_df['flow_status'].str.contains('Outflow')]
            
            prompt = f"""Analyze the following ETF capital flow data and provide insights:

## Strong Inflows (top 5):
{inflows.nlargest(5, 'flow_score')[['ticker', 'name', 'category', 'flow_score', 'price_1w_pct']].to_markdown()}

## Strong Outflows (top 5):
{outflows.nsmallest(5, 'flow_score')[['ticker', 'name', 'category', 'flow_score', 'price_1w_pct']].to_markdown()}

Please provide:
1. Market Sentiment Summary (1-2 sentences)
2. Key Sector Rotations (which sectors are seeing inflows vs outflows)
3. Risk Assessment (Low/Medium/High and why)
4. Investment Implications (2-3 bullet points)

Keep the response concise and actionable.
"""
            
            response = model.generate_content(prompt)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'analysis': response.text,
                'data_summary': {
                    'total_etfs': len(results_df),
                    'inflows': len(inflows),
                    'outflows': len(outflows),
                    'neutral': len(results_df) - len(inflows) - len(outflows)
                }
            }
            
        except ImportError:
            logger.info("⚠️ google-generativeai not installed, skipping AI analysis")
            return None
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return None
    
    def run(self, skip_ai: bool = False) -> pd.DataFrame:
        """Run ETF flow analysis"""
        logger.info("🚀 Starting ETF Flow Analysis...")
        
        results = []
        
        for ticker, info in tqdm(self.etf_list.items(), desc="Analyzing ETFs"):
            df = self.download_etf_data(ticker)
            
            if df.empty:
                logger.debug(f"No data for {ticker}")
                continue
            
            analysis = self.calculate_flow_proxy(df)
            
            if analysis:
                result = {
                    'ticker': ticker,
                    'name': info['name'],
                    'category': info['category'],
                    **analysis
                }
                results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            logger.warning("No ETF data collected")
            return results_df
        
        # Sort by flow score
        results_df = results_df.sort_values('flow_score', ascending=False).reset_index(drop=True)
        
        # Save CSV
        results_df.to_csv(self.output_csv, index=False)
        logger.info(f"✅ Saved ETF analysis to {self.output_csv}")
        
        # Prepare data summary
        inflows = results_df[results_df['flow_status'].str.contains('Inflow')]
        outflows = results_df[results_df['flow_status'].str.contains('Outflow')]
        
        data_summary = {
            'total_etfs': len(results_df),
            'inflows': len(inflows),
            'outflows': len(outflows),
            'neutral': len(results_df) - len(inflows) - len(outflows),
            'top_inflow': results_df.nlargest(3, 'flow_score')[['ticker', 'flow_score', 'flow_status']].to_dict('records') if len(results_df) > 0 else [],
            'top_outflow': results_df.nsmallest(3, 'flow_score')[['ticker', 'flow_score', 'flow_status']].to_dict('records') if len(results_df) > 0 else [],
        }
        
        # Initialize JSON output with basic data
        json_output = {
            'timestamp': datetime.now().isoformat(),
            'ai_analysis': '',
            'data_summary': data_summary
        }
        
        # AI Analysis (optional)
        if not skip_ai:
            ai_analysis = self.generate_ai_analysis(results_df)
            if ai_analysis:
                json_output['ai_analysis'] = ai_analysis.get('analysis', '')
        
        # Always save JSON file
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved analysis JSON to {self.output_json}")
        
        # Print summary
        logger.info("\n📊 ETF Flow Summary:")
        logger.info(f"   Total ETFs analyzed: {len(results_df)}")
        
        by_status = results_df['flow_status'].value_counts()
        for status, count in by_status.items():
            logger.info(f"   {status}: {count}")
        
        return results_df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ETF Flow Analysis')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--skip-ai', action='store_true', help='Skip AI analysis')
    args = parser.parse_args()
    
    analyzer = ETFFlowAnalyzer(data_dir=args.dir)
    results = analyzer.run(skip_ai=args.skip_ai)
    
    if not results.empty:
        # Show top inflows
        print("\n📈 Top 5 ETF Inflows:")
        top_inflows = results.nlargest(5, 'flow_score')
        for _, row in top_inflows.iterrows():
            print(f"   {row['ticker']} ({row['category']}): Score {row['flow_score']} | "
                  f"1W: {row['price_1w_pct']:+.1f}%")
        
        # Show top outflows
        print("\n📉 Top 5 ETF Outflows:")
        top_outflows = results.nsmallest(5, 'flow_score')
        for _, row in top_outflows.iterrows():
            print(f"   {row['ticker']} ({row['category']}): Score {row['flow_score']} | "
                  f"1W: {row['price_1w_pct']:+.1f}%")
        
        # Sector summary
        print("\n🏭 Category Summary:")
        category_avg = results.groupby('category')['flow_score'].mean().sort_values(ascending=False)
        for cat, score in category_avg.head(5).items():
            print(f"   {cat}: Avg Score {score:.1f}")


if __name__ == "__main__":
    main()
