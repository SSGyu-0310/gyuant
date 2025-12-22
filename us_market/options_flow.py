#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Flow Analyzer
Analyzes options trading volume to detect institutional positioning
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptionsFlowAnalyzer:
    """Analyze options flow to detect large trader positioning"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'options_flow.json')
        
        # Default watchlist - major stocks with high options activity
        self.watchlist = [
            'AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'META', 'GOOGL', 
            'SPY', 'QQQ', 'AMD', 'NFLX', 'COIN', 'BABA', 'NIO', 'PLTR'
        ]
    
    def get_options_summary(self, ticker: str) -> Optional[Dict]:
        """Get options summary for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if not expirations:
                return None
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            opt = stock.option_chain(nearest_exp)
            
            calls = opt.calls
            puts = opt.puts
            
            # Volume metrics
            call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            
            # Open Interest
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
            
            # Put/Call Ratio
            pc_ratio = put_volume / call_volume if call_volume > 0 else 0
            
            # Implied Volatility (average of ATM options)
            avg_call_iv = calls['impliedVolatility'].mean() if 'impliedVolatility' in calls.columns else 0
            avg_put_iv = puts['impliedVolatility'].mean() if 'impliedVolatility' in puts.columns else 0
            avg_iv = (avg_call_iv + avg_put_iv) / 2 * 100
            
            # Unusual Activity Detection
            avg_call_vol = calls['volume'].mean() if not calls.empty else 0
            avg_put_vol = puts['volume'].mean() if not puts.empty else 0
            
            unusual_calls = calls[calls['volume'] > avg_call_vol * 3] if avg_call_vol > 0 else pd.DataFrame()
            unusual_puts = puts[puts['volume'] > avg_put_vol * 3] if avg_put_vol > 0 else pd.DataFrame()
            
            # Sentiment interpretation
            if pc_ratio < 0.7:
                sentiment = "Bullish"
            elif pc_ratio > 1.3:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
            
            # Score (0-100)
            score = 50
            if pc_ratio < 0.5:
                score += 25
            elif pc_ratio < 0.7:
                score += 15
            elif pc_ratio > 1.5:
                score -= 20
            elif pc_ratio > 1.2:
                score -= 10
            
            # High unusual call activity is bullish
            if len(unusual_calls) > 3:
                score += 10
            if len(unusual_puts) > 3:
                score -= 10
            
            score = max(0, min(100, score))
            
            return {
                'ticker': ticker,
                'expiration': nearest_exp,
                'metrics': {
                    'pc_ratio': round(pc_ratio, 2),
                    'call_volume': int(call_volume),
                    'put_volume': int(put_volume),
                    'call_oi': int(call_oi),
                    'put_oi': int(put_oi),
                    'implied_volatility': round(avg_iv, 1)
                },
                'unusual_activity': {
                    'unusual_calls': len(unusual_calls),
                    'unusual_puts': len(unusual_puts)
                },
                'sentiment': sentiment,
                'score': score
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return None
    
    def analyze_watchlist(self, tickers: List[str] = None) -> Dict:
        """Analyze options flow for watchlist"""
        logger.info("🔍 Analyzing Options Flow...")
        
        tickers = tickers or self.watchlist
        results = []
        
        for ticker in tickers:
            logger.info(f"   Processing {ticker}...")
            summary = self.get_options_summary(ticker)
            if summary:
                results.append(summary)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Summary statistics
        bullish = [r for r in results if r['sentiment'] == 'Bullish']
        bearish = [r for r in results if r['sentiment'] == 'Bearish']
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_analyzed': len(results),
                'bullish_count': len(bullish),
                'bearish_count': len(bearish),
                'neutral_count': len(results) - len(bullish) - len(bearish)
            },
            'options_flow': results
        }
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"✅ Saved to {self.output_file}")
        
        return output
    
    def print_summary(self, data: Dict):
        """Print analysis summary"""
        print("\n📊 Options Flow Summary:")
        print(f"   Total analyzed: {data['summary']['total_analyzed']}")
        print(f"   Bullish: {data['summary']['bullish_count']}")
        print(f"   Bearish: {data['summary']['bearish_count']}")
        
        print("\n🔥 Top Bullish (Low P/C Ratio):")
        for item in data['options_flow'][:5]:
            print(f"   {item['ticker']}: P/C={item['metrics']['pc_ratio']:.2f} | "
                  f"Sentiment: {item['sentiment']} | Score: {item['score']}")


def main():
    import argparse
    import pandas as pd  # Import here for unusual activity detection
    
    parser = argparse.ArgumentParser(description='Options Flow Analyzer')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    args = parser.parse_args()
    
    analyzer = OptionsFlowAnalyzer(data_dir=args.dir)
    
    if args.tickers:
        data = analyzer.analyze_watchlist(args.tickers)
    else:
        data = analyzer.analyze_watchlist()
    
    analyzer.print_summary(data)


if __name__ == "__main__":
    import pandas as pd  # Required for DataFrame operations
    main()
