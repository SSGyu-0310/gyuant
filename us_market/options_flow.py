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
            logger.info("Options chain data not available via FMP; skipping.")
            return None
            
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
            'options_flow': results,
            'note': 'Options chain data not available via FMP. Configure a dedicated options provider to enable this feature.'
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
    main()
