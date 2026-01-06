#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insider Trading Tracker
Tracks insider buying/selling activity from SEC filings
"""

import os
import sys
from pathlib import Path
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.fmp_client import get_fmp_client
from utils.symbols import to_fmp_symbol
from utils.db_writer import write_market_documents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InsiderTracker:
    """Track insider trading activity"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'insider_moves.json')
        self.fmp = get_fmp_client()
        
    def get_insider_activity(self, ticker: str) -> List[Dict]:
        """Get insider transactions for a ticker"""
        try:
            records = self.fmp.insider_trading(to_fmp_symbol(ticker), limit=100)
            if not records:
                return []

            cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
            recent_transactions = []

            for row in records:
                try:
                    date_str = row.get('transactionDate') or row.get('filingDate')
                    if not date_str:
                        continue
                    trans_date = pd.to_datetime(date_str)
                    if trans_date < cutoff:
                        continue

                    transaction = str(row.get('transactionType', '')).lower()
                    if 'purchase' in transaction or 'buy' in transaction:
                        trans_type = 'Buy'
                    elif 'sale' in transaction or 'sell' in transaction:
                        trans_type = 'Sell'
                    else:
                        trans_type = 'Other'

                    value = float(row.get('value') or 0)
                    shares = int(row.get('securitiesTransacted') or row.get('shares') or 0)
                    price = float(row.get('price') or 0)
                    if value == 0 and shares and price:
                        value = shares * price
                    insider = row.get('insiderName') or row.get('reportingName') or 'N/A'

                    if trans_type == 'Other':
                        continue

                    recent_transactions.append({
                        'date': str(trans_date.date()) if hasattr(trans_date, 'date') else str(trans_date)[:10],
                        'insider': insider,
                        'type': trans_type,
                        'value': value,
                        'shares': shares
                    })

                except Exception:
                    continue

            return recent_transactions[:10]
            
        except Exception as e:
            logger.debug(f"Error getting insider data for {ticker}: {e}")
            return []
    
    def calculate_insider_score(self, transactions: List[Dict]) -> Dict:
        """Calculate insider sentiment score"""
        if not transactions:
            return {'score': 50, 'sentiment': 'Unknown', 'buy_value': 0, 'sell_value': 0}
        
        buy_value = sum(t['value'] for t in transactions if t['type'] == 'Buy')
        sell_value = sum(t['value'] for t in transactions if t['type'] == 'Sell')
        
        buy_count = len([t for t in transactions if t['type'] == 'Buy'])
        sell_count = len([t for t in transactions if t['type'] == 'Sell'])
        
        # Score calculation
        score = 50
        
        # Value-based scoring
        if buy_value > sell_value * 2:
            score += 25
        elif buy_value > sell_value:
            score += 15
        elif sell_value > buy_value * 2:
            score -= 20
        elif sell_value > buy_value:
            score -= 10
        
        # Count-based adjustment
        if buy_count > sell_count:
            score += 10
        elif sell_count > buy_count:
            score -= 5
        
        # Large buy bonus
        large_buys = [t for t in transactions if t['type'] == 'Buy' and t['value'] > 100000]
        score += len(large_buys) * 5
        
        score = max(0, min(100, score))
        
        # Sentiment
        if score >= 70:
            sentiment = "Strong Buying"
        elif score >= 55:
            sentiment = "Buying"
        elif score >= 45:
            sentiment = "Neutral"
        elif score >= 30:
            sentiment = "Selling"
        else:
            sentiment = "Strong Selling"
        
        return {
            'score': score,
            'sentiment': sentiment,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'buy_count': buy_count,
            'sell_count': sell_count
        }
    
    def analyze_tickers(self, tickers: List[str]) -> Dict:
        """Analyze insider activity for multiple tickers"""
        logger.info(f"🔍 Analyzing insider activity for {len(tickers)} stocks...")
        
        results = {}
        significant_buys = []
        
        for ticker in tickers:
            transactions = self.get_insider_activity(ticker)
            
            if transactions:
                analysis = self.calculate_insider_score(transactions)
                
                results[ticker] = {
                    'score': analysis['score'],
                    'sentiment': analysis['sentiment'],
                    'stats': {
                        'buy_value': analysis['buy_value'],
                        'sell_value': analysis['sell_value'],
                        'buy_count': analysis['buy_count'],
                        'sell_count': analysis['sell_count']
                    },
                    'transactions': transactions[:5]  # Top 5 recent
                }
                
                # Track significant buying
                if analysis['sentiment'] in ['Strong Buying', 'Buying']:
                    significant_buys.append({
                        'ticker': ticker,
                        'score': analysis['score'],
                        'buy_value': analysis['buy_value']
                    })
        
        # Sort significant buys
        significant_buys.sort(key=lambda x: x['score'], reverse=True)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_analyzed': len(tickers),
                'with_activity': len(results),
                'significant_buying': len(significant_buys)
            },
            'significant_buys': significant_buys[:10],
            'details': results
        }
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"✅ Saved to {self.output_file}")
        write_market_documents(
            "insider_moves",
            output,
            as_of_date=output.get("timestamp", "")[:10],
        )
        
        return output
    
    def print_summary(self, data: Dict):
        """Print analysis summary"""
        print("\n📊 Insider Activity Summary:")
        print(f"   Total analyzed: {data['summary']['total_analyzed']}")
        print(f"   With activity: {data['summary']['with_activity']}")
        print(f"   Significant buying: {data['summary']['significant_buying']}")
        
        if data['significant_buys']:
            print("\n🔥 Top Insider Buying:")
            for item in data['significant_buys'][:5]:
                value_m = item['buy_value'] / 1e6
                print(f"   {item['ticker']}: Score {item['score']} | Buy Value: ${value_m:.1f}M")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Insider Trading Tracker')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    args = parser.parse_args()
    
    tracker = InsiderTracker(data_dir=args.dir)
    
    # Default tickers if none specified
    default_tickers = [
        'AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'JPM', 'V', 'JNJ',
        'WMT', 'PG', 'XOM', 'UNH', 'HD', 'BAC', 'DIS', 'NFLX', 'CRM', 'AMD'
    ]
    
    tickers = args.tickers if args.tickers else default_tickers
    data = tracker.analyze_tickers(tickers)
    tracker.print_summary(data)


if __name__ == "__main__":
    main()
