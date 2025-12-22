#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Performance Heatmap Data Collector
Collects sector ETF performance data for heatmap visualization
"""

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SectorHeatmapCollector:
    """Collect sector ETF performance data for heatmap visualization"""
    
    def __init__(self):
        # Sector ETFs with full names and colors
        self.sector_etfs = {
            'XLK': {'name': 'Technology', 'color': '#4A90A4'},
            'XLF': {'name': 'Financials', 'color': '#6B8E23'},
            'XLV': {'name': 'Healthcare', 'color': '#FF69B4'},
            'XLE': {'name': 'Energy', 'color': '#FF6347'},
            'XLY': {'name': 'Consumer Disc.', 'color': '#FFD700'},
            'XLP': {'name': 'Consumer Staples', 'color': '#98D8C8'},
            'XLI': {'name': 'Industrials', 'color': '#DDA0DD'},
            'XLB': {'name': 'Materials', 'color': '#F0E68C'},
            'XLU': {'name': 'Utilities', 'color': '#87CEEB'},
            'XLRE': {'name': 'Real Estate', 'color': '#CD853F'},
            'XLC': {'name': 'Comm. Services', 'color': '#9370DB'},
        }
        
        # Sector stocks for detail map
        self.sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE', 'CSCO', 'INTC'],
            'Financials': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP'],
            'Healthcare': ['UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
            'Consumer Disc.': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'CMG'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'KHC'],
            'Industrials': ['CAT', 'UPS', 'RTX', 'HON', 'DE', 'GE', 'BA', 'LMT', 'UNP', 'MMM'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NUE', 'ECL', 'DD', 'NEM', 'DOW', 'PPG'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC'],
            'Real Estate': ['PLD', 'AMT', 'EQIX', 'PSA', 'CCI', 'O', 'SPG', 'WELL', 'DLR', 'AVB'],
            'Comm. Services': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'ATVI'],
        }
    
    def get_sector_performance(self, period: str = '1d') -> Dict:
        """Get sector ETF performance"""
        logger.info(f"📊 Fetching sector ETF performance ({period})...")
        
        tickers = list(self.sector_etfs.keys())
        
        try:
            data = yf.download(tickers, period='5d', progress=False)
            
            if data.empty:
                return {'error': 'No data available'}
            
            sectors = []
            for ticker, info in self.sector_etfs.items():
                try:
                    if ticker not in data['Close'].columns:
                        continue
                    
                    prices = data['Close'][ticker].dropna()
                    if len(prices) < 2:
                        continue
                    
                    current = prices.iloc[-1]
                    prev_day = prices.iloc[-2]
                    week_ago = prices.iloc[0] if len(prices) >= 5 else prices.iloc[0]
                    
                    daily_change = ((current / prev_day) - 1) * 100
                    weekly_change = ((current / week_ago) - 1) * 100
                    
                    # Volume
                    volume = data['Volume'][ticker].iloc[-1] if 'Volume' in data.columns else 0
                    
                    sectors.append({
                        'ticker': ticker,
                        'name': info['name'],
                        'color': info['color'],
                        'price': round(current, 2),
                        'daily_change': round(daily_change, 2),
                        'weekly_change': round(weekly_change, 2),
                        'volume': int(volume),
                        'heat_color': self._get_color(daily_change)
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing {ticker}: {e}")
                    continue
            
            # Sort by daily change
            sectors.sort(key=lambda x: x['daily_change'], reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'period': period,
                'sectors': sectors,
                'summary': {
                    'best': sectors[0]['name'] if sectors else 'N/A',
                    'worst': sectors[-1]['name'] if sectors else 'N/A',
                    'avg_change': round(sum(s['daily_change'] for s in sectors) / len(sectors), 2) if sectors else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching sector data: {e}")
            return {'error': str(e)}
    
    def get_full_market_map(self, period: str = '5d') -> Dict:
        """Get full market map data (Sectors -> Stocks) for Treemap"""
        logger.info(f"📊 Fetching full market map data ({period})...")
        
        all_tickers = []
        ticker_to_sector = {}
        for sector, stocks in self.sector_stocks.items():
            all_tickers.extend(stocks)
            for stock in stocks:
                ticker_to_sector[stock] = sector
                
        try:
            data = yf.download(all_tickers, period=period, progress=False)
            
            if data.empty:
                return {'error': 'No data'}
            
            market_map = {name: [] for name in self.sector_stocks.keys()}
            
            for ticker in all_tickers:
                try:
                    if ticker not in data['Close'].columns:
                        continue
                    prices = data['Close'][ticker].dropna()
                    if len(prices) < 2:
                        continue
                    
                    current = prices.iloc[-1]
                    prev = prices.iloc[-2]
                    change = ((current / prev) - 1) * 100
                    
                    # Weight by Volume * Price (Activity proxy)
                    vol = data['Volume'][ticker].iloc[-1] if 'Volume' in data.columns else 100000
                    weight = current * vol
                    
                    sector = ticker_to_sector.get(ticker, 'Unknown')
                    if sector in market_map:
                        market_map[sector].append({
                            'x': ticker,
                            'y': round(weight, 0),
                            'price': round(current, 2),
                            'change': round(change, 2),
                            'color': self._get_color(change)
                        })
                except Exception:
                    pass
            
            series = []
            for sector_name, stocks in market_map.items():
                if stocks:
                    stocks.sort(key=lambda x: x['y'], reverse=True)
                    series.append({'name': sector_name, 'data': stocks})
            
            series.sort(key=lambda s: sum(i['y'] for i in s['data']), reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'period': period,
                'series': series
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {'error': str(e)}
            
    def _get_color(self, change: float) -> str:
        """Get color based on change percentage"""
        if change >= 3: return '#00C853'
        elif change >= 1: return '#4CAF50'
        elif change >= 0: return '#81C784'
        elif change >= -1: return '#EF9A9A'
        elif change >= -3: return '#F44336'
        else: return '#B71C1C'

    def save_data(self, output_dir: str = '.'):
        """Save heatmap data to JSON"""
        # Sector performance
        sector_data = self.get_sector_performance('1d')
        
        # Full market map
        market_map = self.get_full_market_map('5d')
        
        # Combine data
        output = {
            'timestamp': datetime.now().isoformat(),
            'sector_performance': sector_data,
            'market_map': market_map
        }
        
        output_file = os.path.join(output_dir, 'sector_heatmap.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved to {output_file}")
        
        # Print summary
        if 'sectors' in sector_data:
            print("\n📊 Sector Performance Summary:")
            for s in sector_data['sectors'][:5]:
                emoji = "🟢" if s['daily_change'] > 0 else "🔴"
                print(f"   {emoji} {s['name']}: {s['daily_change']:+.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sector Heatmap Collector')
    parser.add_argument('--dir', default='.', help='Output directory')
    args = parser.parse_args()
    
    collector = SectorHeatmapCollector()
    collector.save_data(output_dir=args.dir)


if __name__ == "__main__":
    main()
