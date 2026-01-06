#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Performance Heatmap Data Collector
Collects sector ETF performance data for heatmap visualization
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.fmp_client import get_fmp_client
from utils.symbols import map_symbols_to_fmp, to_fmp_symbol
from utils.db_writer import write_market_documents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SectorHeatmapCollector:
    """Collect sector ETF performance data for heatmap visualization"""
    
    # Mapping from short sector codes (in SECTOR_MAP) to full sector names (for ETFs)
    SECTOR_NAME_MAP = {
        'Tech': 'Technology',
        'Fin': 'Financials',
        'Health': 'Healthcare',
        'Energy': 'Energy',
        'Cons': 'Consumer Disc.',
        'Staple': 'Consumer Staples',
        'Indust': 'Industrials',
        'Mater': 'Materials',
        'Util': 'Utilities',
        'REIT': 'Real Estate',
        'Comm': 'Comm. Services',
    }
    
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

        self.fmp = get_fmp_client()
        
        # Load sector stocks from CSV or use fallback
        self.sector_stocks = self._load_sectors_from_csv()

    def _get_recent_history(self, ticker: str, days: int = 5) -> List[Dict]:
        end_date = datetime.utcnow().date()
        from_date = (end_date - timedelta(days=days)).isoformat()
        data = self.fmp.historical_price_full(
            to_fmp_symbol(ticker),
            from_date=from_date,
            to_date=end_date.isoformat(),
        )
        hist = data.get("historical", []) if isinstance(data, dict) else []
        return sorted(hist, key=lambda x: x.get("date", ""))
    
    def _load_sectors_from_csv(self) -> Dict[str, List[str]]:
        """Load sector stocks from us_sectors.csv with fallback to hardcoded data."""
        fallback = {
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
        
        # Try to load from CSV
        csv_candidates = [
            os.path.join(os.path.dirname(__file__), 'us_sectors.csv'),
            os.path.join(os.getenv('DATA_DIR', '.'), 'us_sectors.csv'),
        ]
        
        for csv_path in csv_candidates:
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if 'ticker' not in df.columns or 'sector' not in df.columns:
                    logger.warning(f"CSV missing required columns: {csv_path}")
                    continue
                
                df = df.dropna(subset=['ticker', 'sector'])
                if df.empty:
                    continue
                
                # Group by sector and convert short codes to full names
                sector_stocks: Dict[str, List[str]] = {}
                for _, row in df.iterrows():
                    short_sector = str(row['sector']).strip()
                    full_sector = self.SECTOR_NAME_MAP.get(short_sector, short_sector)
                    ticker = str(row['ticker']).strip().upper()
                    
                    if full_sector not in sector_stocks:
                        sector_stocks[full_sector] = []
                    sector_stocks[full_sector].append(ticker)
                
                # Only use if we have reasonable coverage
                if len(sector_stocks) >= 5:
                    logger.info(f"✅ Loaded {sum(len(v) for v in sector_stocks.values())} stocks from {csv_path}")
                    
                    # Log sector counts
                    for sec, stocks in sorted(sector_stocks.items()):
                        logger.debug(f"   {sec}: {len(stocks)} stocks")
                    
                    return sector_stocks
                    
            except Exception as e:
                logger.warning(f"Error loading {csv_path}: {e}")
                continue
        
        logger.info("Using fallback sector stocks (CSV not found or invalid)")
        return fallback
    
    def get_sector_performance(self, period: str = '1d') -> Dict:
        """Get sector ETF performance"""
        logger.info(f"📊 Fetching sector ETF performance ({period})...")
        
        tickers = list(self.sector_etfs.keys())
        
        try:
            fmp_symbols, _ = map_symbols_to_fmp(tickers)
            quotes = self.fmp.quote(fmp_symbols)
            quote_map = {q.get("symbol"): q for q in quotes if isinstance(q, dict)}
            
            if not quote_map:
                return {'error': 'No data available'}
            
            sectors = []
            for ticker, info in self.sector_etfs.items():
                try:
                    fmp_symbol = to_fmp_symbol(ticker)
                    quote = quote_map.get(fmp_symbol)
                    if not quote:
                        continue
                    current = quote.get("price") or quote.get("previousClose")
                    if current is None:
                        continue

                    hist = self._get_recent_history(ticker, days=7)
                    if len(hist) < 2:
                        continue
                    prev_day = hist[-2].get("close") if len(hist) >= 2 else current
                    week_ago = hist[-5].get("close") if len(hist) >= 5 else hist[0].get("close")
                    daily_change = ((current / prev_day) - 1) * 100 if prev_day else 0
                    weekly_change = ((current / week_ago) - 1) * 100 if week_ago else 0

                    volume = quote.get("volume") or 0
                    
                    sectors.append({
                        'ticker': ticker,
                        'name': info['name'],
                        'color': info['color'],
                        'price': round(float(current), 2),
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
            fmp_symbols, _ = map_symbols_to_fmp(all_tickers)
            quotes = self.fmp.quote(fmp_symbols)
            quote_map = {q.get("symbol"): q for q in quotes if isinstance(q, dict)}
            
            if not quote_map:
                return {'error': 'No data'}
            
            market_map = {name: [] for name in self.sector_stocks.keys()}
            
            for ticker in all_tickers:
                try:
                    fmp_symbol = to_fmp_symbol(ticker)
                    quote = quote_map.get(fmp_symbol)
                    if not quote:
                        continue
                    current = quote.get("price")
                    prev = quote.get("previousClose")
                    if current is None or prev is None:
                        continue
                    change = ((current / prev) - 1) * 100 if prev else 0
                    
                    vol = quote.get("volume") or 100000
                    weight = current * vol
                    
                    sector = ticker_to_sector.get(ticker, 'Unknown')
                    if sector in market_map:
                        market_map[sector].append({
                            'x': ticker,
                            'y': round(float(weight), 0),
                            'price': round(float(current), 2),
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
        """Get color based on change percentage (muted/desaturated palette)"""
        # Muted green palette for positive changes
        if change >= 3: return '#2E8B57'      # SeaGreen (muted dark green)
        elif change >= 1: return '#4A9A6A'    # Medium muted green
        elif change >= 0: return '#6AAF8B'    # Light muted green
        # Muted red palette for negative changes  
        elif change >= -1: return '#C08080'   # Muted light red/pink
        elif change >= -3: return '#B25A5A'   # Medium muted red
        else: return '#8B3A3A'                # Dark muted red

    def save_data(self, output_dir: str = '.'):
        """Save heatmap data to JSON"""
        # Sector performance
        sector_data = self.get_sector_performance('1d')
        
        # Full market map
        market_map = self.get_full_market_map('5d')
        
        # Flatten structure for frontend compatibility
        # Frontend expects 'series' at top level, and optionally 'sectors' for ETF data
        output = {
            'timestamp': datetime.now().isoformat(),
            'data_date': datetime.now().strftime('%Y-%m-%d'),
            # Top-level series for treemap (from market_map)
            'series': market_map.get('series', []),
            # Sector ETF performance data
            'sectors': sector_data.get('sectors', []),
            'summary': sector_data.get('summary', {}),
            # Keep original nested structure for backward compatibility
            'sector_performance': sector_data,
            'market_map': market_map
        }
        
        output_file = os.path.join(output_dir, 'sector_heatmap.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved to {output_file}")
        write_market_documents(
            "sector_heatmap",
            output,
            as_of_date=output.get("data_date"),
        )
        
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
