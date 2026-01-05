#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Top 10 Report Generator
Combines quant scores with AI analysis for final recommendations
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalReportGenerator:
    """Generate final investment report combining quant and AI"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'final_top10_report.json')
        self.dashboard_file = os.path.join(data_dir, 'smart_money_current.json')
        
    def run(self, top_n: int = 10) -> Dict:
        """Generate final report"""
        logger.info(f"📊 Generating Final Top {top_n} Report...")
        
        # Load Quant Data
        quant_path = os.path.join(self.data_dir, 'smart_money_picks_v2.csv')
        if not os.path.exists(quant_path):
            logger.warning("⚠️ smart_money_picks_v2.csv not found")
            return {'error': 'Quant data not found'}
        
        df = pd.read_csv(quant_path)
        logger.info(f"📂 Loaded {len(df)} stocks from quant screening")
        
        # Load AI Summaries
        ai_path = os.path.join(self.data_dir, 'ai_summaries.json')
        ai_data = {}
        if os.path.exists(ai_path):
            with open(ai_path, 'r', encoding='utf-8') as f:
                ai_data = json.load(f)
            logger.info(f"📂 Loaded {len(ai_data)} AI summaries")
        else:
            logger.warning("⚠️ AI summaries not found. Using quant data only.")
        
        # Process and score
        results = []
        for _, row in df.iterrows():
            ticker = row['ticker']
            
            # Get AI data if available
            ai_info = ai_data.get(ticker, {})
            summary = ai_info.get('summary', '')
            summary_en = ai_info.get('summary_en', summary)
            
            # Calculate AI bonus based on sentiment
            ai_bonus = 0
            recommendation = "Hold"
            
            if summary:
                summary_lower = summary.lower()
                if any(word in summary_lower for word in ['매수', 'buy', 'bullish', '긍정']):
                    ai_bonus = 5
                    recommendation = "Buy"
                if any(word in summary_lower for word in ['적극', 'strong', 'outperform', '강력']):
                    ai_bonus = 10
                    recommendation = "Strong Buy"
                if any(word in summary_lower for word in ['매도', 'sell', 'bearish', '부정']):
                    ai_bonus = -5
                    recommendation = "Sell"
            
            # Final score (80% quant + 20% AI bonus)
            quant_score = row.get('composite_score', 50)
            final_score = quant_score * 0.8 + (50 + ai_bonus) * 0.2
            
            results.append({
                'ticker': ticker,
                'name': str(row.get('name', ticker)) if pd.notna(row.get('name')) else ticker,
                'final_score': round(final_score, 1),
                'quant_score': round(quant_score, 1),
                'quant_grade': row.get('grade', 'N/A'),
                'ai_recommendation': recommendation,
                'current_price': row.get('current_price', 0),
                'target_upside': row.get('target_upside', 0),
                'ai_summary': summary[:500] if summary else 'No AI analysis available',
                'ai_summary_en': summary_en[:500] if summary_en else 'No AI analysis available',
                'sector': row.get('sector', 'N/A'),
                'size': row.get('size', 'N/A'),
                'sub_scores': {
                    'supply_demand': row.get('sd_score', 50),
                    'technical': row.get('tech_score', 50),
                    'fundamental': row.get('fund_score', 50),
                    'analyst': row.get('analyst_score', 50),
                    'relative_strength': row.get('rs_score', 50)
                }
            })
        
        # Sort by final score and get top N
        results.sort(key=lambda x: x['final_score'], reverse=True)
        top_picks = results[:top_n]
        
        # Add ranks
        for i, pick in enumerate(top_picks, 1):
            pick['rank'] = i
        
        # Calculate summary stats
        avg_score = sum(p['final_score'] for p in top_picks) / len(top_picks) if top_picks else 0
        strong_buys = len([p for p in top_picks if p['ai_recommendation'] == 'Strong Buy'])
        
        # Build output
        output = {
            'timestamp': datetime.now().isoformat(),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'summary': {
                'total_analyzed': len(df),
                'top_picks_count': len(top_picks),
                'avg_final_score': round(avg_score, 1),
                'strong_buy_count': strong_buys,
                'market_sentiment': 'Bullish' if avg_score > 65 else ('Neutral' if avg_score > 50 else 'Bearish')
            },
            'top_picks': top_picks
        }
        
        # Save main report
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved report to {self.output_file}")
        
        # Save dashboard-friendly version
        dashboard_output = {
            'updated': output['timestamp'],
            'picks': top_picks,
            'summary': output['summary']
        }
        with open(self.dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved dashboard data to {self.dashboard_file}")
        
        return output
    
    def print_report(self, output: Dict):
        """Print report summary"""
        if 'error' in output:
            print(f"❌ {output['error']}")
            return
            
        summary = output.get('summary', {})
        picks = output.get('top_picks', [])
        
        print("\n" + "=" * 60)
        print("🏆 FINAL TOP 10 INVESTMENT PICKS")
        print("=" * 60)
        print(f"Generated: {output.get('generated_at', 'N/A')}")
        print(f"Market Sentiment: {summary.get('market_sentiment', 'N/A')}")
        print(f"Avg Score: {summary.get('avg_final_score', 0):.1f}")
        print(f"Strong Buys: {summary.get('strong_buy_count', 0)}")
        print("-" * 60)
        
        for pick in picks:
            rec_emoji = {"Strong Buy": "🔥", "Buy": "📈", "Hold": "📊", "Sell": "⚠️"}.get(pick['ai_recommendation'], "📊")
            name_display = pick['name'][:25] if isinstance(pick['name'], str) else pick['ticker']
            print(f"\n#{pick['rank']} {pick['ticker']} ({name_display})")
            print(f"   Score: {pick['final_score']} | {rec_emoji} {pick['ai_recommendation']}")
            print(f"   Price: ${pick['current_price']} | Upside: {pick['target_upside']}%")
            print(f"   Quant: {pick['quant_grade']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Report Generator')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--top', type=int, default=10, help='Number of top picks')
    args = parser.parse_args()
    
    generator = FinalReportGenerator(data_dir=args.dir)
    output = generator.run(top_n=args.top)
    generator.print_report(output)


if __name__ == "__main__":
    main()
