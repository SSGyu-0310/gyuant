#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Stock Summary Generator
Generates investment summaries for top picks using Gemini AI
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env

load_env()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsCollector:
    """Collect recent news for stocks"""
    
    def get_news(self, ticker: str) -> list:
        """Fetch news from Google RSS"""
        news = []
        try:
            import xml.etree.ElementTree as ET
            
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:3]:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    news.append({
                        'title': title.text if title is not None else '',
                        'published': pub_date.text if pub_date is not None else ''
                    })
        except Exception as e:
            logger.debug(f"News fetch error for {ticker}: {e}")
            
        return news


class GeminiGenerator:
    """Generate AI summaries using Gemini"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
    def generate(self, ticker: str, data: dict, news: list, lang: str = 'ko') -> str:
        """Generate investment summary"""
        if not self.api_key:
            return "API Key not configured"
        
        # Format news
        news_txt = "\n".join([n.get('title', '') for n in news]) if news else "No recent news"
        
        # Format stock data
        score_info = f"Composite Score: {data.get('composite_score', 'N/A')}/100"
        grade = data.get('grade', 'N/A')
        price = data.get('current_price', 'N/A')
        upside = data.get('target_upside', 'N/A')
        
        if lang == 'ko':
            prompt = f"""종목: {ticker}
현재가: ${price}
등급: {grade}
종합점수: {score_info}
목표 상승률: {upside}%
최근 뉴스:
{news_txt}

요청: 이 종목에 대해 3-4문장으로 투자 의견을 작성하세요.
수급, 기술적, 펀더멘털 측면을 간략히 언급하고 투자 전략을 제안하세요.
이모지 사용하지 마세요. 전문적으로 작성하세요."""
        else:
            prompt = f"""Stock: {ticker}
Current Price: ${price}
Grade: {grade}
Score: {score_info}
Target Upside: {upside}%
Recent News:
{news_txt}

Request: Write a 3-4 sentence investment opinion on this stock.
Briefly cover supply/demand, technical, and fundamental aspects with a strategy suggestion.
No emojis. Professional tone."""

        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
            }
            resp = requests.post(
                f"{self.url}?key={self.api_key}",
                json=payload,
                timeout=30
            )
            
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"API Error: {resp.status_code}"
                
        except Exception as e:
            return f"Generation failed: {e}"


class AIStockAnalyzer:
    """Main AI stock analyzer"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'ai_summaries.json')
        self.generator = GeminiGenerator()
        self.news_collector = NewsCollector()
        
    def run(self, top_n: int = 20) -> dict:
        """Generate AI summaries for top stocks"""
        logger.info(f"🤖 Starting AI Summary Generation (top {top_n})...")
        
        # Load smart money picks
        csv_path = os.path.join(self.data_dir, 'smart_money_picks_v2.csv')
        if not os.path.exists(csv_path):
            logger.warning("⚠️ smart_money_picks_v2.csv not found. Run smart_money_screener_v2.py first.")
            return {}
        
        df = pd.read_csv(csv_path).head(top_n)
        logger.info(f"📊 Processing {len(df)} stocks")
        
        # Load existing summaries
        results = {}
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"📂 Loaded {len(results)} existing summaries")
            except:
                results = {}
        
        # Generate summaries
        new_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
            ticker = row['ticker']
            
            # Skip if already exists and recent (within 24 hours)
            if ticker in results:
                existing = results[ticker]
                if 'updated' in existing:
                    try:
                        updated = datetime.fromisoformat(existing['updated'].replace('Z', '+00:00'))
                        if (datetime.now(updated.tzinfo) - updated).total_seconds() < 86400:
                            continue
                    except:
                        pass
            
            # Get news
            news = self.news_collector.get_news(ticker)
            
            # Generate summaries in both languages
            summary_ko = self.generator.generate(ticker, row.to_dict(), news, 'ko')
            summary_en = self.generator.generate(ticker, row.to_dict(), news, 'en')
            
            results[ticker] = {
                'ticker': ticker,
                'name': row.get('name', ticker),
                'summary': summary_ko,
                'summary_ko': summary_ko,
                'summary_en': summary_en,
                'composite_score': row.get('composite_score', 0),
                'grade': row.get('grade', 'N/A'),
                'current_price': row.get('current_price', 0),
                'news_count': len(news),
                'updated': datetime.utcnow().isoformat() + 'Z'
            }
            
            new_count += 1
            time.sleep(1)  # Rate limiting
        
        # Save results
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated {new_count} new summaries, total {len(results)}")
        logger.info(f"📁 Saved to {self.output_file}")
        
        return results
    
    def print_summary(self, results: dict):
        """Print summary of generated content"""
        if not results:
            print("No results to display")
            return
            
        print(f"\n🤖 AI Summaries Generated: {len(results)}")
        print("=" * 50)
        
        for ticker, data in list(results.items())[:5]:
            print(f"\n📈 {ticker} ({data.get('name', 'N/A')})")
            print(f"   Score: {data.get('composite_score', 'N/A')} | Grade: {data.get('grade', 'N/A')}")
            summary = data.get('summary', '')[:200]
            print(f"   Summary: {summary}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Stock Summary Generator')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--top', type=int, default=20, help='Number of stocks to process')
    args = parser.parse_args()
    
    analyzer = AIStockAnalyzer(data_dir=args.dir)
    results = analyzer.run(top_n=args.top)
    analyzer.print_summary(results)


if __name__ == "__main__":
    main()
