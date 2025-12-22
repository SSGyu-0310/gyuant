#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Macro Market Analyzer
- Collects macro indicators (VIX, Yields, Commodities, etc.)
- Uses Gemini AI to generate investment strategy
"""

import os
import json
import requests
import yfinance as yf
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MacroDataCollector:
    """Collect macro market data from various sources"""
    
    def __init__(self):
        self.macro_tickers = {
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',
            '2Y_Yield': '^IRX',
            '10Y_Yield': '^TNX',
            'GOLD': 'GC=F',
            'OIL': 'CL=F',
            'BTC': 'BTC-USD',
            'SPY': 'SPY',
            'QQQ': 'QQQ'
        }
    
    def get_current_macro_data(self) -> Dict:
        """Fetch current macro indicators"""
        logger.info("📊 Fetching macro data...")
        macro_data = {}
        
        try:
            tickers = list(self.macro_tickers.values())
            data = yf.download(tickers, period='5d', progress=False)
            
            for name, ticker in self.macro_tickers.items():
                try:
                    if ticker not in data['Close'].columns:
                        continue
                    hist = data['Close'][ticker].dropna()
                    if len(hist) < 2:
                        continue
                    
                    val = hist.iloc[-1]
                    prev = hist.iloc[-2]
                    change = ((val / prev) - 1) * 100
                    
                    # 52-week High/Low (simplified)
                    try:
                        full_hist = yf.Ticker(ticker).history(period='1y')
                        high = full_hist['High'].max() if not full_hist.empty else val
                        low = full_hist['Low'].min() if not full_hist.empty else val
                        pct_high = ((val / high) - 1) * 100 if high > 0 else 0
                    except:
                        high, low, pct_high = val, val, 0
                    
                    macro_data[name] = {
                        'value': round(float(val), 2),
                        'change_1d': round(float(change), 2),
                        'pct_from_high': round(float(pct_high), 1),
                        '52w_high': round(float(high), 2),
                        '52w_low': round(float(low), 2)
                    }
                except Exception as e:
                    logger.debug(f"Error processing {name}: {e}")
                    continue
            
            # Yield Spread (10Y - 2Y)
            if '2Y_Yield' in macro_data and '10Y_Yield' in macro_data:
                spread = macro_data['10Y_Yield']['value'] - macro_data['2Y_Yield']['value']
                macro_data['YieldSpread'] = {
                    'value': round(spread, 2),
                    'change_1d': 0,
                    'pct_from_high': 0,
                    'interpretation': 'Normal' if spread > 0 else 'Inverted (Recession Warning)'
                }
            
            # Fear & Greed (placeholder - would need CNN API)
            if 'VIX' in macro_data:
                vix = macro_data['VIX']['value']
                if vix > 30:
                    fg_score = 25  # Fear
                    fg_label = "Extreme Fear"
                elif vix > 20:
                    fg_score = 40
                    fg_label = "Fear"
                elif vix > 15:
                    fg_score = 55
                    fg_label = "Neutral"
                else:
                    fg_score = 75
                    fg_label = "Greed"
                    
                macro_data['FearGreed'] = {
                    'value': fg_score,
                    'label': fg_label,
                    'change_1d': 0
                }
                
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            
        return macro_data

    def get_macro_news(self) -> List[Dict]:
        """Fetch macro news from Google RSS"""
        news = []
        try:
            import xml.etree.ElementTree as ET
            
            url = "https://news.google.com/rss/search?q=Federal+Reserve+Economy&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:5]:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    news.append({
                        'title': title.text if title is not None else 'No title',
                        'published': pub_date.text if pub_date is not None else '',
                        'source': 'Google News'
                    })
        except Exception as e:
            logger.debug(f"Error fetching news: {e}")
            
        return news
        
    def get_historical_patterns(self) -> List[Dict]:
        """Get historical market patterns for context"""
        return [
            {
                'event': 'Fed Pivot Signal (2023)',
                'conditions': 'VIX declining, Yields peaking',
                'outcome': {'SPY_3m': '+15%', 'best_sectors': ['Tech', 'Communications']}
            },
            {
                'event': 'Inflation Peak (2022)',
                'conditions': 'CPI declining, Fed pause',
                'outcome': {'SPY_3m': '+8%', 'best_sectors': ['Growth', 'Consumer Discretionary']}
            }
        ]


class MacroAIAnalyzer:
    """Gemini AI Analysis for Macro Data"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    def analyze(self, data: Dict, news: List, patterns: List, lang: str = 'ko') -> str:
        """Generate AI analysis"""
        if not self.api_key:
            return "⚠️ GOOGLE_API_KEY not set. Please add it to .env file."
        
        prompt = self._build_prompt(data, news, patterns, lang)
        
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
            }
            resp = requests.post(
                f"{self.url}?key={self.api_key}",
                json=payload,
                timeout=30
            )
            
            if resp.status_code == 200:
                result = resp.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"API Error: {resp.status_code} - {resp.text[:200]}"
                
        except Exception as e:
            return f"Error generating analysis: {e}"
    
    def _build_prompt(self, data: Dict, news: List, patterns: List, lang: str) -> str:
        """Build analysis prompt"""
        # Format metrics
        metrics_lines = []
        for k, v in data.items():
            if isinstance(v, dict):
                val = v.get('value', 'N/A')
                chg = v.get('change_1d', 0)
                metrics_lines.append(f"- {k}: {val} ({chg:+.2f}%)")
            else:
                metrics_lines.append(f"- {k}: {v}")
        metrics = "\n".join(metrics_lines)
        
        # Format headlines
        headlines = "\n".join([n['title'] for n in news[:5]])
        
        if lang == 'en':
            return f"""You are a professional macro market analyst. Analyze current conditions and provide investment strategy.

## Current Macro Indicators:
{metrics}

## Recent News:
{headlines}

## Request:
Provide a concise analysis covering:
1. **Market Summary** (2-3 sentences)
2. **Key Opportunities** (sectors/assets to favor)
3. **Key Risks** (what to watch)
4. **Recommended Strategy** (specific actionable advice)

Be direct and professional. No emojis."""
        else:
            return f"""당신은 전문 매크로 시장 분석가입니다. 현재 상황을 분석하고 투자 전략을 제안하세요.

## 현재 매크로 지표:
{metrics}

## 최근 뉴스:
{headlines}

## 요청:
다음을 포함한 간결한 분석을 제공하세요:
1. **시장 요약** (2-3문장)
2. **기회 요인** (주목할 섹터/자산)
3. **리스크 요인** (주의할 점)
4. **추천 전략** (구체적 조언)

전문적이고 직접적으로 작성하세요. 이모지 사용하지 마세요."""


class MultiModelAnalyzer:
    """Main analyzer combining data collection and AI analysis"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.collector = MacroDataCollector()
        self.gemini = MacroAIAnalyzer()
        self.output_file = os.path.join(data_dir, 'macro_analysis.json')
    
    def run(self) -> Dict:
        """Run full macro analysis"""
        logger.info("🚀 Starting Macro Market Analysis...")
        
        # Collect data
        data = self.collector.get_current_macro_data()
        news = self.collector.get_macro_news()
        patterns = self.collector.get_historical_patterns()
        
        logger.info(f"📊 Collected {len(data)} indicators, {len(news)} news items")
        
        # AI Analysis
        logger.info("🤖 Generating AI analysis...")
        analysis_ko = self.gemini.analyze(data, news, patterns, 'ko')
        analysis_en = self.gemini.analyze(data, news, patterns, 'en')
        
        # Build output
        output = {
            'timestamp': datetime.now().isoformat(),
            'macro_indicators': data,
            'news': news,
            'historical_patterns': patterns,
            'ai_analysis': {
                'ko': analysis_ko,
                'en': analysis_en
            }
        }
        
        # Save Korean version
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved to {self.output_file}")
        
        # Save English version
        en_output_file = os.path.join(self.data_dir, 'macro_analysis_en.json')
        en_output = {
            'timestamp': output['timestamp'],
            'macro_indicators': data,
            'ai_analysis': analysis_en
        }
        with open(en_output_file, 'w') as f:
            json.dump(en_output, f, indent=2)
        
        return output
    
    def print_summary(self, output: Dict):
        """Print analysis summary"""
        print("\n📊 Macro Market Summary")
        print("=" * 50)
        
        data = output.get('macro_indicators', {})
        
        # Key indicators
        key_indicators = ['VIX', 'SPY', 'QQQ', '10Y_Yield', 'GOLD', 'BTC']
        for ind in key_indicators:
            if ind in data:
                d = data[ind]
                val = d.get('value', 'N/A')
                chg = d.get('change_1d', 0)
                emoji = "🟢" if chg > 0 else "🔴" if chg < 0 else "⚪"
                print(f"   {emoji} {ind}: {val} ({chg:+.2f}%)")
        
        # Fear & Greed
        if 'FearGreed' in data:
            fg = data['FearGreed']
            print(f"\n   💭 Fear & Greed: {fg.get('value')} ({fg.get('label', 'N/A')})")
        
        # AI Summary (first 500 chars)
        ai = output.get('ai_analysis', {})
        if isinstance(ai, dict):
            analysis = ai.get('ko', ai.get('en', ''))
        else:
            analysis = str(ai)
            
        if analysis:
            print(f"\n🤖 AI Analysis Preview:")
            print(f"   {analysis[:500]}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Macro Market Analyzer')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--lang', default='ko', choices=['ko', 'en'], help='Output language')
    args = parser.parse_args()
    
    analyzer = MultiModelAnalyzer(data_dir=args.dir)
    output = analyzer.run()
    analyzer.print_summary(output)


if __name__ == "__main__":
    main()
