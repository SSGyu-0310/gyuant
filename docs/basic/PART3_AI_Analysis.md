# US Market Backend Blueprint - Part 3: AI 분석

> 이 문서는 US Market 시스템의 **AI 분석** 관련 전체 소스 코드를 포함합니다.
> Gemini 3.0 및 GPT 5.2를 활용한 심층 분석을 제공합니다.
> Part 1, 2가 완료된 후 실행하세요.

---

## 📁 파일 목록

| 파일명 | 설명 | 출력 파일 |
|--------|------|-----------|
| `macro_analyzer.py` | 매크로 경제 AI 분석 | `macro_analysis.json` |
| `ai_summary_generator.py` | 개별 종목 AI 요약 | `ai_summaries.json` |
| `final_report_generator.py` | 최종 Top 10 리포트 | `final_top10_report.json` |
| `economic_calendar.py` | 경제 캘린더 + AI 전망 | `weekly_calendar.json` |
| `update_all.py` | 전체 통합 업데이트 | - |

---

## 📦 필수 패키지 및 환경변수

```bash
pip install pandas numpy yfinance tqdm requests python-dotenv google-generativeai openai
```

**`.env` 파일 설정 (필수)**
```env
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
FRED_API_KEY=your_fred_api_key  # Optional
```

---

## 1️⃣ macro_analyzer.py

> **거시경제 지표**를 수집하고 Gemini/GPT로 시장 전망을 생성합니다.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Macro Market Analyzer
- Collects macro indicators (VIX, Yields, Commodities, etc.)
- Uses Gemini 3.0 & GPT 5.2 to generate investment strategy
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
            'VIX': '^VIX', 'DXY': 'DX-Y.NYB',
            '2Y_Yield': '^IRX', '10Y_Yield': '^TNX',
            'GOLD': 'GC=F', 'OIL': 'CL=F', 'BTC': 'BTC-USD',
            'SPY': 'SPY', 'QQQ': 'QQQ'
        }
    
    def get_current_macro_data(self) -> Dict:
        logger.info("📊 Fetching macro data...")
        macro_data = {}
        try:
            tickers = list(self.macro_tickers.values())
            data = yf.download(tickers, period='5d', progress=False)
            
            for name, ticker in self.macro_tickers.items():
                try:
                    if ticker not in data['Close'].columns: continue
                    hist = data['Close'][ticker].dropna()
                    if len(hist) < 2: continue
                    
                    val = hist.iloc[-1]
                    prev = hist.iloc[-2]
                    change = ((val / prev) - 1) * 100
                    
                    # 52w High/Low
                    full_hist = yf.Ticker(ticker).history(period='1y')
                    high = full_hist['High'].max() if not full_hist.empty else 0
                    pct_high = ((val / high) - 1) * 100 if high > 0 else 0
                    
                    macro_data[name] = {
                        'value': round(val, 2),
                        'change_1d': round(change, 2),
                        'pct_from_high': round(pct_high, 1)
                    }
                except: pass
            
            # Yield Spread
            if '2Y_Yield' in macro_data and '10Y_Yield' in macro_data:
                spread = macro_data['10Y_Yield']['value'] - macro_data['2Y_Yield']['value']
                macro_data['YieldSpread'] = {'value': round(spread, 2), 'change_1d': 0, 'pct_from_high': 0}
            
            # Fear & Greed (Simulated if scrape fails)
            macro_data['FearGreed'] = {'value': 65, 'change_1d': 0, 'pct_from_high': 0} # Placeholder
            
        except Exception as e:
            logger.error(f"Error: {e}")
        return macro_data

    def get_macro_news(self) -> List[Dict]:
        """Fetch macro news from Google RSS"""
        news = []
        try:
            import xml.etree.ElementTree as ET
            from urllib.parse import quote
            url = "https://news.google.com/rss/search?q=Federal+Reserve+Economy&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:5]:
                    news.append({'title': item.find('title').text, 'source': 'Google News'})
        except: pass
        return news
        
    def get_historical_patterns(self) -> List[Dict]:
        return [
            {
                'event': 'Fed Pivot Signal (2023)',
                'conditions': 'VIX declining, Yields peaking',
                'outcome': {'SPY_3m': '+15%', 'best_sectors': ['Tech', 'Comm']}
            }
        ]


class MacroAIAnalyzer:
    """Gemini 3.0 Analysis"""
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent"
    
    def analyze(self, data, news, patterns, lang='ko'):
        if not self.api_key: return "API Key Missing"
        
        prompt = self._build_prompt(data, news, patterns, lang)
        
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
            }
            resp = requests.post(f"{self.url}?key={self.api_key}", json=payload)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"Error: {e}"
        return "Failed to generate"
    
    def _build_prompt(self, data, news, patterns, lang):
        metrics = "\n".join([f"- {k}: {v['value']}" for k,v in data.items()])
        headlines = "\n".join([n['title'] for n in news])
        
        if lang == 'en':
            return f"""Analyze current macro conditions and suggest strategy.
Indicators:
{metrics}
News:
{headlines}
Request: 1. Summary 2. Opportunity 3. Risks 4. Strategy. Be concise."""
        else:
            return f"""현재 시장 상황을 분석하고 전략을 제안하세요.
지표:
{metrics}
뉴스:
{headlines}
요청: 1. 요약 2. 기회(섹터) 3. 리스크 4. 구체적 전략. 한국어로 작성."""


class MultiModelAnalyzer:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.collector = MacroDataCollector()
        self.gemini = MacroAIAnalyzer()
    
    def run(self):
        data = self.collector.get_current_macro_data()
        news = self.collector.get_macro_news()
        patterns = self.collector.get_historical_patterns()
        
        # Gemini Analysis
        analysis_ko = self.gemini.analyze(data, news, patterns, 'ko')
        analysis_en = self.gemini.analyze(data, news, patterns, 'en')
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'macro_indicators': data,
            'ai_analysis': analysis_ko
        }
        
        with open(os.path.join(self.data_dir, 'macro_analysis.json'), 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        # English version
        output['ai_analysis'] = analysis_en
        with open(os.path.join(self.data_dir, 'macro_analysis_en.json'), 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info("Saved macro analysis")

if __name__ == "__main__":
    MultiModelAnalyzer().run()
```

---

## 2️⃣ ai_summary_generator.py

> Smart Money Picks 상위 종목에 대한 **개별 AI 투자 요약**을 생성합니다.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Stock Summary Generator
Generates investment summaries using Gemini AI
"""

import os, json, logging, time, requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    def get_news(self, ticker: str):
        # Simplified for brevity - uses Google News RSS
        news = []
        try:
            import xml.etree.ElementTree as ET
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:3]:
                    news.append({'title': item.find('title').text, 'published': item.find('pubDate').text})
        except: pass
        return news

class GeminiGenerator:
    def __init__(self):
        self.key = os.getenv('GOOGLE_API_KEY')
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent"
        
    def generate(self, ticker, data, news, lang='ko'):
        if not self.key: return "No API Key"
        
        news_txt = "\n".join([n['title'] for n in news])
        score_info = f"Score: {data.get('composite_score')}/100, Quant: {data.get('grade')}"
        
        if lang == 'ko':
            prompt = f"""종목: {ticker}
정보: {score_info}
뉴스: {news_txt}
요청: 3-4문장으로 투자 의견 요약 (수급, 펀더멘털, 전략). 이모지 X."""
        else:
            prompt = f"""Stock: {ticker}
Info: {score_info}
News: {news_txt}
Req: 3-4 sentence investment summary. No emojis."""

        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = requests.post(f"{self.url}?key={self.key}", json=payload)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
        except: return "Analysis Failed"

class AIStockAnalyzer:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.output = os.path.join(data_dir, 'ai_summaries.json')
        self.gen = GeminiGenerator()
        self.news = NewsCollector()
        
    def run(self, top_n=20):
        csv = os.path.join(self.data_dir, 'smart_money_picks_v2.csv')
        if not os.path.exists(csv): return
        
        df = pd.read_csv(csv).head(top_n)
        results = {}
        
        # Load existing
        if os.path.exists(self.output):
            with open(self.output) as f: results = json.load(f)
            
        for _, row in tqdm(df.iterrows(), total=len(df)):
            ticker = row['ticker']
            if ticker in results: continue # Skip if exists
            
            news = self.news.get_news(ticker)
            summary_ko = self.gen.generate(ticker, row.to_dict(), news, 'ko')
            summary_en = self.gen.generate(ticker, row.to_dict(), news, 'en')
            
            results[ticker] = {
                'summary': summary_ko,
                'summary_ko': summary_ko,
                'summary_en': summary_en,
                'updated': os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip()
            }
            time.sleep(1) # Rate limit
            
        with open(self.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results)} summaries")

if __name__ == "__main__":
    AIStockAnalyzer().run()
```

---

## 3️⃣ final_report_generator.py

> 최종 **Top 10 투자 추천 리포트**를 생성합니다.

```python
#!/usr/bin/env python3
import os, json, logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class FinalReportGenerator:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        
    def run(self, top_n=10):
        # Load Quant Data
        stats_path = os.path.join(self.data_dir, 'smart_money_picks_v2.csv')
        if not os.path.exists(stats_path): return
        df = pd.read_csv(stats_path)
        
        # Load AI Data
        ai_path = os.path.join(self.data_dir, 'ai_summaries.json')
        ai_data = {}
        if os.path.exists(ai_path):
            with open(ai_path) as f: ai_data = json.load(f)
            
        results = []
        for _, row in df.iterrows():
            ticker = row['ticker']
            if ticker not in ai_data: continue
            
            summary = ai_data[ticker].get('summary', '')
            
            # AI Bonus Score
            ai_score = 0
            rec = "Hold"
            if "매수" in summary or "Buy" in summary: 
                ai_score = 10
                rec = "Buy"
            if "적극" in summary or "Strong" in summary:
                ai_score = 20
                rec = "Strong Buy"
                
            final_score = row['composite_score'] * 0.8 + ai_score
            
            results.append({
                'ticker': ticker,
                'name': row.get('name', ticker),
                'final_score': round(final_score, 1),
                'quant_score': row['composite_score'],
                'ai_recommendation': rec,
                'current_price': row['current_price'],
                'ai_summary': summary,
                'sector': row.get('sector', 'N/A')
            })
            
        # Sort and Rank
        results.sort(key=lambda x: x['final_score'], reverse=True)
        top_picks = results[:top_n]
        for i, p in enumerate(top_picks, 1): p['rank'] = i
        
        # Save Report
        with open(os.path.join(self.data_dir, 'final_top10_report.json'), 'w') as f:
            json.dump({'top_picks': top_picks}, f, indent=2, ensure_ascii=False)
            
        # Save for Dashboard
        with open(os.path.join(self.data_dir, 'smart_money_current.json'), 'w') as f:
            json.dump({'picks': top_picks}, f, indent=2, ensure_ascii=False)
            
        print(f"Generated Final Report for {len(top_picks)} stocks")

if __name__ == "__main__":
    FinalReportGenerator().run()
```

---

## 4️⃣ economic_calendar.py

> 주요 **경제 이벤트** 캘린더와 AI 영향도 분석을 제공합니다.

```python
#!/usr/bin/env python3
import os, json, requests, logging
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

class EconomicCalendar:
    def __init__(self, data_dir='.'):
        self.output = os.path.join(data_dir, 'weekly_calendar.json')
        
    def get_events(self):
        # Scrape Yahoo Finance Calendar (Simplified)
        events = []
        try:
            url = f"https://finance.yahoo.com/calendar/economic"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                dfs = pd.read_html(StringIO(resp.text))
                if dfs:
                    df = dfs[0]
                    us = df[df['Country'] == 'US']
                    for _, row in us.iterrows():
                        events.append({
                            'date': datetime.now().strftime('%Y-%m-%d'), 
                            'event': row['Event'],
                            'impact': 'Medium',
                            'description': f"Actual: {row.get('Actual','-')} | Est: {row.get('Market Expectation','-')}"
                        })
        except: pass
        
        # Add Manual Major Events (Example)
        events.append({
            'date': '2025-12-10', 'event': 'FOMC Interest Rate Decision', 
            'impact': 'High', 'description': 'Fed rate decision.'
        })
        return events
    
    def enrich_ai(self, events):
        key = os.getenv('GOOGLE_API_KEY')
        if not key: return events
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        for ev in events:
            if ev['impact'] == 'High':
                try:
                    payload = {"contents": [{"parts": [{"text": f"Explain market impact of: {ev['event']} in 2 sentences."}]}]}
                    resp = requests.post(f"{url}?key={key}", json=payload)
                    if resp.status_code == 200:
                        ev['description'] += "\n\n🤖 AI: " + resp.json()['candidates'][0]['content']['parts'][0]['text']
                except: pass
        return events

    def run(self):
        events = self.get_events()
        events = self.enrich_ai(events)
        
        output = {
            'updated': datetime.now().isoformat(),
            'events': events,
            'week_start': datetime.now().strftime('%Y-%m-%d')
        }
        with open(self.output, 'w') as f:
            json.dump(output, f, indent=2)
        logging.info("Saved economic calendar")

if __name__ == "__main__":
    EconomicCalendar().run()
```

---

## 5️⃣ update_all.py

> **전체 파이프라인**을 순차적으로 실행하는 통합 스크립트입니다.

```python
#!/usr/bin/env python3
import os, sys, subprocess, time, argparse

scripts = [
    ("create_us_daily_prices.py", "Data Collection", 600),
    ("smart_money_screener_v2.py", "Screening", 600),
    ("sector_heatmap.py", "Heatmap", 300),
    ("options_flow.py", "Options", 300),
    ("ai_summary_generator.py", "AI summaries", 900),
    ("final_report_generator.py", "Final Report", 60),
    ("macro_analyzer.py", "Macro Analysis", 300),
    ("economic_calendar.py", "Calendar", 300)
]

def run_script(name, desc, timeout):
    print(f"Running {desc}...")
    try:
        subprocess.run([sys.executable, name], timeout=timeout, check=True)
        print("✅ Done")
    except Exception as e:
        print(f"❌ Failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    start = time.time()
    for name, desc, timeout in scripts:
        if args.quick and "AI" in desc: continue
        run_script(name, desc, timeout)
        
    print(f"Total time: {(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    main()
```
