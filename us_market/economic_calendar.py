#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Economic Calendar with AI Impact Analysis
Fetches economic events and provides AI-powered impact assessment
"""

import os
import json
import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EconomicCalendar:
    """Economic calendar with AI analysis"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'weekly_calendar.json')
        self.api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        # Important recurring events
        self.major_events = [
            {'event': 'FOMC Interest Rate Decision', 'impact': 'High', 'frequency': 'Every 6 weeks'},
            {'event': 'Non-Farm Payrolls', 'impact': 'High', 'frequency': 'First Friday monthly'},
            {'event': 'CPI (Consumer Price Index)', 'impact': 'High', 'frequency': 'Monthly'},
            {'event': 'PPI (Producer Price Index)', 'impact': 'Medium', 'frequency': 'Monthly'},
            {'event': 'Retail Sales', 'impact': 'Medium', 'frequency': 'Monthly'},
            {'event': 'GDP Report', 'impact': 'High', 'frequency': 'Quarterly'},
            {'event': 'Jobless Claims', 'impact': 'Medium', 'frequency': 'Weekly (Thursday)'},
            {'event': 'Consumer Confidence', 'impact': 'Medium', 'frequency': 'Monthly'},
        ]
    
    def get_events(self) -> List[Dict]:
        """Get economic events"""
        events = []
        
        # Try to scrape Yahoo Finance calendar
        try:
            url = "https://finance.yahoo.com/calendar/economic"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                try:
                    dfs = pd.read_html(StringIO(resp.text))
                    if dfs:
                        df = dfs[0]
                        # Filter for US events if Country column exists
                        if 'Country' in df.columns:
                            df = df[df['Country'] == 'US']
                        
                        for _, row in df.head(20).iterrows():
                            event_name = row.get('Event', row.get('Name', 'Unknown'))
                            
                            # Determine impact level
                            impact = 'Medium'
                            high_impact_keywords = ['FOMC', 'Fed', 'CPI', 'GDP', 'Payroll', 'NFP', 'Interest Rate']
                            if any(kw.lower() in str(event_name).lower() for kw in high_impact_keywords):
                                impact = 'High'
                            
                            events.append({
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'event': str(event_name),
                                'impact': impact,
                                'actual': str(row.get('Actual', '-')),
                                'expected': str(row.get('Market Expectation', row.get('Expected', '-'))),
                                'previous': str(row.get('Prior', row.get('Previous', '-'))),
                                'description': ''
                            })
                except Exception as e:
                    logger.debug(f"Error parsing Yahoo calendar: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not fetch Yahoo calendar: {e}")
        
        # Add manual important events for this week
        today = datetime.now()
        week_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
        
        # Common recurring events
        manual_events = [
            {
                'date': week_dates[0],
                'event': 'Market Open Analysis',
                'impact': 'Low',
                'description': 'Weekly market conditions review'
            }
        ]
        
        # Add weekly Thursday jobless claims
        for i, date in enumerate(week_dates):
            if (today + timedelta(days=i)).weekday() == 3:  # Thursday
                manual_events.append({
                    'date': date,
                    'event': 'Initial Jobless Claims',
                    'impact': 'Medium',
                    'description': 'Weekly unemployment claims data'
                })
        
        events.extend(manual_events)
        
        # Sort by date
        events.sort(key=lambda x: x['date'])
        
        return events
    
    def enrich_with_ai(self, events: List[Dict]) -> List[Dict]:
        """Add AI analysis for high-impact events"""
        if not self.api_key:
            logger.warning("⚠️ No API key for AI enrichment")
            return events
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        for event in events:
            if event.get('impact') != 'High':
                continue
                
            try:
                prompt = f"""Explain the potential market impact of this economic event in 2 concise sentences:
Event: {event['event']}
Expected: {event.get('expected', 'N/A')}
Previous: {event.get('previous', 'N/A')}

Focus on: How this might affect stocks, bonds, and sectors. Be specific."""

                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.5, "maxOutputTokens": 200}
                }
                
                resp = requests.post(f"{url}?key={self.api_key}", json=payload, timeout=15)
                
                if resp.status_code == 200:
                    ai_text = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    event['ai_analysis'] = ai_text.strip()
                    event['description'] = f"{event.get('description', '')}\n\n🤖 AI: {ai_text}".strip()
                    
            except Exception as e:
                logger.debug(f"AI enrichment failed for {event['event']}: {e}")
                
        return events
    
    def run(self) -> Dict:
        """Run calendar generation"""
        logger.info("📅 Generating Economic Calendar...")
        
        # Get events
        events = self.get_events()
        logger.info(f"📊 Found {len(events)} events")
        
        # AI enrichment for high-impact events
        high_impact = [e for e in events if e.get('impact') == 'High']
        if high_impact:
            logger.info(f"🤖 Enriching {len(high_impact)} high-impact events with AI...")
            events = self.enrich_with_ai(events)
        
        # Build output
        today = datetime.now()
        output = {
            'updated': today.isoformat(),
            'week_start': today.strftime('%Y-%m-%d'),
            'week_end': (today + timedelta(days=6)).strftime('%Y-%m-%d'),
            'summary': {
                'total_events': len(events),
                'high_impact': len([e for e in events if e.get('impact') == 'High']),
                'medium_impact': len([e for e in events if e.get('impact') == 'Medium'])
            },
            'events': events,
            'major_recurring_events': self.major_events
        }
        
        # Save output
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved to {self.output_file}")
        
        return output
    
    def print_calendar(self, output: Dict):
        """Print calendar summary"""
        print("\n📅 Economic Calendar")
        print("=" * 50)
        print(f"Week: {output.get('week_start')} to {output.get('week_end')}")
        
        summary = output.get('summary', {})
        print(f"Events: {summary.get('total_events', 0)} total ({summary.get('high_impact', 0)} high-impact)")
        print("-" * 50)
        
        events = output.get('events', [])
        for event in events[:10]:
            impact_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(event.get('impact'), '⚪')
            print(f"\n{impact_emoji} {event.get('date')} - {event.get('event')}")
            if event.get('ai_analysis'):
                print(f"   🤖 {event['ai_analysis'][:150]}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Economic Calendar Generator')
    parser.add_argument('--dir', default='.', help='Data directory')
    args = parser.parse_args()
    
    calendar = EconomicCalendar(data_dir=args.dir)
    output = calendar.run()
    calendar.print_calendar(output)


if __name__ == "__main__":
    main()
