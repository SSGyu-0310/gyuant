#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Market Data Update Script (Full Pipeline)
Runs all data collection and analysis scripts in sequence
Based on PART2/PART3 Blueprint specifications
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime
import argparse

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Pipeline Definition (PART2/PART3 Blueprint)
# ============================================================

# Phase 1: Data Collection
PHASE_1_DATA = [
    ('create_us_daily_prices.py', 'Price Data Collection', 600),
]

# Phase 2: Basic Analysis
PHASE_2_ANALYSIS = [
    ('analyze_volume.py', 'Volume Analysis', 300),
    ('analyze_13f.py', '13F Institutional Analysis', 600),  # Increased timeout
    ('analyze_etf_flows.py', 'ETF Flow Analysis', 300),
]

# Phase 3: Screening & Additional Analysis (PART2)
PHASE_3_SCREENING = [
    ('smart_money_screener_v2.py', 'Smart Money Screening', 600),
    ('sector_heatmap.py', 'Sector Heatmap', 300),
    ('options_flow.py', 'Options Flow Analysis', 300),
    ('insider_tracker.py', 'Insider Trading Tracker', 300),
    ('portfolio_risk.py', 'Portfolio Risk Analysis', 300),
]

# Phase 4: AI Analysis (PART3) - Requires API Keys
PHASE_4_AI = [
    ('ai_summary_generator.py', 'AI Stock Summaries', 900),
    ('final_report_generator.py', 'Final Top 10 Report', 60),
    ('macro_analyzer.py', 'Macro Economic Analysis', 300),
    ('economic_calendar.py', 'Economic Calendar', 300),
]

# All output files to check
OUTPUT_FILES = [
    # Phase 1
    'us_daily_prices.csv',
    'us_stocks_list.csv',
    # Phase 2
    'us_volume_analysis.csv',
    'us_13f_holdings.csv',
    'us_etf_flows.csv',
    'etf_flow_analysis.json',
    # Phase 3
    'smart_money_picks_v2.csv',
    'sector_heatmap.json',
    'options_flow.json',
    'insider_moves.json',
    'portfolio_risk.json',
    # Phase 4
    'ai_summaries.json',
    'final_top10_report.json',
    'smart_money_current.json',
    'macro_analysis.json',
    'weekly_calendar.json',
]


def run_script(script_name: str, description: str, timeout: int = 600, args: list = None) -> bool:
    """Run a Python script and return success status"""
    try:
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)
        
        logger.info(f"▶️  Running {description}...")
        start = time.time()
        
        # Set UTF-8 encoding for subprocess to handle emoji characters on Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed ({elapsed:.1f}s)")
            return True
        else:
            logger.error(f"❌ {description} failed:")
            if result.stderr:
                # Only show last 500 chars of error
                error_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                logger.error(error_msg)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {description} timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        return False


def run_phase(phase_name: str, scripts: list, skip_ai: bool = False, script_args: dict = None) -> dict:
    """Run a phase of scripts"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 {phase_name}")
    logger.info('='*60)
    
    results = {}
    script_args = script_args or {}
    
    for script_name, description, timeout in scripts:
        # Skip AI scripts if --quick flag is set
        if skip_ai and 'ai' in script_name.lower():
            logger.info(f"⏭️  Skipping {description} (--quick mode)")
            results[script_name] = None
            continue
        
        args = script_args.get(script_name, [])
        results[script_name] = run_script(script_name, description, timeout, args)
    
    return results


def check_output_files(output_dir: str) -> None:
    """Check and report on output files"""
    logger.info("\n📁 Output Files Status:")
    
    for filename in OUTPUT_FILES:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            
            # Warn if file is suspiciously small
            if size < 100 and filename.endswith('.csv'):
                logger.warning(f"   ⚠️  {filename} ({size_str}) - may be empty!")
            elif size < 50 and filename.endswith('.json'):
                logger.warning(f"   ⚠️  {filename} ({size_str}) - may be empty!")
            else:
                logger.info(f"   ✅ {filename} ({size_str})")
        else:
            logger.info(f"   ⚠️  {filename} (not created)")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='US Market Full Data Pipeline Update',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_all.py                    # Full pipeline
  python update_all.py --quick            # Skip AI analysis
  python update_all.py --prices-only      # Only price data
  python update_all.py --analysis-only    # Skip data collection
  python update_all.py --screening-only   # Only run screening scripts
  python update_all.py --ai-only          # Only run AI analysis
        """
    )
    parser.add_argument('--quick', action='store_true', 
                       help='Quick update (skip AI analysis)')
    parser.add_argument('--prices-only', action='store_true',
                       help='Only update price data')
    parser.add_argument('--full', action='store_true',
                       help='Full refresh of price data')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Skip data collection, run analysis only')
    parser.add_argument('--screening-only', action='store_true',
                       help='Only run screening scripts (Phase 3)')
    parser.add_argument('--ai-only', action='store_true',
                       help='Only run AI analysis scripts (Phase 4)')
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"🚀 US Market Full Pipeline Update")
    logger.info(f"   Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    all_results = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine which phases to run
    run_data = not (args.analysis_only or args.screening_only or args.ai_only)
    run_analysis = not (args.prices_only or args.screening_only or args.ai_only)
    run_screening = not (args.prices_only or args.ai_only) or args.screening_only
    run_ai = not (args.prices_only or args.quick or args.screening_only) or args.ai_only
    
    # Phase 1: Data Collection
    if run_data:
        price_args = {'create_us_daily_prices.py': ['--full'] if args.full else []}
        results = run_phase("Phase 1: Data Collection", PHASE_1_DATA, script_args=price_args)
        all_results.update(results)
        
        if args.prices_only:
            logger.info("\n🎉 Price-only update complete!")
            check_output_files(script_dir)
            return
    
    # Phase 2: Basic Analysis
    if run_analysis:
        results = run_phase("Phase 2: Basic Analysis", PHASE_2_ANALYSIS)
        all_results.update(results)
    
    # Phase 3: Screening
    if run_screening:
        results = run_phase("Phase 3: Screening & Analysis", PHASE_3_SCREENING)
        all_results.update(results)
    
    # Phase 4: AI Analysis
    if run_ai:
        results = run_phase("Phase 4: AI Analysis", PHASE_4_AI, skip_ai=args.quick)
        all_results.update(results)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 Pipeline Summary")
    logger.info("=" * 60)
    
    success_count = sum(1 for v in all_results.values() if v is True)
    fail_count = sum(1 for v in all_results.values() if v is False)
    skip_count = sum(1 for v in all_results.values() if v is None)
    
    for script, success in all_results.items():
        if success is True:
            logger.info(f"   ✅ {script}")
        elif success is False:
            logger.info(f"   ❌ {script}")
        else:
            logger.info(f"   ⏭️  {script} (skipped)")
    
    logger.info(f"\n📊 Results: {success_count} success, {fail_count} failed, {skip_count} skipped")
    logger.info(f"⏱️  Total time: {duration}")
    
    # Check output files
    check_output_files(script_dir)
    
    # Final status
    if fail_count == 0:
        logger.info("\n🎉 Pipeline completed successfully!")
    else:
        logger.warning(f"\n⚠️  {fail_count} script(s) failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
