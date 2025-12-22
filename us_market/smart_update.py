#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Update Script - Intelligent Data Refreshing
Checks file modification times AND content validity before running updates.
"""

import os
import sys
import subprocess
import logging
import time
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# Minimum file size (bytes) - files smaller than this are considered empty/invalid
MIN_FILE_SIZE = 100  # 100 bytes minimum


def get_file_age_hours(filepath: Path) -> float:
    """Get file age in hours. Returns infinity if file doesn't exist."""
    if not filepath.exists():
        return float('inf')
    mtime = os.path.getmtime(filepath)
    age_seconds = time.time() - mtime
    return age_seconds / 3600


def check_csv_validity(filepath: Path, required_columns: list = None, min_data_rows: int = 1) -> dict:
    """
    Check if a CSV file has valid content.
    
    Returns:
        dict with keys: valid (bool), reason (str), rows (int), columns (list)
    """
    result = {'valid': False, 'reason': '', 'rows': 0, 'columns': []}
    
    if not filepath.exists():
        result['reason'] = 'file_missing'
        return result
    
    if filepath.stat().st_size < MIN_FILE_SIZE:
        result['reason'] = 'file_too_small'
        return result
    
    try:
        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            result['reason'] = 'empty_file'
            return result
        
        result['columns'] = rows[0] if rows else []
        result['rows'] = len(rows) - 1  # Exclude header
        
        if result['rows'] < min_data_rows:
            result['reason'] = 'insufficient_data'
            return result
        
        if required_columns:
            missing = [c for c in required_columns if c not in result['columns']]
            if missing:
                result['reason'] = f'missing_columns: {missing}'
                return result
        
        result['valid'] = True
        result['reason'] = 'valid'
        return result
        
    except Exception as e:
        result['reason'] = f'parse_error: {e}'
        return result


def check_json_validity(filepath: Path, required_keys: list = None, min_size_bytes: int = MIN_FILE_SIZE) -> dict:
    """
    Check if a JSON file has valid content.
    
    Returns:
        dict with keys: valid (bool), reason (str), keys (list)
    """
    result = {'valid': False, 'reason': '', 'keys': []}
    
    if not filepath.exists():
        result['reason'] = 'file_missing'
        return result
    
    if filepath.stat().st_size < min_size_bytes:
        result['reason'] = 'file_too_small'
        return result
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            result['reason'] = 'empty_json'
            return result
        
        if isinstance(data, dict):
            result['keys'] = list(data.keys())
            
            if required_keys:
                missing = [k for k in required_keys if k not in data]
                if missing:
                    result['reason'] = f'missing_keys: {missing}'
                    return result
        
        result['valid'] = True
        result['reason'] = 'valid'
        return result
        
    except Exception as e:
        result['reason'] = f'parse_error: {e}'
        return result


def is_update_needed(filepath: Path, threshold_hours: float, file_type: str, 
                     required_columns: list = None, required_keys: list = None) -> tuple:
    """
    Check if a file needs updating based on age AND validity.
    
    Returns:
        tuple: (needs_update: bool, reason: str)
    """
    # Check if file exists
    if not filepath.exists():
        return True, 'file_missing'
    
    # Check file size
    if filepath.stat().st_size < MIN_FILE_SIZE:
        return True, 'file_too_small'
    
    # Check content validity
    if file_type == 'csv':
        validity = check_csv_validity(filepath, required_columns)
        if not validity['valid']:
            return True, f"invalid_content: {validity['reason']}"
    elif file_type == 'json':
        validity = check_json_validity(filepath, required_keys)
        if not validity['valid']:
            return True, f"invalid_content: {validity['reason']}"
    
    # Check file age
    age_hours = get_file_age_hours(filepath)
    if age_hours > threshold_hours:
        return True, f'stale: {age_hours:.1f}h old (threshold: {threshold_hours}h)'
    
    return False, f'fresh: {age_hours:.1f}h old, valid content'


# Data file configurations
DATA_FILES = {
    'prices': {
        'path': SCRIPT_DIR / 'us_daily_prices.csv',
        'type': 'csv',
        'threshold_hours': 1,
        'required_columns': ['ticker', 'date', 'close'],
        'command': ['--prices-only'],
        'description': 'Price Data',
    },
    'screening': {
        'path': SCRIPT_DIR / 'smart_money_picks_v2.csv',
        'type': 'csv',
        'threshold_hours': 4,
        'required_columns': ['ticker', 'composite_score', 'grade'],
        'command': ['--screening-only'],
        'description': 'Smart Money Screening',
    },
    'ai': {
        'path': SCRIPT_DIR / 'ai_summaries.json',
        'type': 'json',
        'threshold_hours': 12,
        'required_keys': None,  # Just check it's not empty
        'command': ['--ai-only'],
        'description': 'AI Summaries',
    },
}


def check_all_files() -> dict:
    """Check all data files and report status."""
    status = {}
    
    for name, config in DATA_FILES.items():
        filepath = config['path']
        needs_update, reason = is_update_needed(
            filepath,
            config['threshold_hours'],
            config['type'],
            config.get('required_columns'),
            config.get('required_keys')
        )
        
        status[name] = {
            'needs_update': needs_update,
            'reason': reason,
            'path': str(filepath),
            'exists': filepath.exists(),
            'size': filepath.stat().st_size if filepath.exists() else 0,
        }
        
        if needs_update:
            logger.info(f"📁 {config['description']}: NEEDS UPDATE - {reason}")
        else:
            logger.info(f"✅ {config['description']}: Up to date - {reason}")
    
    return status


def run_update(data_types: list) -> bool:
    """Run update_all.py with appropriate flags."""
    if not data_types:
        logger.info("🎉 All data is up to date and valid. Skipping update.")
        return True
    
    # Build command based on what needs updating
    update_script = SCRIPT_DIR / 'update_all.py'
    
    # If multiple types need update, run them in sequence
    for dt in data_types:
        if dt not in DATA_FILES:
            continue
            
        config = DATA_FILES[dt]
        cmd = [sys.executable, str(update_script)] + config['command']
        
        logger.info(f"🚀 Updating {config['description']}...")
        logger.info(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(SCRIPT_DIR),
                capture_output=False,
                text=True,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            
            if result.returncode != 0:
                logger.error(f"❌ {config['description']} update failed")
                return False
            else:
                logger.info(f"✅ {config['description']} updated successfully")
                
        except Exception as e:
            logger.error(f"❌ Error updating {config['description']}: {e}")
            return False
    
    return True


def smart_update(force: bool = False) -> bool:
    """
    Main smart update function.
    
    Args:
        force: If True, run full update regardless of file status
        
    Returns:
        bool: True if update was successful or not needed
    """
    logger.info("=" * 60)
    logger.info("🔍 Smart Update - Checking data freshness and validity...")
    logger.info("=" * 60)
    
    if force:
        logger.info("⚡ Force mode enabled - running full update")
        return run_update(list(DATA_FILES.keys()))
    
    # Check all files
    status = check_all_files()
    
    # Determine what needs updating
    needs_update = [name for name, info in status.items() if info['needs_update']]
    
    if not needs_update:
        logger.info("\n🎉 All data is fresh and valid!")
        return True
    
    logger.info(f"\n📋 Files needing update: {needs_update}")
    return run_update(needs_update)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Update - Intelligent data refreshing')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Force full update regardless of file status')
    parser.add_argument('--check-only', '-c', action='store_true',
                        help='Only check file status, do not update')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed file information')
    
    args = parser.parse_args()
    
    if args.check_only:
        logger.info("📊 Checking data status (dry run)...")
        status = check_all_files()
        
        if args.verbose:
            for name, info in status.items():
                print(f"\n{name}:")
                for k, v in info.items():
                    print(f"  {k}: {v}")
        
        needs_update = [n for n, i in status.items() if i['needs_update']]
        if needs_update:
            logger.info(f"\n📋 Files needing update: {needs_update}")
            return 1
        else:
            logger.info("\n✅ All data is fresh and valid")
            return 0
    
    success = smart_update(force=args.force)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
