#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate us_sectors.csv from SECTOR_MAP in flask_app.py

This script extracts sector mapping data and writes it to a CSV file
that can be used by both sector_heatmap.py and flask_app.py.
"""

import csv
import os
import sys
from pathlib import Path

# Add parent directory to path for flask_app import
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from flask_app import SECTOR_MAP

US_DIR = Path(os.getenv("DATA_DIR", SCRIPT_DIR)).resolve()


def generate_sectors_csv(output_path: Path = None) -> int:
    """
    Generate us_sectors.csv from SECTOR_MAP.
    
    Args:
        output_path: Output file path. Defaults to US_DIR / 'us_sectors.csv'
    
    Returns:
        Number of rows written
    """
    if output_path is None:
        output_path = US_DIR / 'us_sectors.csv'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by sector first, then by ticker for consistent output
    sorted_items = sorted(SECTOR_MAP.items(), key=lambda x: (x[1], x[0]))
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ticker', 'sector'])
        for ticker, sector in sorted_items:
            writer.writerow([ticker, sector])
    
    # Summary statistics
    sector_counts = {}
    for ticker, sector in SECTOR_MAP.items():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    print(f"✅ Generated {output_path}")
    print(f"   Total tickers: {len(SECTOR_MAP)}")
    print(f"   Sectors: {len(sector_counts)}")
    print("\n📊 Sector breakdown:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"   {sector}: {count} stocks")
    
    return len(SECTOR_MAP)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate us_sectors.csv from SECTOR_MAP')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: us_sectors.csv in DATA_DIR)')
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    generate_sectors_csv(output_path)


if __name__ == "__main__":
    main()
