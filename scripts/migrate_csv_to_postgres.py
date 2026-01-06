#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to PostgreSQL migration wrapper.
Delegates to backtest/migrate_csv_to_pg.py.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env

load_env()

from backtest.migrate_csv_to_pg import main as migrate_main


if __name__ == "__main__":
    sys.exit(migrate_main())
