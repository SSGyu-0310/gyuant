#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL schema initializer (idempotent).
Creates schemas and tables from backtest/db_schema_pg.py.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env

load_env()

from backtest.db_schema_pg import init_db


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Initialize PostgreSQL schemas/tables")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables before creating")
    args = parser.parse_args()

    init_db(drop_existing=args.drop)
    print("PostgreSQL schema initialization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
