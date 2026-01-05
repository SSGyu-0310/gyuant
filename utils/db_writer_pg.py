#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance PostgreSQL DB Writer
Bulk insert and upsert operations optimized for large financial datasets

Features:
1. Singleton pattern for connection management
2. Bulk insert using psycopg executemany with batching
3. Upsert (ON CONFLICT) for handling duplicates
4. Transaction management with rollback on error
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date

import psycopg
from psycopg import sql
from psycopg import Connection as PgConnection

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PostgresDBWriter:
    """
    Singleton DB Writer for PostgreSQL with bulk operations.
    """
    _instance: Optional["PostgresDBWriter"] = None
    _connection: Optional[PgConnection] = None
    
    def __new__(cls) -> "PostgresDBWriter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._batch_size = int(os.getenv("PG_BATCH_SIZE", "1000"))
    
    @property
    def connection(self) -> PgConnection:
        """Get or create database connection."""
        if self._connection is None or self._connection.closed:
            self._connection = self._create_connection()
        return self._connection
    
    def _create_connection(self) -> PgConnection:
        """Create a new PostgreSQL connection."""
        host = os.getenv("PG_HOST", "localhost")
        port = os.getenv("PG_PORT", "5432")
        database = os.getenv("PG_DATABASE", "gyuant_market")
        user = os.getenv("PG_USER", "postgres")
        password = os.getenv("PG_PASSWORD", "")
        
        conn = psycopg.connect(
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password,
        )
        conn.autocommit = False
        logger.info(f"PostgreSQL connected: {host}:{port}/{database}")
        return conn
    
    def close(self):
        """Close the database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
            logger.info("PostgreSQL connection closed")
    
    # =========================================================================
    # Ticker Operations
    # =========================================================================
    
    def save_tickers(
        self,
        records: List[Dict[str, Any]],
        on_conflict: str = "update"  # "update", "ignore", or "error"
    ) -> int:
        """
        Bulk upsert tickers to the database.
        
        Args:
            records: List of ticker dicts with keys:
                symbol, name, sector, industry, exchange, market_cap,
                is_active, date_added, date_removed, source
            on_conflict: How to handle duplicates
        
        Returns:
            Number of rows affected
        """
        if not records:
            return 0
        
        if on_conflict == "update":
            query = """
                INSERT INTO tickers (
                    symbol, name, sector, industry, exchange, market_cap,
                    is_active, date_added, date_removed, source, updated_at
                ) VALUES (
                    %(symbol)s, %(name)s, %(sector)s, %(industry)s, %(exchange)s,
                    %(market_cap)s, %(is_active)s, %(date_added)s, %(date_removed)s,
                    %(source)s, NOW()
                )
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    exchange = EXCLUDED.exchange,
                    market_cap = EXCLUDED.market_cap,
                    is_active = EXCLUDED.is_active,
                    date_added = COALESCE(EXCLUDED.date_added, tickers.date_added),
                    date_removed = EXCLUDED.date_removed,
                    source = EXCLUDED.source,
                    updated_at = NOW()
            """
        elif on_conflict == "ignore":
            query = """
                INSERT INTO tickers (
                    symbol, name, sector, industry, exchange, market_cap,
                    is_active, date_added, date_removed, source
                ) VALUES (
                    %(symbol)s, %(name)s, %(sector)s, %(industry)s, %(exchange)s,
                    %(market_cap)s, %(is_active)s, %(date_added)s, %(date_removed)s,
                    %(source)s
                )
                ON CONFLICT (symbol) DO NOTHING
            """
        else:
            query = """
                INSERT INTO tickers (
                    symbol, name, sector, industry, exchange, market_cap,
                    is_active, date_added, date_removed, source
                ) VALUES (
                    %(symbol)s, %(name)s, %(sector)s, %(industry)s, %(exchange)s,
                    %(market_cap)s, %(is_active)s, %(date_added)s, %(date_removed)s,
                    %(source)s
                )
            """
        
        # Normalize records
        normalized = []
        for r in records:
            normalized.append({
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "sector": r.get("sector"),
                "industry": r.get("industry"),
                "exchange": r.get("exchange"),
                "market_cap": r.get("market_cap") or r.get("marketCap"),
                "is_active": r.get("is_active", True),
                "date_added": r.get("date_added") or r.get("dateAdded"),
                "date_removed": r.get("date_removed") or r.get("dateRemoved"),
                "source": r.get("source", "FMP"),
            })
        
        return self._execute_batch(query, normalized)
    
    # =========================================================================
    # Price Operations
    # =========================================================================
    
    def save_bulk_prices(
        self,
        records: List[Dict[str, Any]],
        on_conflict: str = "update"
    ) -> int:
        """
        Bulk upsert daily prices to the database.
        
        Args:
            records: List of price dicts with keys:
                symbol, date, open, high, low, close, adj_close, volume,
                change, change_pct, source
            on_conflict: "update" or "ignore"
        
        Returns:
            Number of rows affected
        """
        if not records:
            return 0
        
        if on_conflict == "update":
            query = """
                INSERT INTO daily_prices (
                    symbol, date, open, high, low, close, adj_close, volume,
                    change, change_pct, source, ingested_at
                ) VALUES (
                    %(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s,
                    %(adj_close)s, %(volume)s, %(change)s, %(change_pct)s,
                    %(source)s, NOW()
                )
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume,
                    change = EXCLUDED.change,
                    change_pct = EXCLUDED.change_pct,
                    source = EXCLUDED.source,
                    ingested_at = NOW()
            """
        else:
            query = """
                INSERT INTO daily_prices (
                    symbol, date, open, high, low, close, adj_close, volume,
                    change, change_pct, source
                ) VALUES (
                    %(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s,
                    %(adj_close)s, %(volume)s, %(change)s, %(change_pct)s, %(source)s
                )
                ON CONFLICT (symbol, date) DO NOTHING
            """
        
        # Normalize records
        normalized = []
        for r in records:
            # Handle date conversion
            d = r.get("date")
            if isinstance(d, str):
                d = datetime.strptime(d, "%Y-%m-%d").date() if d else None
            
            normalized.append({
                "symbol": r.get("symbol") or r.get("ticker"),
                "date": d,
                "open": r.get("open"),
                "high": r.get("high"),
                "low": r.get("low"),
                "close": r.get("close"),
                "adj_close": r.get("adj_close") or r.get("adjClose"),
                "volume": r.get("volume"),
                "change": r.get("change"),
                "change_pct": r.get("change_pct") or r.get("changePercent"),
                "source": r.get("source", "FMP"),
            })
        
        return self._execute_batch(query, normalized)
    
    # =========================================================================
    # Financial Statement Operations
    # =========================================================================
    
    def save_bulk_financials(
        self,
        records: List[Dict[str, Any]],
        on_conflict: str = "update"
    ) -> int:
        """
        Bulk upsert financial statements with PIT support.
        
        CRITICAL: filing_date must be provided for PIT queries to work correctly.
        
        Args:
            records: List of financial dicts with keys including:
                symbol, period_end_date, statement_type, filing_date,
                fiscal_year, fiscal_quarter, revenue, net_income, etc.
            on_conflict: "update" or "ignore"
        
        Returns:
            Number of rows affected
        """
        if not records:
            return 0
        
        if on_conflict == "update":
            query = """
                INSERT INTO financial_statements (
                    symbol, period_end_date, statement_type, filing_date,
                    fiscal_year, fiscal_quarter, period_type,
                    revenue, gross_profit, operating_income, net_income,
                    eps, eps_diluted, total_assets, total_liabilities, total_equity,
                    cash_and_equivalents, total_debt, operating_cash_flow,
                    capex, free_cash_flow, roe, roa, debt_to_equity, current_ratio,
                    source, ingested_at
                ) VALUES (
                    %(symbol)s, %(period_end_date)s, %(statement_type)s, %(filing_date)s,
                    %(fiscal_year)s, %(fiscal_quarter)s, %(period_type)s,
                    %(revenue)s, %(gross_profit)s, %(operating_income)s, %(net_income)s,
                    %(eps)s, %(eps_diluted)s, %(total_assets)s, %(total_liabilities)s,
                    %(total_equity)s, %(cash_and_equivalents)s, %(total_debt)s,
                    %(operating_cash_flow)s, %(capex)s, %(free_cash_flow)s,
                    %(roe)s, %(roa)s, %(debt_to_equity)s, %(current_ratio)s,
                    %(source)s, NOW()
                )
                ON CONFLICT (symbol, period_end_date, statement_type) DO UPDATE SET
                    filing_date = EXCLUDED.filing_date,
                    fiscal_year = EXCLUDED.fiscal_year,
                    fiscal_quarter = EXCLUDED.fiscal_quarter,
                    revenue = EXCLUDED.revenue,
                    gross_profit = EXCLUDED.gross_profit,
                    operating_income = EXCLUDED.operating_income,
                    net_income = EXCLUDED.net_income,
                    eps = EXCLUDED.eps,
                    eps_diluted = EXCLUDED.eps_diluted,
                    total_assets = EXCLUDED.total_assets,
                    total_liabilities = EXCLUDED.total_liabilities,
                    total_equity = EXCLUDED.total_equity,
                    cash_and_equivalents = EXCLUDED.cash_and_equivalents,
                    total_debt = EXCLUDED.total_debt,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    capex = EXCLUDED.capex,
                    free_cash_flow = EXCLUDED.free_cash_flow,
                    roe = EXCLUDED.roe,
                    roa = EXCLUDED.roa,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    current_ratio = EXCLUDED.current_ratio,
                    source = EXCLUDED.source,
                    ingested_at = NOW()
            """
        else:
            query = """
                INSERT INTO financial_statements (
                    symbol, period_end_date, statement_type, filing_date,
                    fiscal_year, fiscal_quarter, period_type,
                    revenue, gross_profit, operating_income, net_income,
                    eps, eps_diluted, total_assets, total_liabilities, total_equity,
                    cash_and_equivalents, total_debt, operating_cash_flow,
                    capex, free_cash_flow, roe, roa, debt_to_equity, current_ratio,
                    source
                ) VALUES (
                    %(symbol)s, %(period_end_date)s, %(statement_type)s, %(filing_date)s,
                    %(fiscal_year)s, %(fiscal_quarter)s, %(period_type)s,
                    %(revenue)s, %(gross_profit)s, %(operating_income)s, %(net_income)s,
                    %(eps)s, %(eps_diluted)s, %(total_assets)s, %(total_liabilities)s,
                    %(total_equity)s, %(cash_and_equivalents)s, %(total_debt)s,
                    %(operating_cash_flow)s, %(capex)s, %(free_cash_flow)s,
                    %(roe)s, %(roa)s, %(debt_to_equity)s, %(current_ratio)s,
                    %(source)s
                )
                ON CONFLICT (symbol, period_end_date, statement_type) DO NOTHING
            """
        
        # Normalize records
        normalized = []
        for r in records:
            normalized.append({
                "symbol": r.get("symbol"),
                "period_end_date": r.get("period_end_date") or r.get("date"),
                "statement_type": r.get("statement_type", "income"),
                "filing_date": r.get("filing_date") or r.get("fillingDate") or r.get("acceptedDate"),
                "fiscal_year": r.get("fiscal_year") or r.get("calendarYear"),
                "fiscal_quarter": r.get("fiscal_quarter") or r.get("period"),
                "period_type": r.get("period_type", "quarterly"),
                "revenue": r.get("revenue"),
                "gross_profit": r.get("gross_profit") or r.get("grossProfit"),
                "operating_income": r.get("operating_income") or r.get("operatingIncome"),
                "net_income": r.get("net_income") or r.get("netIncome"),
                "eps": r.get("eps"),
                "eps_diluted": r.get("eps_diluted") or r.get("epsdiluted"),
                "total_assets": r.get("total_assets") or r.get("totalAssets"),
                "total_liabilities": r.get("total_liabilities") or r.get("totalLiabilities"),
                "total_equity": r.get("total_equity") or r.get("totalStockholdersEquity"),
                "cash_and_equivalents": r.get("cash_and_equivalents") or r.get("cashAndCashEquivalents"),
                "total_debt": r.get("total_debt") or r.get("totalDebt"),
                "operating_cash_flow": r.get("operating_cash_flow") or r.get("operatingCashFlow"),
                "capex": r.get("capex") or r.get("capitalExpenditure"),
                "free_cash_flow": r.get("free_cash_flow") or r.get("freeCashFlow"),
                "roe": r.get("roe") or r.get("returnOnEquity"),
                "roa": r.get("roa") or r.get("returnOnAssets"),
                "debt_to_equity": r.get("debt_to_equity") or r.get("debtEquityRatio"),
                "current_ratio": r.get("current_ratio") or r.get("currentRatio"),
                "source": r.get("source", "FMP"),
            })
        
        return self._execute_batch(query, normalized)
    
    # =========================================================================
    # Universe Snapshot Operations
    # =========================================================================
    
    def save_universe_snapshot(
        self,
        as_of_date: date,
        records: List[Dict[str, Any]],
        market: str = "S&P500"
    ) -> int:
        """
        Save a universe snapshot for a specific date.
        
        Args:
            as_of_date: Snapshot date
            records: List of dicts with symbol, name, sector, weight
            market: Market/index name
        
        Returns:
            Number of rows affected
        """
        if not records:
            return 0
        
        query = """
            INSERT INTO universe_snapshots (
                as_of_date, symbol, name, sector, market, weight, source, ingested_at
            ) VALUES (
                %(as_of_date)s, %(symbol)s, %(name)s, %(sector)s, %(market)s,
                %(weight)s, %(source)s, NOW()
            )
            ON CONFLICT (as_of_date, symbol) DO UPDATE SET
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                market = EXCLUDED.market,
                weight = EXCLUDED.weight,
                ingested_at = NOW()
        """
        
        normalized = []
        for r in records:
            normalized.append({
                "as_of_date": as_of_date,
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "sector": r.get("sector"),
                "market": market,
                "weight": r.get("weight"),
                "source": r.get("source", "FMP"),
            })
        
        return self._execute_batch(query, normalized)
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _execute_batch(
        self,
        query: str,
        records: List[Dict[str, Any]]
    ) -> int:
        """
        Execute a batch insert/upsert operation.
        
        Uses psycopg executemany in batches for performance.
        """
        if not records:
            return 0
        
        try:
            with self.connection.cursor() as cursor:
                if self._batch_size <= 0:
                    cursor.executemany(query, records)
                else:
                    for i in range(0, len(records), self._batch_size):
                        batch = records[i:i + self._batch_size]
                        cursor.executemany(query, batch)
            self.connection.commit()
            affected = len(records)
            logger.debug(f"Batch executed: {affected} records")
            return affected
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Batch execution failed: {e}")
            raise
    
    def execute_raw(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a raw SQL query."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return cursor.fetchall()
            self.connection.commit()
            return None
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Raw query failed: {e}")
            raise
    
    def get_row_count(self, table: str) -> int:
        """Get the number of rows in a table."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception:
            return 0


# Convenience function
def get_db_writer() -> PostgresDBWriter:
    """Get the singleton DB writer instance."""
    return PostgresDBWriter()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🐘 PostgreSQL DB Writer Test")
    print("=" * 50)
    
    try:
        writer = get_db_writer()
        
        # Test connection
        print("✅ Connection established")
        
        # Test row counts
        tables = ["tickers", "daily_prices", "financial_statements", "universe_snapshots"]
        for table in tables:
            try:
                count = writer.get_row_count(table)
                print(f"   {table}: {count} rows")
            except Exception as e:
                print(f"   {table}: Error - {e}")
        
        writer.close()
        print("✅ Connection closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
