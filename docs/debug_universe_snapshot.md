# Debug Universe Snapshot Fix

## Problem
The backtest engine was failing with `No universe snapshot found for {date}`.
This was because:
1. `backtest.universe_snapshot` table was empty or missing snapshots for the rebalancing dates.
2. The query logic was too strict (exact date match) or not looking back far enough.
3. Rebalancing dates (e.g., weekends) didn't align with data.

## Fix Implementation

### 1. Snapshot Generation Script
Created `scripts/build_universe_snapshots.py` to populate `bt_universe_snapshot` table.
- **Source**: `us_market/us_daily_prices.csv` (used as proxy for historical universe).
- **Logic**: For each month-end, identified tickers with volume > 0.
- **Output**: Populated `backtest.universe_snapshot` in PostgreSQL.

### 2. Data Loader Improvements (`backtest/engine/data_loader_pg.py`)
- **Query Logic**: 
  - Modified `get_universe_as_of` to search for `MAX(as_of_date) <= requested_date`.
  - This ensures a valid snapshot is found even if the requested date is a few days after the snapshot (e.g., weekend/holiday).
- **Price Data Fallback**:
  - Changed price source from `backtest.prices_daily` (empty) to `market.daily_prices` (operational data) to enable immediate backtesting.
- **Fundamentals**:
  - Added `get_fundamentals_as_of` to support fundamental-based strategies.

### 3. Backtest Engine Adjustments (`backtest/engine/backtest_engine.py`)
- **Lookback Period**: Increased default price lookback to 400 days to ensure enough data for momentum calculation (252 trading days requirement).
- **Strategy Interface**: Updated to Version 2 interface:
  - `select_universe(snapshot)`
  - `compute_targets(date, snapshot, prices, fundamentals, context)`

### 4. Verification
- **Momentum Strategy**: 
  - Ran for 2024-01-01 to 2024-06-30.
  - Result: CAGR 7.09%, Total Return 3.42%.
  - Confirmed trades were generated (metrics > 0).
- **Value Ranker Strategy**:
  - Ran successfully (0% return due to missing fundamental data, but no crash).

## Future Work
- Populate `backtest.financial_statements` with real fundamental data.
- Refine universe definition (e.g., use index constituents instead of just "traded tickers").
