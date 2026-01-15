# FMP Migration Guide (yfinance -> Financial Modeling Prep)

## Goal
Replace Yahoo/yfinance-based market data fetches with the Financial Modeling Prep (FMP) API, while keeping current data shapes and downstream CSV/JSON contracts intact.

## Current Finance Data Fetch Inventory (FMP-first)
| Location | Data Used | Current Source |
| --- | --- | --- |
| `flask_app.py` | sector lookup, batch quotes, charts, macro live updates | FMP (`utils/fmp_client`) |
| `us_market/create_us_daily_prices.py` | daily OHLCV per ticker | FMP |
| `us_market/smart_money_screener_v2.py` | technicals, fundamentals, analyst data | FMP |
| `us_market/sector_heatmap.py` | sector/ETF performance | FMP |
| `us_market/analyze_etf_flows.py` | ETF OHLCV + optional Gemini insight | FMP |
| `us_market/macro_analyzer.py` | macro indicators + 52w range | FMP |
| `us_market/portfolio_risk.py` | price history for risk stats | FMP |
| `us_market/insider_tracker.py` | insider transactions | FMP |
| `us_market/economic_calendar.py` | economic calendar | FMP |
| `us_market/options_flow.py` | options chain & IV | yfinance (only remaining Yahoo dependency) |

## FMP Official Docs (Auth + Key Endpoints)
Docs home:
- https://site.financialmodelingprep.com/developer/docs/

Auth note from docs:
- Add `?apikey=YOUR_API_KEY` to every request.
- If the endpoint already has query params, use `&apikey=YOUR_API_KEY`.

Common endpoints referenced in FMP docs (examples):
- Quotes (single or batch):
  - https://financialmodelingprep.com/api/v3/quote/AAPL
  - https://financialmodelingprep.com/api/v3/quote/AAPL,MSFT
  - https://financialmodelingprep.com/stable/quote?symbol=AAPL
  - https://financialmodelingprep.com/stable/batch-quote?symbols=AAPL,MSFT
- Company profile:
  - https://financialmodelingprep.com/api/v3/profile/AAPL
  - https://financialmodelingprep.com/stable/profile?symbol=AAPL
- Historical daily prices:
  - https://financialmodelingprep.com/api/v3/historical-price-full/AAPL
- Intraday candles:
  - https://financialmodelingprep.com/stable/historical-chart/1min?symbol=AAPL
  - https://financialmodelingprep.com/api/v3/historical-chart/5min/AAPL?from=2023-08-10&to=2023-09-10
- Ratios / key metrics:
  - https://financialmodelingprep.com/api/v3/key-metrics-ttm/AAPL
  - https://financialmodelingprep.com/api/v3/ratios-ttm/AAPL
  - https://financialmodelingprep.com/stable/key-metrics-ttm?symbol=AAPL
  - https://financialmodelingprep.com/stable/ratios-ttm?symbol=AAPL
- Analyst estimates / ratings / price targets:
  - https://financialmodelingprep.com/api/v3/analyst-estimates/AAPL
  - https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/AAPL
  - https://financialmodelingprep.com/stable/ratings-snapshot?symbol=AAPL
  - https://financialmodelingprep.com/stable/price-target-consensus?symbol=AAPL
- Insider trading:
  - https://financialmodelingprep.com/api/v4/insider-trading?symbol=AAPL&page=0
  - https://financialmodelingprep.com/stable/insider-trading/search?page=0&limit=100
- Economic calendar:
  - https://financialmodelingprep.com/stable/economic-calendar
  - https://financialmodelingprep.com/api/v3/economic_calendar
- Treasury rates:
  - https://financialmodelingprep.com/stable/treasury-rates
  - https://financialmodelingprep.com/api/v4/treasury?from=2023-08-10&to=2023-10-10
- Symbol lists (for mapping Yahoo symbols to FMP):
  - https://financialmodelingprep.com/stable/index-list
  - https://financialmodelingprep.com/stable/forex-list
  - https://financialmodelingprep.com/stable/commodities-list
  - https://financialmodelingprep.com/stable/cryptocurrency-list

## Migration Status (Summary)
- FMP client (`utils/fmp_client.py`) is the default path for quotes, profiles, historical prices, calendar, and insider data.
- Remaining Yahoo/yfinance usage is limited to `us_market/options_flow.py` for options chain data.
- CSV/JSON outputs are still generated for export and fallback, while PostgreSQL/SQLite are used where available.

## Remaining Work (If You Want to Fully Remove yfinance)
1. Replace `us_market/options_flow.py` with a dedicated options data source (FMP does not provide full options chains).
2. Remove yfinance dependency once options flow is migrated.

## Symbol Mapping Notes (Yahoo -> FMP)
Confirm symbol formats using FMP list endpoints:
- Indices: `index-list` (examples in docs: `^GSPC`).
- Crypto: `BTCUSD` instead of `BTC-USD`.
- Commodities: `GCUSD`, `CLUSD` instead of `GC=F`, `CL=F`.

## Testing/Docs Update Checklist
- Add FMP mock coverage for critical API paths (quotes, historical prices, calendar).
- When options flow is migrated, remove yfinance dependency and update docs accordingly.
