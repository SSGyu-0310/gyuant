# FMP Migration Guide (yfinance -> Financial Modeling Prep)

## Goal
Replace Yahoo/yfinance-based market data fetches with the Financial Modeling Prep (FMP) API, while keeping current data shapes and downstream CSV/JSON contracts intact.

## Current Finance Data Fetch Inventory (yfinance/Yahoo)
| Location | Data Used | Current Source |
| --- | --- | --- |
| `flask_app.py:get_sector` | sector lookup | `yf.Ticker().info` |
| `flask_app.py:fetch_price_map` | latest close for many tickers | `yf.download()` |
| `flask_app.py:get_kr_market_status` | 1y OHLC for ^KS11/^KQ11 | `yf.Ticker().history()` |
| `flask_app.py:get_portfolio_data` | indices + FX 5d prices | `yf.Ticker().history()` |
| `flask_app.py:get_us_portfolio_data` | US indices/commodities/crypto | `yf.Ticker().history()` |
| `flask_app.py:get_us_stock_chart` | OHLC candles | `yf.Ticker().history()` |
| `flask_app.py:get_us_macro_analysis` | live macro tickers | `yf.download()` |
| `flask_app.py:get_realtime_prices` | intraday 1m bars | `yf.download(interval='1m')` |
| `flask_app.py:get_technical_indicators` | 1y OHLC | `yf.Ticker().history()` |
| `flask_app.py:get_stock_detail` | KR ticker OHLC | `yf.Ticker().history()` |
| `us_market/create_us_daily_prices.py` | daily OHLCV per ticker | `yf.Ticker().history()` |
| `us_market/smart_money_screener_v2.py` | technicals, fundamentals, analyst data | `yf.Ticker().history()` / `yf.Ticker().info` |
| `us_market/sector_heatmap.py` | 5d OHLCV for ETFs/stocks | `yf.download()` |
| `us_market/macro_analyzer.py` | macro indicators + 52w high/low | `yf.download()` / `yf.Ticker().history()` |
| `us_market/analyze_etf_flows.py` | 90d OHLCV for ETFs | `yf.Ticker().history()` |
| `us_market/portfolio_risk.py` | 6mo close history | `yf.download()` |
| `us_market/options_flow.py` | options chain & IV | `yf.Ticker().options` / `.option_chain()` |
| `us_market/insider_tracker.py` | insider transactions | `yf.Ticker().insider_transactions` |
| `us_market/analyze_13f.py` | ownership, insider activity | `yf.Ticker().info` / `.insider_transactions` / `.institutional_holders` |
| `us_market/economic_calendar.py` | economic calendar | `finance.yahoo.com/calendar/economic` |
| `tests/conftest.py` | test mocks | `yfinance` monkeypatch |
| `requirements*.txt` | dependency | `yfinance` |

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
- Institutional ownership:
  - https://financialmodelingprep.com/api/v3/institutional-holder/AAPL
  - https://financialmodelingprep.com/api/v4/institutional-ownership/symbol-ownership?symbol=AAPL&includeCurrentQuarter=false
  - https://financialmodelingprep.com/stable/institutional-ownership/symbol-positions-summary?symbol=AAPL&year=2023&quarter=3
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

## Migration Plan (High Level)
1. Add config:
   - `FMP_API_KEY` in `.env` (and CI env if applicable).
   - Optional: `FMP_BASE_URL` (default `https://financialmodelingprep.com`).
2. Implement a thin FMP client (recommended new module):
   - Single `requests.Session`.
   - `get_json(path, params)` that appends `apikey`.
   - Basic retry/backoff for 429 responses.
   - Optional caching to replace yfinance batch calls.
3. Replace yfinance usage module-by-module (details below).
4. Update tests and dependencies:
   - Replace `mock_yfinance` in `tests/conftest.py` with `mock_fmp_client`.
   - Remove `yfinance` from `requirements*.txt` once migration is complete.

## Module-by-Module Edit Guidance
### `flask_app.py`
- `get_sector`: replace `yf.Ticker().info['sector']` with FMP `profile` (sector field).
- `fetch_price_map`: replace `yf.download()` with `quote` or `batch-quote`. Use `price` or `previousClose` depending on whether you need last close vs live price.
- `get_us_stock_chart`, `get_stock_detail`, `get_technical_indicators`: replace `Ticker.history()` with `historical-price-full` (daily OHLC). Map `historical[].date/open/high/low/close`.
- `get_realtime_prices`: replace `interval='1m'` with `historical-chart/1min` (stable) or use `quote` for current/open/high/low if 1m data is not required.
- `get_us_portfolio_data`, `get_us_macro_analysis`: use `quote` for indices/commodities/crypto/forex. For yields, prefer `treasury-rates`.
- `get_kr_market_status` / KR endpoints: check coverage via `index-list` and `forex-list`. If ^KS11/^KQ11 or KRW pairs are missing, keep Yahoo/yfinance fallback or add an alternate provider.
- `utils/perf_utils.py`: rename `yf_calls` / `yf_batches` to `fmp_calls` / `fmp_batches` (or add parallel counters).

### `us_market/create_us_daily_prices.py`
- Replace `yf.Ticker().history(start, end)` with `historical-price-full/{symbol}`.
- Convert FMP `historical` list to the CSV columns used here (`date`, `open`, `high`, `low`, `current_price`, `volume`, `change`, `change_rate`).
- Consider bulk endpoints (`eod-bulk` or batch quotes) if API limits are tight.

### `us_market/smart_money_screener_v2.py`
- Technicals: use `historical-price-full` for 6mo OHLC.
- Fundamentals: use `profile` (market cap, sector, name) + `key-metrics-ttm` + `ratios-ttm`.
- Analyst ratings/targets: use `ratings-snapshot` or `analyst-stock-recommendations` plus `price-target-consensus`.

### `us_market/sector_heatmap.py`
- Replace `yf.download()` with `batch-quote` for current price/volume plus `historical-price-full` (last 2-5 days) for change calculations.

### `us_market/analyze_etf_flows.py`
- Replace `Ticker.history()` with `historical-price-full` for ETFs (90d).

### `us_market/portfolio_risk.py`
- Replace `yf.download(..., period='6mo')['Close']` with `historical-price-full` and compute close series.
- Use `quote` or `historical-price-full` for SPY.

### `us_market/macro_analyzer.py`
- For VIX, SPY, BTC, GOLD, OIL: use `quote` with symbol mappings (see below).
- For yields (2Y/10Y): use `treasury-rates` instead of index proxies.

### `us_market/insider_tracker.py`
- Replace `insider_transactions` with `insider-trading` endpoints.
- Map fields like date, reporting name, transaction type, value, and shares from FMP response.

### `us_market/analyze_13f.py`
- Replace `info` ownership fields with `institutional-ownership` endpoints (symbol ownership and holders).
- Replace `insider_transactions` with `insider-trading` for insider sentiment.

### `us_market/options_flow.py`
- FMP docs do not list options chain endpoints. Decide between:
  - Keep yfinance for options only, or
  - Integrate a dedicated options data provider.

### `us_market/economic_calendar.py`
- Replace Yahoo scrape with `economic-calendar` endpoint from FMP.

## Symbol Mapping Notes (Yahoo -> FMP)
Confirm symbol formats using FMP list endpoints:
- Indices: `index-list` (examples in docs: `^GSPC`).
- Forex: `forex-list` (likely `USDKRW` rather than `KRW=X`).
- Crypto: `BTCUSD` instead of `BTC-USD`.
- Commodities: `GCUSD`, `CLUSD` instead of `GC=F`, `CL=F`.

## Testing/Docs Update Checklist
- Update `tests/conftest.py` to mock FMP client responses instead of yfinance.
- Remove `yfinance` from `requirements*.txt` once no code paths depend on it.
- Update project docs that mention yfinance (e.g., `PART1_Data_Collection.md`, `PART2_Analysis_Screening.md`, `strategy_overview.md`).

