# Improved US Stock Analysis System Blueprint - Part 6: Frontend Logic

This document details the core JavaScript logic embedded in `index.html` (Section 5.1 of the blueprint) that powers the interactive dashboard. The frontend is built with vanilla JavaScript, utilizing `fetch` for API communication and lightweight libraries for visualization.

## 6.1 Architecture Overview

The frontend architecture follows a component-based update pattern:
1.  **Initialization**: `DOMContentLoaded` triggers initial data loading and sets up event listeners.
2.  **State Management**: Global variables and `localStorage` manage user preferences (Language: `ko`/`en`, Model: `gemini`/`gpt`) and current view state (Selected Ticker, Chart Period).
3.  **Data Cycle**: 
    - `updateUSMarketDashboard()` is the central function for fetching snapshot data.
    - `updateRealtimePrices()` runs periodically to update individual stock prices.
    - `reloadMacroAnalysis()` refreshes AI insights separately to avoid blocking the UI.

## 6.2 Key Global Variables

| Variable | Description |
|---|---|
| `currentLang` | Current language ('ko' or 'en'), persisted in `localStorage`. |
| `currentModel` | Current AI model ('gemini' or 'gpt'), persisted in `localStorage`. |
| `usStockChart` | Instance of the Lightweight Charts object for the stock chart. |
| `currentChartPick` | Object containing details of the currently selected stock. |
| `indicatorState` | Object tracking active technical indicators (e.g., `{ bb: false, rsi: true }`). |

## 6.3 Core Functions & Logic

### 1. Dashboard Data Management

#### `updateUSMarketDashboard()`
**Purpose:** Fetches and orchestrates the rendering of all major dashboard sections.
**Logic:**
- Uses `Promise.all` to fetch data in parallel from:
    - `/api/us/portfolio` (Market Indices)
    - `/api/us/smart-money` (Top Picks)
    - `/api/us/etf-flows` (ETF Data)
    - `/api/us/history-dates` (Available Historical Data)
- Calls specific render functions for each section (`renderUSMarketIndices`, `renderUSSmartMoneyPicks`, etc.).
- Handles basic error logging.

#### `reloadMacroAnalysis()`
**Purpose:** Refreshes the Macro Analysis section, which may take longer due to AI generation.
**Logic:**
- Fetches from `/api/us/macro-analysis`.
- Passes `lang` and `model` parameters to backend.
- Updates the Macro Indicators grid and the AI Analysis text block.
- **Auto-Refresh:** Set to run every 10 minutes via `setInterval`.

### 2. Stock Chart & Technical Analysis

#### `loadUSStockChart(pick, idx, period)`
**Purpose:** Loads and displays the detailed chart for a selected stock.
**Logic:**
1.  **UI Updates:** Highlights the selected row in the table, updates the chart header with Ticker/Name/Score.
2.  **Data Fetching:** Calls `/api/us/stock-chart/<ticker>?period=<period>`.
3.  **Rendering:** 
    - Destroys existing chart instance if present.
    - Creates a new `LightweightCharts` instance.
    - Adds Candlestick series with Red/Green coloring.
4.  **Indicator Re-application:** If indicators (RSI, BB) were active, re-calculates and overlays them on the new chart.
5.  **AI Summary:** Triggers `loadUSAISummary` to fetch the stock's specific AI report.

#### `toggleIndicator(type)`
**Purpose:** Toggles technical indicators (Bollinger Bands, RSI, MACD, S/R).
**Logic:**
- Updates `indicatorState`.
- Toggles button styling (Active/Inactive).
- Checks if indicator data is already loaded; if not, fetches from `/api/us/technical-indicators`.
- Calls `renderIndicator(type)` to draw or remove the series from the chart.

### 3. Rendering Components

#### `renderUSSmartMoneyPicks(data)`
**Purpose:** Populates the main "Smart Money Picks" table.
**Logic:**
- Iterates through `data.top_picks`.
- Generates HTML rows (`<tr>`) with color-coded values for:
    - **Score**: Blue text.
    - **Upside**: Green/Red based on value.
    - **AI Recommendation**: Adds emojis (🔥/📈/📊).
- Attaches `click` event listener to each row to trigger `loadUSStockChart`.
- **Performance:** Calculates `change_since_rec` to show performance of the pick since analysis.

#### `renderUSMacroAnalysis(data)`
**Purpose:** Displays the grid of macro indicators (VIX, 10Y Yield, etc.).
**Logic:**
- Iterates `data.macro_indicators`.
- Applies specific styling (Purple glow for Volatility, Orange for Crypto, Blue for Yields).
- Shows daily change percentage with arrows.

### 4. Utility Functions

#### `translateUI()`
**Purpose:** Applies internationalization.
**Logic:**
- Selects all elements with `data-i18n` attribute.
- Replaces `textContent` with the string corresponding to `currentLang` from the `i18n` dictionary.

#### `updateRealtimePrices()`
**Purpose:** Live price updates for the table without reloading the page.
**Logic:**
- Collects visible tickers from the DOM.
- Sends batch request to `/api/realtime-prices`.
- Updates the "Current Price" cell and flashes Green/Red to indicate tick direction.
- Updates the Chart's last candle if the ticker matches the currently viewed chart.

## 6.4 API Integration Points

The frontend communicates with the Flask backend via the following key endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/us/portfolio` | GET | Market indices data |
| `/api/us/smart-money` | GET | Top stock picks and scores |
| `/api/us/stock-chart/<ticker>` | GET | OHLC data for charts |
| `/api/us/macro-analysis` | GET | Macro indicators and AI summary |
| `/api/realtime-prices` | POST | Batch real-time price updates |
| `/api/us/technical-indicators/<ticker>` | GET | Calculated RSI, MACD, BB data |

---
**Note:** This logic is fully contained within the `<script>` tag at the end of `index.html`. To modify behavior, edit the corresponding functions in that file.
