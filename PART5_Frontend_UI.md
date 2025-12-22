# Improved US Stock Analysis System Blueprint - Part 5: Frontend UI

This document contains the complete source code for the frontend HTML structure (`templates/index.html`).

## 5.1 Main HTML Template (`templates/index.html`)

This file defines the UI structure, Tailwind CSS styling, and embedded JavaScript for the market dashboard.

```html
<!DOCTYPE html>
<html lang="ko">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Korean Market - AI Stock Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        body {
            font-family: 'Inter', sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            overflow: hidden;
            /* Prevent full page scroll */
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1a1a1a;
        }

        ::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #444;
        }

        .koyfin-border {
            border-color: #2a2a2a;
        }

        .koyfin-header-bg {
            background-color: #121212;
        }

        .koyfin-panel-bg {
            background-color: #121212;
            /* Main background */
        }

        .koyfin-card-bg {
            background-color: #121212;
            /* Cards blend in or slightly lighter if needed, but Koyfin is flat */
        }

        .koyfin-green {
            color: #00E396;
        }

        .koyfin-red {
            color: #FF4560;
        }

        /* Tab Styling */
        .nav-tab {
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 500;
            color: #888;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }

        .nav-tab:hover {
            color: #fff;
        }

        .nav-tab.active {
            color: #fff;
            border-bottom-color: #fff;
            /* White underline for active tab */
            font-weight: 600;
        }

        /* Table Styling */
        .koyfin-table th {
            font-weight: 500;
            color: #888;
            font-size: 11px;
            padding: 8px;
            border-bottom: 1px solid #2a2a2a;
            text-align: right;
        }

        .koyfin-table th:first-child {
            text-align: left;
        }

        .koyfin-table td {
            padding: 8px;
            font-size: 12px;
            border-bottom: 1px solid #1f1f1f;
            text-align: right;
        }

        .koyfin-table td:first-child {
            text-align: left;
        }

        .koyfin-table tr:hover td {
            background-color: #1a1a1a;
        }

        /* Style Box Grid */
        .style-box-cell {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 500;
            color: #121212;
            /* Text color on colored bg */
            border: 1px solid #121212;
            /* Grid lines */
        }

        /* Utility for dense layout */
        .text-xxs {
            font-size: 10px;
        }
    </style>
</head>

<body class="flex flex-col h-screen w-screen text-sm bg-[#121212]">

    <!-- Top Navigation Bar -->
    <header class="h-14 border-b koyfin-border flex items-center px-4 justify-between shrink-0 bg-[#121212]">
        <div class="flex items-center gap-4">
            <!-- Sidebar Toggle / Menu -->
            <button class="text-gray-400 hover:text-white"><i class="fas fa-bars"></i></button>

            <!-- Breadcrumb / Title Area -->
            <div class="flex items-center gap-2 text-sm">
                <button class="text-gray-500 hover:text-white flex items-center gap-1 text-xs font-medium">
                    <i class="fas fa-chevron-left"></i> Back
                </button>
                <div class="h-4 w-px bg-[#333] mx-2"></div>
                <h1 id="main-header-title" class="text-lg font-bold text-white tracking-tight">🇰🇷 KR Market</h1>
            </div>
        </div>

        <!-- Search Bar -->
        <div class="flex-1 max-w-xl mx-8">
            <div class="relative group">
                <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <i class="fas fa-search text-gray-600 group-focus-within:text-blue-500"></i>
                </div>
                <input type="text"
                    class="block w-full pl-10 pr-3 py-1.5 border border-[#333] rounded bg-[#1a1a1a] text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600 sm:text-sm transition-colors"
                    placeholder="Search for a name, ticker, or function (/)">
                <div class="absolute inset-y-0 right-0 pr-2 flex items-center pointer-events-none">
                    <span class="text-gray-600 text-xs border border-[#333] rounded px-1.5">/</span>
                </div>
            </div>
        </div>

        <!-- Action Buttons -->
        <div class="flex items-center gap-2">
            <button
                class="bg-[#1a1a1a] hover:bg-[#2a2a2a] text-gray-300 border border-[#333] px-3 py-1.5 rounded text-xs font-medium flex items-center gap-2 transition-colors">
                <i class="far fa-file-alt"></i> Create Report
            </button>
            <button
                class="bg-[#1a1a1a] hover:bg-[#2a2a2a] text-gray-300 border border-[#333] px-3 py-1.5 rounded text-xs font-medium flex items-center gap-2 transition-colors">
                <i class="far fa-sticky-note"></i> My Notes
            </button>
            <button
                class="bg-[#1a1a1a] hover:bg-[#2a2a2a] text-gray-300 border border-[#333] px-3 py-1.5 rounded text-xs font-medium flex items-center gap-2 transition-colors">
                <i class="fas fa-share-alt"></i> Share
            </button>
            <button
                class="bg-[#e0e0e0] hover:bg-white text-black border border-transparent px-3 py-1.5 rounded text-xs font-bold flex items-center gap-2 transition-colors">
                <i class="fas fa-pen"></i> Edit Portfolio
            </button>
        </div>
    </header>

    <!-- Tab Navigation -->
    <div class="h-10 border-b koyfin-border flex items-center px-4 gap-1 bg-[#121212] shrink-0">
        <div class="nav-tab active" onclick="switchMarketTab(this, '🇰🇷 KR Market')">🇰🇷 KR Market</div>
        <div class="nav-tab" onclick="switchMarketTab(this, '🇺🇸 US Market')">🇺🇸 US Market</div>
        <div class="nav-tab" onclick="switchMarketTab(this, 'Economic Calendar')">Economic Calendar</div>
        <div class="nav-tab">Historical Returns</div>
        <div class="nav-tab">Risk</div>
        <div class="nav-tab">Holdings Matrix</div>
    </div>

    <!-- Main Content Area (Scrollable) -->
    <main class="flex-1 overflow-auto p-4 bg-[#121212]">
        <!-- KR Market Content -->
        <div id="content-kr-market">
            <!-- Key Stats Section -->
            <section class="mb-6">
                <div class="flex items-center justify-between mb-2">
                    <h2 class="text-sm font-bold text-gray-300">Market Indices</h2>
                    <span class="text-xs text-gray-500">Real-time Data</span>
                </div>
                <div id="market-indices-container" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    <!-- Populated by JS -->
                    <div
                        class="bg-[#1a1a1a] border border-[#2a2a2a] rounded p-3 flex flex-col items-center justify-center h-20 animate-pulse">
                        <div class="h-3 w-16 bg-[#2a2a2a] rounded mb-2"></div>
                        <div class="h-4 w-20 bg-[#2a2a2a] rounded"></div>
                    </div>
                </div>
            </section>

            <!-- Grid Layout for Charts & Tables -->
            <div class="grid grid-cols-12 gap-6">

                <!-- Left Column: Holdings Chart & Table (Span 8) -->
                <div class="col-span-12 lg:col-span-8 flex flex-col gap-6">

                    <!-- Interactive Price Chart -->
                    <div class="flex-1 min-h-[300px] border border-[#2a2a2a] rounded p-4 relative flex flex-col">
                        <div class="flex items-center justify-between mb-2 shrink-0">
                            <div>
                                <h2 class="text-sm font-bold text-gray-300">Price Chart</h2>
                                <span id="summary-chart-ticker" class="text-xs text-blue-400 font-mono ml-2"></span>
                            </div>
                            <!-- Time Range Buttons -->
                            <div class="flex items-center gap-1 bg-[#1a1a1a] rounded p-0.5 border border-[#333]">
                                <button onclick="setChartRange('1D')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-gray-400 hover:text-white hover:bg-[#333] rounded transition-colors"
                                    data-range="1D">1D</button>
                                <button onclick="setChartRange('1W')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-gray-400 hover:text-white hover:bg-[#333] rounded transition-colors"
                                    data-range="1W">1W</button>
                                <button onclick="setChartRange('1M')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-gray-400 hover:text-white hover:bg-[#333] rounded transition-colors"
                                    data-range="1M">1M</button>
                                <button onclick="setChartRange('3M')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-gray-400 hover:text-white hover:bg-[#333] rounded transition-colors"
                                    data-range="3M">3M</button>
                                <button onclick="setChartRange('6M')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-gray-400 hover:text-white hover:bg-[#333] rounded transition-colors"
                                    data-range="6M">6M</button>
                                <button onclick="setChartRange('1Y')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-white bg-[#333] rounded transition-colors"
                                    data-range="1Y">1Y</button>
                                <button onclick="setChartRange('5Y')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-gray-400 hover:text-white hover:bg-[#333] rounded transition-colors"
                                    data-range="5Y">5Y</button>
                                <button onclick="setChartRange('All')"
                                    class="chart-range-btn px-2 py-0.5 text-[10px] font-medium text-gray-400 hover:text-white hover:bg-[#333] rounded transition-colors"
                                    data-range="All">All</button>
                            </div>
                        </div>
                        <div id="summary-chart-container" class="flex-1 w-full relative"></div>
                    </div>

                    <!-- Holdings Table -->
                    <div class="flex-1 min-h-[300px] border border-[#2a2a2a] rounded p-4 flex flex-col">
                        <div class="flex items-center justify-between mb-4 shrink-0">
                            <h2 class="text-lg font-semibold text-gray-100 flex items-center gap-2">
                                AI Recommendations
                            </h2>
                            <div class="flex items-center gap-4">
                                <div class="flex items-center gap-2">
                                    <label for="history-date" class="text-gray-400 text-sm">Date:</label>
                                    <input type="date" id="history-date"
                                        class="bg-gray-800 text-white border border-gray-700 rounded px-3 py-1 text-sm focus:outline-none focus:border-blue-500">
                                    <button onclick="resetDate()"
                                        class="text-xs text-gray-500 hover:text-white underline">Reset</button>
                                </div>
                                <span class="text-sm text-gray-400">Update Interval: <span
                                        class="text-blue-400 font-mono">20s</span></span>
                            </div>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-left border-collapse">
                                <thead>
                                    <tr class="text-xs text-gray-400 border-b border-gray-800">
                                        <th class="py-3 text-left font-semibold text-gray-400">Ticker</th>
                                        <th class="py-3 text-left font-semibold text-gray-400">Name</th>
                                        <th class="py-3 text-left font-semibold text-gray-400">Rec. Price</th>
                                        <th class="py-3 text-left font-semibold text-gray-400">Price</th>
                                        <th class="py-3 text-left font-semibold text-gray-400">Return</th>
                                        <th class="py-3 text-center font-semibold text-gray-400">AI Score</th>
                                        <th class="py-3 text-center font-semibold text-gray-400">Grade</th>
                                        <th class="py-3 text-center font-semibold text-gray-400">Wave Stage</th>
                                        <th class="py-3 text-center font-semibold text-gray-400">S/D Stage</th>
                                        <th class="py-3 text-center font-semibold text-gray-400">Inst. Trend</th>
                                        <th class="py-3 text-center font-semibold text-gray-400">YTD</th>
                                    </tr>
                                </thead>
                                <tbody id="holdings-table-body" class="divide-y divide-gray-700">
                                    <!-- Rows will be populated by JS -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Right Column: Style Box (Span 4) -->
                <div class="col-span-12 lg:col-span-4 flex flex-col gap-6">

                    <!-- Style Box Component -->
                    <div class="h-full border border-[#2a2a2a] rounded p-4 flex flex-col">
                        <div class="flex items-center justify-between mb-4 shrink-0">
                            <h2 class="text-sm font-bold text-gray-300">Style Box</h2>
                            <button class="text-xs text-gray-500 hover:text-white flex items-center gap-1"><i
                                    class="fas fa-download"></i> DOWNLOAD</button>
                        </div>

                        <!-- Style Box Controls -->
                        <div class="flex gap-1 mb-4 bg-[#1a1a1a] p-1 rounded w-max">
                            <button onclick="switchTab('summary')" id="tab-summary"
                                class="tab-btn active text-sm font-medium text-white border-b-2 border-blue-500 pb-2 px-1">Summary</button>
                            <button onclick="switchTab('analysis')" id="tab-analysis"
                                class="tab-btn text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent pb-2 px-1 transition-colors">Analysis</button>
                            <button
                                class="tab-btn text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent pb-2 px-1 transition-colors">Exposure</button>
                        </div>

                        <!-- The 3x3 Grid -->
                        <div class="flex-1 flex flex-col">
                            <!-- Column Headers -->
                            <div class="flex text-xs text-gray-500 mb-1 text-center font-medium">
                                <div class="w-8"></div> <!-- Row Header Spacer -->
                                <div class="flex-1">Value</div>
                                <div class="flex-1">Core</div>
                                <div class="flex-1">Growth</div>
                            </div>

                            <div class="flex-1 flex gap-1">
                                <!-- Row Headers -->
                                <div class="w-8 flex flex-col justify-between text-xs text-gray-500 font-medium py-4">
                                    <div class="flex-1 flex items-center justify-start">Large</div>
                                    <div class="flex-1 flex items-center justify-start">Mid</div>
                                    <div class="flex-1 flex items-center justify-start">Small</div>
                                </div>

                                <!-- Grid -->
                                <div class="flex-1 flex flex-col gap-1 h-[300px]">
                                    <!-- Large Row -->
                                    <div class="flex-1 flex gap-1">
                                        <div id="sb-large-value" class="flex-1 style-box-cell bg-[#9dcba3]">--%</div>
                                        <div id="sb-large-core" class="flex-1 style-box-cell bg-[#86bf8e]">--%</div>
                                        <div id="sb-large-growth" class="flex-1 style-box-cell bg-[#b4d7b8]">--%</div>
                                    </div>
                                    <!-- Mid Row -->
                                    <div class="flex-1 flex gap-1">
                                        <div id="sb-mid-value" class="flex-1 style-box-cell bg-[#cce4ce]">--%</div>
                                        <div id="sb-mid-core" class="flex-1 style-box-cell bg-[#b4d7b8]">--%</div>
                                        <div id="sb-mid-growth" class="flex-1 style-box-cell bg-[#dcecd0]">--%</div>
                                    </div>
                                    <!-- Small Row -->
                                    <div class="flex-1 flex gap-1">
                                        <div id="sb-small-value" class="flex-1 style-box-cell bg-[#dcecd0]">--%</div>
                                        <div id="sb-small-core" class="flex-1 style-box-cell bg-[#dcecd0]">--%</div>
                                        <div id="sb-small-growth" class="flex-1 style-box-cell bg-[#dcecd0]">--%</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- US Market Content -->
        <div id="content-us-market" class="hidden">
            <!-- Market Indices Section -->
            <section class="mb-6">
                <div class="flex items-center justify-between mb-2">
                    <h2 class="text-sm font-bold text-gray-300">🇺🇸 US Market Indices</h2>
                    <div class="flex items-center gap-3">
                        <div id="lang-toggle" class="flex items-center gap-1 bg-[#2a2a2a] rounded-full p-0.5">
                            <button data-lang="ko"
                                class="px-2 py-0.5 text-xs rounded-full bg-blue-600 text-white transition-colors">한국어</button>
                            <button data-lang="en"
                                class="px-2 py-0.5 text-xs rounded-full text-gray-400 hover:text-white transition-colors">English</button>
                        </div>
                        <span class="text-xs text-gray-500">Real-time Data</span>
                    </div>
                </div>
                <div id="us-market-indices-container" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    <!-- Populated by JS -->
                    <div
                        class="bg-[#1a1a1a] border border-[#2a2a2a] rounded p-3 flex flex-col items-center justify-center h-20 animate-pulse">
                        <div class="h-3 w-16 bg-[#2a2a2a] rounded mb-2"></div>
                        <div class="h-4 w-20 bg-[#2a2a2a] rounded"></div>
                    </div>
                </div>
            </section>

            <!-- US Stock Chart Section -->
            <section class="mb-6">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <!-- Chart -->
                    <div class="lg:col-span-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded p-4">
                        <div class="flex items-center justify-between mb-3">
                            <div>
                                <h2 id="us-chart-ticker" class="text-lg font-bold text-white">Select a stock</h2>
                                <span id="us-chart-info" class="text-xs text-gray-400">Click on a stock from the table
                                    below</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <div id="us-chart-period-btns" class="flex gap-1">
                                    <button data-period="1mo"
                                        class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white transition-colors">1M</button>
                                    <button data-period="3mo"
                                        class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white transition-colors">3M</button>
                                    <button data-period="6mo"
                                        class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white transition-colors">6M</button>
                                    <button data-period="1y"
                                        class="px-2 py-1 text-xs rounded bg-blue-600 text-white">1Y</button>
                                    <button data-period="max"
                                        class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white transition-colors">ALL</button>
                                </div>
                                <div id="us-chart-grade"
                                    class="px-3 py-1 rounded text-sm font-bold bg-gray-700 text-gray-300">-</div>
                            </div>
                        </div>
                        <div id="us-stock-chart" class="h-[300px] flex items-center justify-center text-gray-500">
                            <i class="fas fa-chart-line text-3xl"></i>
                        </div>
                        <!-- Technical Indicator Toggles -->
                        <div class="flex items-center gap-2 mt-3 pt-3 border-t border-gray-700">
                            <span class="text-xs text-gray-500">Indicators:</span>
                            <button id="toggle-bb" onclick="toggleIndicator('bb')"
                                class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-purple-600 transition-colors">📊
                                BB</button>
                            <button id="toggle-sr" onclick="toggleIndicator('sr')"
                                class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-orange-600 transition-colors">📏
                                S/R</button>
                            <button id="toggle-rsi" onclick="toggleIndicator('rsi')"
                                class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-green-600 transition-colors">📈
                                RSI</button>
                            <button id="toggle-macd" onclick="toggleIndicator('macd')"
                                class="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-blue-600 transition-colors">📉
                                MACD</button>
                        </div>
                        <!-- RSI Sub-Chart -->
                        <div id="us-rsi-chart" class="hidden h-[80px] mt-2 border-t border-gray-700 pt-2"></div>
                        <!-- MACD Sub-Chart -->
                        <div id="us-macd-chart" class="hidden h-[80px] mt-2 border-t border-gray-700 pt-2"></div>
                    </div>
                    <!-- Sector Heatmap -->
                    <div class="bg-[#1a1a1a] border border-[#2a2a2a] rounded p-4">
                        <div class="flex justify-between items-center mb-3">
                            <h3 class="text-sm font-bold text-gray-400"><i
                                    class="fas fa-globe-americas text-blue-400 mr-1"></i> Market Map</h3>
                            <span id="us-heatmap-date" class="text-xs text-gray-500"></span>
                        </div>
                        <div id="us-sector-heatmap" class="w-full h-[300px] relative">
                            <!-- ApexCharts Treemap will be rendered here -->
                            <div class="flex items-center justify-center h-full text-xs text-gray-500">Loading Market
                                Map...</div>
                        </div>
                    </div>
                </div>
                <!-- AI Summary Section -->
                <div class="mt-4 bg-[#1a1a1a] border border-[#2a2a2a] rounded p-4">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-sm font-bold text-purple-400" data-i18n="ai-analysis">🤖 AI 투자 분석</h3>
                        <span id="us-ai-updated" class="text-xs text-gray-500"></span>
                    </div>
                    <div id="us-ai-summary" class="text-sm text-gray-300 leading-relaxed">
                        <span class="text-gray-500">종목을 선택하면 AI 분석이 표시됩니다.</span>
                    </div>
                </div>
            </section>

            <!-- Smart Money Picks Section -->
            <section class="mb-6">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-3">
                        <h2 class="text-sm font-bold text-gray-300" data-i18n="final-top10">📊 Final Top 10 - Smart
                            Money Picks</h2>
                        <select id="us-history-date-select"
                            class="bg-[#1a1a1a] border border-[#2a2a2a] rounded px-2 py-1 text-xs text-gray-300"
                            onchange="loadUSHistoryByDate(this.value)">
                            <option value="">📅 날짜 선택</option>
                        </select>
                    </div>
                    <span id="us-smart-money-summary" class="text-xs text-gray-500">Loading...</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="bg-[#1a1a1a] text-gray-400 text-xs">
                                <th class="p-2 text-left">#</th>
                                <th class="p-2 text-left">Ticker</th>
                                <th class="p-2 text-center">Sector</th>
                                <th class="p-2 text-center">Score</th>
                                <th class="p-2 text-center">AI 추천</th>
                                <th class="p-2 text-center">추천가</th>
                                <th class="p-2 text-center">현재가</th>
                                <th class="p-2 text-center">변동</th>
                                <th class="p-2 text-center">목표 Upside</th>
                            </tr>
                        </thead>
                        <tbody id="us-smart-money-table">
                            <tr class="animate-pulse">
                                <td colspan="8" class="p-4 text-center text-gray-500">Loading...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

            <!-- ETF Flows Section -->
            <section class="mb-6">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center">
                        <h2 class="text-sm font-bold text-gray-300" data-i18n="etf-flows">💰 ETF Fund Flows - 자금 흐름</h2>
                        <button id="us-etf-ai-btn"
                            class="hidden ml-2 px-2 py-0.5 text-xs rounded-full bg-indigo-600 hover:bg-indigo-500 text-white transition-colors">
                            🤖 AI 분석
                        </button>
                    </div>
                    <span id="us-etf-sentiment"
                        class="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300">Loading...</span>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <!-- Top Inflows -->
                    <div class="bg-[#1a1a1a] rounded border border-[#2a2a2a] p-4">
                        <h3 class="text-xs text-green-400 font-bold mb-2" data-i18n="top-inflows">📈 Top Inflows</h3>
                        <div id="us-etf-inflows" class="space-y-2"></div>
                    </div>
                    <!-- Top Outflows -->
                    <div class="bg-[#1a1a1a] rounded border border-[#2a2a2a] p-4">
                        <h3 class="text-xs text-red-400 font-bold mb-2" data-i18n="top-outflows">📉 Top Outflows</h3>
                        <div id="us-etf-outflows" class="space-y-2"></div>
                    </div>
                    <!-- AI Analysis Container -->
                    <div id="us-etf-ai-container"
                        class="hidden mt-4 col-span-2 bg-[#1e1e1e] border border-indigo-500/30 rounded p-4">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="text-lg">🧠</span>
                            <span class="text-sm font-bold text-indigo-300">Gemini 3.0 Market Insight</span>
                        </div>
                        <div id="us-etf-ai-content" class="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                        </div>
                    </div>
                </div>
            </section>

            <!-- Options Flow Section -->
            <section class="mb-6">
                <div class="flex items-center justify-between mb-2">
                    <h2 class="text-sm font-bold text-gray-300"><i class="fas fa-chart-area text-orange-400 mr-1"></i>
                        <span data-i18n="options-flow">Options Flow - 기관 포지션</span>
                    </h2>
                    <span id="us-options-timestamp" class="text-xs text-gray-500"></span>
                </div>
                <div class="grid grid-cols-2 md:grid-cols-5 gap-3" id="us-options-flow">
                    <div class="text-center text-xs text-gray-500">Loading options...</div>
                </div>
            </section>

            <!-- Macro Analysis Section -->
            <section class="mb-6">
                <div class="flex items-center justify-between mb-2">
                    <h2 class="text-sm font-bold text-gray-300" data-i18n="macro-analysis">🌍 Macro Analysis - AI 예측
                    </h2>
                    <span id="us-macro-timestamp" class="text-xs text-gray-500"></span>
                </div>

                <!-- Macro Indicators Grid -->
                <div class="grid grid-cols-5 md:grid-cols-10 gap-2 mb-4" id="us-macro-indicators">
                    <div class="text-center text-xs text-gray-500">Loading...</div>
                </div>

                <!-- Calendar Moved to Performance Tab -->

                <!-- AI Analysis Card -->
                <div
                    class="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg border border-purple-800/50 p-4">
                    <div class="flex items-center justify-between mb-3">
                        <div class="flex items-center gap-2">
                            <span class="text-lg">🤖</span>
                            <span id="macro-model-label" class="text-sm font-bold text-purple-300"
                                data-i18n="gemini-macro">Gemini 3.0 매크로 분석</span>
                        </div>
                        <!-- Model Toggle Button -->
                        <div id="model-toggle" class="flex items-center gap-1 bg-[#2a2a2a] rounded-full p-0.5">
                            <button data-model="gemini"
                                class="px-2 py-0.5 text-xs rounded-full bg-purple-600 text-white transition-colors">Gemini</button>
                            <button data-model="gpt"
                                class="px-2 py-0.5 text-xs rounded-full text-gray-400 hover:text-white transition-colors">GPT-5.2</button>
                        </div>
                    </div>
                    <div id="us-macro-ai-analysis" class="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                        분석 로딩 중...
                    </div>
                </div>
            </section>
        </div>

        <!-- Weekly Economic Calendar Tab Content -->
        <div id="content-economic-calendar" class="hidden p-4 space-y-6">
            <h2 class="text-xl font-bold text-gray-200 mb-4">Economic Calendar</h2>
            <!-- Weekly Economic Calendar (Moved) -->
            <div class="mb-4 bg-[#1a1a1a] border border-[#2a2a2a] rounded p-4 max-w-2xl">
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center gap-3">
                        <h3 class="text-sm font-bold text-gray-300">📅 Weekly Economic Calendar</h3>
                        <span id="us-calendar-range" class="text-xs text-gray-500"></span>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="expandAllCalendarEvents()"
                            class="px-2 py-1 text-[10px] bg-gray-800 hover:bg-gray-700 text-gray-300 rounded transition-colors">
                            🔽 Expand All
                        </button>
                        <button onclick="collapseAllCalendarEvents()"
                            class="px-2 py-1 text-[10px] bg-gray-800 hover:bg-gray-700 text-gray-300 rounded transition-colors">
                            🔼 Collapse All
                        </button>
                    </div>
                </div>
                <div id="us-calendar-events" class="space-y-2">
                    <div class="text-center text-xs text-gray-500">Loading calendar...</div>
                </div>
            </div>

            <div class="text-gray-500 text-sm">
                Future performance charts will be displayed here.
            </div>
        </div>

        <!-- Analysis Tab Content -->
        <div id="content-analysis" class="hidden space-y-4">
            <!-- Top Section: Chart & Metrics -->
            <div class="grid grid-cols-12 gap-4">
                <!-- Chart Section -->
                <div
                    class="col-span-12 lg:col-span-8 bg-gray-900 rounded-lg border border-gray-800 p-4 h-[500px] flex flex-col">
                    <div class="flex justify-between items-center mb-4">
                        <div>
                            <h2 id="analysis-ticker-name" class="text-xl font-bold text-white">Select a Stock</h2>
                            <span id="analysis-ticker-code" class="text-sm text-gray-400"></span>
                        </div>
                        <div class="flex gap-2">
                            <span id="analysis-grade"
                                class="px-2 py-1 rounded text-xs font-bold bg-gray-800 text-gray-300">Grade</span>
                            <span id="analysis-score"
                                class="px-2 py-1 rounded text-xs font-bold bg-gray-800 text-gray-300">Score</span>
                        </div>
                    </div>
                    <div id="analysis-chart-container" class="flex-1 w-full relative"></div>
                </div>

                <!-- Metrics Side Panel -->
                <div class="col-span-12 lg:col-span-4 space-y-4">
                    <!-- Wave & S/D -->
                    <div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
                        <h3 class="text-sm font-medium text-gray-400 mb-3">Market Status</h3>
                        <div class="space-y-3">
                            <div>
                                <div class="text-xs text-gray-500 mb-1">Wave Stage</div>
                                <div id="analysis-wave" class="text-lg font-semibold text-white">-</div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-500 mb-1">Supply/Demand</div>
                                <div id="analysis-sd" class="text-lg font-semibold text-white">-</div>
                            </div>
                        </div>
                    </div>

                    <!-- Trends -->
                    <div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
                        <h3 class="text-sm font-medium text-gray-400 mb-3">Institutional Trends</h3>
                        <div class="space-y-3">
                            <div class="flex justify-between items-center">
                                <span class="text-sm text-gray-400">Institutional</span>
                                <span id="analysis-inst-trend" class="text-sm font-medium text-white">-</span>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-sm text-gray-400">Foreign</span>
                                <span id="analysis-for-trend" class="text-sm font-medium text-white">-</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- AI Report Section -->
            <div class="bg-gray-900 rounded-lg border border-gray-800 p-6">
                <h2 class="text-lg font-semibold text-gray-100 mb-4 flex items-center gap-2">
                    <i class="fas fa-robot text-blue-400"></i> AI Analysis Report
                </h2>
                <div id="analysis-report-content" class="prose prose-invert max-w-none text-sm text-gray-300">
                    <p class="text-gray-500 italic">Select a stock to view the detailed AI analysis report.</p>
                </div>
            </div>
        </div>
    </main>

    <script>
        // --- Global Variables ---
        let priceUpdateInterval;
        let currentChartTicker = null;
        let selectedDate = null;

        // --- Market Tab Switching ---
        function switchMarketTab(tabElement, marketName) {
            // Update header title
            document.getElementById('main-header-title').textContent = marketName;

            // Update active tab styling
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            tabElement.classList.add('active');

            // Toggle content visibility
            const krContent = document.getElementById('content-kr-market');
            const usContent = document.getElementById('content-us-market');
            const calcContent = document.getElementById('content-economic-calendar');

            // Hide all first
            krContent.classList.add('hidden');
            usContent.classList.add('hidden');
            if (calcContent) calcContent.classList.add('hidden');

            if (marketName.includes('KR')) {
                krContent.classList.remove('hidden');
            } else if (marketName.includes('US')) {
                usContent.classList.remove('hidden');
                updateUSMarketDashboard();
            } else if (marketName.includes('Calendar')) {
                if (calcContent) calcContent.classList.remove('hidden');
                // We might want to trigger calendar update here if needed, 
                // but usually updateUSMarketDashboard handles it. 
                // Let's call it just in case to fetch calendar data
                updateUSMarketDashboard();
            }
        }


        // --- Language Toggle ---
        let currentLang = localStorage.getItem('dashboardLang') || 'ko';

        // Translation dictionary
        const i18n = {
            'us-market-indices': { ko: '🇺🇸 US Market Indices', en: '🇺🇸 US Market Indices' },
            'realtime-data': { ko: 'Real-time Data', en: 'Real-time Data' },
            'final-top10': { ko: '📊 Final Top 10 - Smart Money Picks', en: '📊 Final Top 10 - Smart Money Picks' },
            'sector': { ko: 'Sector', en: 'Sector' },
            'score': { ko: 'Score', en: 'Score' },
            'ai-rec': { ko: 'AI 추천', en: 'AI Rec' },
            'rec-price': { ko: '추천가', en: 'Rec Price' },
            'current-price': { ko: '현재가', en: 'Current' },
            'change': { ko: '변동', en: 'Change' },
            'target-upside': { ko: '목표 Upside', en: 'Target Upside' },
            'market-map': { ko: 'Market Map (Sector & Stocks)', en: 'Market Map (Sector & Stocks)' },
            'ai-analysis': { ko: '🤖 AI 투자 분석', en: '🤖 AI Investment Analysis' },
            'etf-flows': { ko: '💰 ETF Fund Flows - 자금 흐름', en: '💰 ETF Fund Flows' },
            'top-inflows': { ko: '📈 Top Inflows', en: '📈 Top Inflows' },
            'top-outflows': { ko: '📉 Top Outflows', en: '📉 Top Outflows' },
            'options-flow': { ko: 'Options Flow - 기관 포지션', en: 'Options Flow - Institutional Positions' },
            'macro-analysis': { ko: '🌍 Macro Analysis - AI 예측', en: '🌍 Macro Analysis - AI Predictions' },
            'gemini-macro': { ko: 'Gemini 3.0 매크로 분석', en: 'Gemini 3.0 Macro Analysis' },
            'gpt-macro': { ko: 'GPT-5.2 매크로 분석', en: 'GPT-5.2 Macro Analysis' },
            'loading': { ko: '분석 로딩 중...', en: 'Loading analysis...' },
            'loading-options': { ko: 'Loading options...', en: 'Loading options...' },
            'loading-market-map': { ko: 'Loading Market Map...', en: 'Loading Market Map...' }
        };

        function translateUI() {
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.dataset.i18n;
                if (i18n[key]) {
                    el.textContent = i18n[key][currentLang] || i18n[key]['en'];
                }
            });
        }

        // Initialize language toggle UI
        document.getElementById('lang-toggle')?.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') {
                const lang = e.target.dataset.lang;
                if (lang !== currentLang) {
                    currentLang = lang;
                    localStorage.setItem('dashboardLang', lang);

                    // Update button styles
                    document.querySelectorAll('#lang-toggle button').forEach(btn => {
                        if (btn.dataset.lang === lang) {
                            btn.className = 'px-2 py-0.5 text-xs rounded-full bg-blue-600 text-white transition-colors';
                        } else {
                            btn.className = 'px-2 py-0.5 text-xs rounded-full text-gray-400 hover:text-white transition-colors';
                        }
                    });

                    // Translate UI text
                    translateUI();

                    // Reload macro analysis with new language
                    reloadMacroAnalysis();
                }
            }
        });

        // --- Model Toggle (Gemini/GPT) ---
        let currentModel = localStorage.getItem('dashboardModel') || 'gemini';

        // Initialize model toggle UI
        document.getElementById('model-toggle')?.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') {
                const model = e.target.dataset.model;
                if (model && model !== currentModel) {
                    currentModel = model;
                    localStorage.setItem('dashboardModel', model);

                    // Update button styles
                    document.querySelectorAll('#model-toggle button').forEach(btn => {
                        if (btn.dataset.model === model) {
                            btn.className = 'px-2 py-0.5 text-xs rounded-full bg-purple-600 text-white transition-colors';
                        } else {
                            btn.className = 'px-2 py-0.5 text-xs rounded-full text-gray-400 hover:text-white transition-colors';
                        }
                    });

                    // Update model label
                    const labelEl = document.getElementById('macro-model-label');
                    if (labelEl) {
                        if (model === 'gpt') {
                            labelEl.textContent = i18n['gpt-macro'][currentLang] || 'GPT-5.2 Macro Analysis';
                            labelEl.dataset.i18n = 'gpt-macro';
                        } else {
                            labelEl.textContent = i18n['gemini-macro'][currentLang] || 'Gemini 3.0 Macro Analysis';
                            labelEl.dataset.i18n = 'gemini-macro';
                        }
                    }

                    // Reload macro analysis with new model
                    reloadMacroAnalysis();
                }
            }
        });

        // Set initial button state based on saved language and model
        document.addEventListener('DOMContentLoaded', () => {
            // Language toggle
            document.querySelectorAll('#lang-toggle button').forEach(btn => {
                if (btn.dataset.lang === currentLang) {
                    btn.className = 'px-2 py-0.5 text-xs rounded-full bg-blue-600 text-white transition-colors';
                } else {
                    btn.className = 'px-2 py-0.5 text-xs rounded-full text-gray-400 hover:text-white transition-colors';
                }
            });

            // Model toggle
            document.querySelectorAll('#model-toggle button').forEach(btn => {
                if (btn.dataset.model === currentModel) {
                    btn.className = 'px-2 py-0.5 text-xs rounded-full bg-purple-600 text-white transition-colors';
                } else {
                    btn.className = 'px-2 py-0.5 text-xs rounded-full text-gray-400 hover:text-white transition-colors';
                }
            });

            // Update model label based on saved preference
            const labelEl = document.getElementById('macro-model-label');
            if (labelEl && currentModel === 'gpt') {
                labelEl.textContent = i18n['gpt-macro'][currentLang] || 'GPT-5.2 Macro Analysis';
                labelEl.dataset.i18n = 'gpt-macro';
            }

            // Apply initial translation
            translateUI();
        });

        async function reloadMacroAnalysis() {
            try {
                const macroRes = await fetch(`/api/us/macro-analysis?lang=${currentLang}&model=${currentModel}`);
                const macroData = await macroRes.json();
                if (macroData.macro_indicators) {
                    renderUSMacroAnalysis(macroData);
                }
            } catch (e) {
                console.log('Failed to reload macro analysis');
            }

            // Reload AI summary for current stock if one is selected
            if (currentChartPick) {
                loadUSAISummary(currentChartPick.ticker);
            }
        }

        // Auto-refresh Macro Analysis every 10 minutes (600000 ms)
        setInterval(() => {
            console.log('🔄 Auto-refreshing Macro Analysis...');
            reloadMacroAnalysis();
        }, 600000);  // 10 minutes

        // --- US Market Dashboard ---
        async function updateUSMarketDashboard() {
            try {
                // Fetch all US data in parallel
                const [portfolioRes, smartMoneyRes, etfFlowsRes, historyDatesRes] = await Promise.all([
                    fetch('/api/us/portfolio'),
                    fetch('/api/us/smart-money'),
                    fetch('/api/us/etf-flows'),
                    fetch('/api/us/history-dates')
                ]);

                const portfolioData = await portfolioRes.json();
                const smartMoneyData = await smartMoneyRes.json();
                const etfFlowsData = await etfFlowsRes.json();
                const historyDatesData = await historyDatesRes.json();

                // Render Market Indices
                if (portfolioData.market_indices) {
                    renderUSMarketIndices(portfolioData.market_indices);
                }

                // Populate history date selector
                if (historyDatesData.dates) {
                    populateUSHistoryDates(historyDatesData.dates);
                }

                // Render Smart Money Picks
                if (smartMoneyData.top_picks) {
                    renderUSSmartMoneyPicks(smartMoneyData);
                }

                // Render ETF Flows
                if (etfFlowsData.top_inflows) {
                    renderUSETFFlows(etfFlowsData);
                }

                // Fetch and render Macro Analysis
                try {
                    const macroRes = await fetch('/api/us/macro-analysis');
                    const macroData = await macroRes.json();
                    if (macroData.macro_indicators) {
                        renderUSMacroAnalysis(macroData);
                    }
                } catch (e) {
                    console.log('Macro analysis not available');
                }

                // Fetch and render Sector Heatmap
                try {
                    const sectorRes = await fetch('/api/us/sector-heatmap');
                    const sectorData = await sectorRes.json();
                    if (sectorData.series) {
                        renderUSSectorHeatmap(sectorData);
                    }
                } catch (e) {
                    console.log('Sector heatmap not available');
                }

                // Fetch and render Options Flow
                try {
                    const optionsRes = await fetch('/api/us/options-flow');
                    const optionsData = await optionsRes.json();
                    if (optionsData.options_flow) {
                        renderUSOptionsFlow(optionsData);
                    }
                } catch (e) {
                    console.log('Options flow not available');
                }

                // Fetch and render Weekly Calendar
                try {
                    const calendarRes = await fetch('/api/us/calendar');
                    const calendarData = await calendarRes.json();
                    renderUSCalendar(calendarData);
                } catch (e) {
                    console.log('Calendar not available');
                }

            } catch (e) {
                console.error("Error updating US dashboard:", e);
            }
        }

        function renderUSOptionsFlow(data) {
            const container = document.getElementById('us-options-flow');
            const timestampEl = document.getElementById('us-options-timestamp');

            if (!container) return;
            container.innerHTML = '';

            // Update timestamp
            if (timestampEl && data.timestamp) {
                const date = new Date(data.timestamp);
                timestampEl.textContent = `📅 ${date.toLocaleDateString('ko-KR')}`;
            }

            // Render top 10 stocks
            const stocks = data.options_flow.slice(0, 10);

            stocks.forEach(stock => {
                const signal = stock.signal;
                const metrics = stock.metrics;
                const unusual = stock.unusual_activity;

                // Determine background color based on sentiment
                let bgClass, borderClass;
                if (signal.sentiment_score >= 70) {
                    bgClass = 'bg-green-900/30';
                    borderClass = 'border-green-600/50';
                } else if (signal.sentiment_score <= 30) {
                    bgClass = 'bg-red-900/30';
                    borderClass = 'border-red-600/50';
                } else {
                    bgClass = 'bg-yellow-900/30';
                    borderClass = 'border-yellow-600/50';
                }

                const hasUnusual = unusual.unusual_call_count > 0 || unusual.unusual_put_count > 0;

                const div = document.createElement('div');
                div.className = `${bgClass} border ${borderClass} rounded-lg p-3 cursor-pointer hover:opacity-80 transition-opacity`;
                div.innerHTML = `
                    <div class="flex justify-between items-center mb-1">
                        <span class="text-sm font-bold text-white">${stock.ticker}</span>
                        <span class="text-xs text-gray-400">$${stock.current_price}</span>
                    </div>
                    <div class="text-xs mb-1">${signal.sentiment}</div>
                    <div class="flex justify-between text-xs text-gray-400">
                        <span>P/C: ${metrics.pc_volume_ratio.toFixed(2)}</span>
                        <span>IV: ${metrics.call_iv.toFixed(0)}%</span>
                    </div>
                    ${hasUnusual ? `<div class="text-xs text-orange-400 mt-1">⚡ ${unusual.unusual_call_count + unusual.unusual_put_count} unusual</div>` : ''}
                `;
                container.appendChild(div);
            });
        }

        function renderUSMacroAnalysis(data) {
            const indicatorsEl = document.getElementById('us-macro-indicators');
            const analysisEl = document.getElementById('us-macro-ai-analysis');
            const timestampEl = document.getElementById('us-macro-timestamp');

            // Render timestamp
            if (timestampEl && data.timestamp) {
                const date = new Date(data.timestamp);
                timestampEl.textContent = `📅 ${date.toLocaleDateString('ko-KR')} ${date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}`;
            }

            // Render indicators grid
            if (indicatorsEl && data.macro_indicators) {
                indicatorsEl.innerHTML = '';
                const indicators = data.macro_indicators;

                // Define important indicators to highlight
                const volatilityIndicators = ['VIX', 'SKEW', 'FearGreed'];
                const cryptoIndicators = ['BTC', 'ETH'];
                const yieldIndicators = ['YieldSpread', '2Y_Yield', '10Y_Yield'];

                for (const [name, values] of Object.entries(indicators)) {
                    const change = values.change_1d || 0;
                    const changeClass = change >= 0 ? 'text-green-400' : 'text-red-400';
                    const arrow = change >= 0 ? '▲' : '▼';

                    // Special styling for different categories
                    let boxClass = 'bg-[#1a1a1a] rounded p-2 text-center border border-[#2a2a2a]';
                    let labelClass = 'text-xs text-gray-500';

                    if (volatilityIndicators.includes(name)) {
                        // Purple glow for volatility
                        boxClass = 'bg-purple-900/30 rounded p-2 text-center border border-purple-600/50 shadow-lg shadow-purple-500/20';
                        labelClass = 'text-xs text-purple-300 font-bold';
                    } else if (cryptoIndicators.includes(name)) {
                        // Orange glow for crypto
                        boxClass = 'bg-orange-900/30 rounded p-2 text-center border border-orange-600/50 shadow-lg shadow-orange-500/20';
                        labelClass = 'text-xs text-orange-300 font-bold';
                    } else if (yieldIndicators.includes(name)) {
                        // Blue glow for yields
                        boxClass = 'bg-blue-900/30 rounded p-2 text-center border border-blue-600/50 shadow-lg shadow-blue-500/20';
                        labelClass = 'text-xs text-blue-300 font-bold';
                    }

                    const div = document.createElement('div');
                    div.className = boxClass;
                    div.innerHTML = `
                        <div class="${labelClass}">${name}</div>
                        <div class="text-sm font-bold text-white">${values.current ?? values.value ?? '-'}</div>
                        <div class="${changeClass} text-xs">${arrow} ${Math.abs(change).toFixed(2)}%</div>
                    `;
                    indicatorsEl.appendChild(div);
                }
            }

            // Render AI analysis
            if (analysisEl && data.ai_analysis) {
                // Convert markdown-like formatting
                let html = data.ai_analysis
                    .replace(/### (.*)/g, '<h3 class="text-purple-300 font-bold mt-3 mb-1">$1</h3>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
                    .replace(/🚀/g, '<span class="text-green-400">🚀</span>')
                    .replace(/⚠️/g, '<span class="text-yellow-400">⚠️</span>')
                    .replace(/\n/g, '<br>');
                analysisEl.innerHTML = html;
            }
        }

        function renderUSCalendar(data) {
            const container = document.getElementById('us-calendar-events');
            const rangeEl = document.getElementById('us-calendar-range');

            if (!container) return;

            // Set date range
            if (rangeEl && data.week_start && data.week_end) {
                rangeEl.textContent = `${data.week_start} ~ ${data.week_end}`;
            }

            container.innerHTML = '';

            if (!data.events || data.events.length === 0) {
                container.innerHTML = '<div class="text-center text-xs text-gray-500 py-2">No major events schedule for this week.</div>';
                return;
            }

            // Render events
            data.events.forEach(event => {
                let icon = '📅';
                let colorClass = 'text-gray-300';
                let bgClass = 'bg-[#2a2a2a]';

                if (event.type === 'Earnings') {
                    icon = '🏢';
                    colorClass = 'text-blue-300';
                    bgClass = 'bg-blue-900/20 border border-blue-800/30';
                } else if (event.impact === 'High') {
                    icon = '🔥';
                    colorClass = 'text-red-300';
                    bgClass = 'bg-red-900/20 border border-red-800/30';
                }

                const eventId = `cal-evt-${Math.random().toString(36).substr(2, 9)}`;

                const div = document.createElement('div');
                div.className = 'mb-2';
                div.innerHTML = `
                    <div class="flex items-center justify-between p-2 rounded text-xs ${bgClass} cursor-pointer hover:bg-opacity-80 transition-all" onclick="toggleCalendarDetails('${eventId}')">
                        <div class="flex items-center gap-2">
                            <span>${icon}</span>
                            <span class="font-medium ${colorClass}">${event.event}</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-gray-500">${event.date}</span>
                            <i class="fas fa-chevron-down text-[10px] text-gray-500 transition-transform" id="arrow-${eventId}"></i>
                        </div>
                    </div>
                    <div id="${eventId}" class="hidden bg-[#222] border-x border-b border-[#2a2a2a] rounded-b p-3 text-xs text-gray-400 space-y-2">
                        <div class="flex justify-between items-center border-b border-gray-700 pb-2 mb-2">
                            <span class="font-semibold text-gray-300">Detailed Info</span>
                            <span class="bg-gray-800 px-2 py-0.5 rounded text-[10px]">Impact: ${event.impact}</span>
                        </div>
                        <div class="leading-relaxed whitespace-pre-wrap">
                            ${event.description ? event.description.replace(/\*\*(.*?)\*\*/g, '<strong class="text-gray-200">$1</strong>').replace(/\\n/g, '<br>') : '<span class="text-gray-500">상세 정보가 없습니다.</span>'}
                        </div>
                        ${event.type === 'Earnings' ? `
                        <div class="mt-2 flex gap-2">
                            <button class="flex-1 bg-blue-900/30 hover:bg-blue-900/50 text-blue-300 py-1 rounded transition-colors text-center" onclick="loadUSAISummary('${event.event.split(' ')[0]}'); event.stopPropagation();">
                                🤖 AI Analysis
                            </button>
                            <button class="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-300 py-1 rounded transition-colors text-center" onclick="window.open('https://finance.yahoo.com/quote/${event.event.split(' ')[0]}', '_blank'); event.stopPropagation();">
                                📊 Yahoo Finance
                            </button>
                        </div>` : ''}
                    </div>
                `;
                container.appendChild(div);
            });
        }

        function toggleCalendarDetails(id) {
            const el = document.getElementById(id);
            const arrow = document.getElementById(`arrow-${id}`);
            if (el) {
                if (el.classList.contains('hidden')) {
                    el.classList.remove('hidden');
                    arrow.style.transform = 'rotate(180deg)';
                } else {
                    el.classList.add('hidden');
                    arrow.style.transform = 'rotate(0deg)';
                }
            }
        }

        function expandAllCalendarEvents() {
            document.querySelectorAll('[id^="cal-evt-"]').forEach(el => {
                el.classList.remove('hidden');
            });
            document.querySelectorAll('[id^="arrow-cal-evt-"]').forEach(arrow => {
                arrow.style.transform = 'rotate(180deg)';
            });
        }

        function collapseAllCalendarEvents() {
            document.querySelectorAll('[id^="cal-evt-"]').forEach(el => {
                el.classList.add('hidden');
            });
            document.querySelectorAll('[id^="arrow-cal-evt-"]').forEach(arrow => {
                arrow.style.transform = 'rotate(0deg)';
            });
        }

        function renderUSSectorHeatmap(data) {
            const container = document.getElementById('us-sector-heatmap');
            if (!container) return;

            // Should verify data structure
            if (!data.series) {
                // Fallback if old data format
                container.innerHTML = '<div class="flex items-center justify-center h-full text-xs text-gray-500">Data updating... please refresh later</div>';
                return;
            }

            // Update date if available
            const dateEl = document.getElementById('us-heatmap-date');
            if (dateEl && data.data_date) {
                dateEl.textContent = `기준: ${data.data_date}`;
            }

            // Clear loading
            container.innerHTML = '';

            // Map colors properly
            const seriesData = data.series.map(sector => ({
                name: sector.name,
                data: sector.data.map(stock => ({
                    x: stock.x,
                    y: stock.y,
                    fillColor: stock.color, // Use color from backend
                    // Add meta for tooltip
                    meta: {
                        change: stock.change,
                        price: stock.price
                    }
                }))
            }));

            const options = {
                series: seriesData,
                chart: {
                    type: 'treemap',
                    height: 300,
                    toolbar: { show: false },
                    fontFamily: 'Inter, sans-serif',
                    background: 'transparent'
                },
                legend: { show: false },
                colors: ['#3b82f6'], // Default color, will be overridden by fillColor
                plotOptions: {
                    treemap: {
                        distributed: true,
                        enableShades: false,
                        useFillColorAsStroke: true
                    }
                },
                dataLabels: {
                    enabled: true,
                    style: {
                        fontSize: '10px',
                        fontWeight: 'bold',
                        colors: ['#ffffff']
                    },
                    formatter: function (text, op) {
                        // Safely get change value
                        try {
                            const dataPoint = op.w.config.series[op.seriesIndex].data[op.dataPointIndex];
                            const change = dataPoint?.meta?.change;
                            if (change !== undefined) {
                                return [text, (change >= 0 ? '+' : '') + change.toFixed(1) + '%'];
                            }
                        } catch (e) { }
                        return text;
                    },
                    offsetY: -4
                },
                tooltip: {
                    theme: 'dark',
                    y: {
                        formatter: function (value, op) {
                            // Find the original data using dataPointIndex
                            const dataPoint = op.w.config.series[op.seriesIndex].data[op.dataPointIndex];
                            const change = dataPoint.meta.change;
                            const price = dataPoint.meta.price;
                            return `$${price} (${change > 0 ? '+' : ''}${change}%)`;
                        }
                    }
                },
                stroke: {
                    width: 1,
                    colors: ['#1a1a1a']
                }
            };

            const chart = new ApexCharts(container, options);
            chart.render();
        }

        function renderUSSmartMoneyPicks(data) {
            const table = document.getElementById('us-smart-money-table');
            const summary = document.getElementById('us-smart-money-summary');

            if (!table) return;
            table.innerHTML = '';

            // Store picks data for later use
            window.usSmartMoneyPicks = data.top_picks;

            // Update summary with analysis date
            if (summary) {
                const analysisDate = data.analysis_date || '';
                const avgScore = data.summary?.avg_score || 0;
                summary.innerHTML = `<span class="text-purple-400">📅 분석일: ${analysisDate}</span> | 평균: ${avgScore}점`;
            }

            // Render rows
            data.top_picks.forEach((pick, idx) => {
                const gradeClass = pick.grade?.includes('A급') ? 'text-yellow-400' :
                    pick.grade?.includes('B급') ? 'text-green-400' : 'text-gray-400';
                const upsideClass = pick.target_upside > 0 ? 'text-green-400' : 'text-red-400';

                const tr = document.createElement('tr');
                tr.className = 'border-b border-[#2a2a2a] hover:bg-[#252525] cursor-pointer';
                tr.setAttribute('data-ticker', pick.ticker);
                tr.setAttribute('data-idx', idx);

                // Performance change color
                const change = pick.change_since_rec || 0;
                const changeClass = change >= 0 ? 'text-green-400' : 'text-red-400';
                const changePrefix = change >= 0 ? '+' : '';

                // Score display
                const score = pick.final_score || pick.composite_score || 0;

                // AI recommendation
                const aiRec = pick.ai_recommendation || '분석 중';
                const recEmoji = aiRec.includes('적극') ? '🔥' : aiRec.includes('매수') ? '📈' : '📊';

                // Prices
                const priceAtRec = pick.price_at_rec || pick.price_at_analysis || pick.current_price || 0;
                const currentPrice = pick.current_price || priceAtRec;

                tr.innerHTML = `
                    <td class="p-2 text-gray-500">${pick.rank || idx + 1}</td>
                    <td class="p-2"><span class="text-gray-400 text-xs">${pick.name || ''}</span> <span class="text-white font-bold">(${pick.ticker})</span></td>
                    <td class="p-2 text-center"><span class="text-xs px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300">${pick.sector || '-'}</span></td>
                    <td class="p-2 text-center text-blue-400 font-bold">${score}</td>
                    <td class="p-2 text-center text-xs"><span class="text-yellow-400">${recEmoji} ${aiRec}</span></td>
                    <td class="p-2 text-center text-gray-300">$${priceAtRec.toFixed(2)}</td>
                    <td class="p-2 text-center text-white font-bold">$${currentPrice.toFixed(2)}</td>
                    <td class="p-2 text-center ${changeClass} font-bold">${changePrefix}${change.toFixed(2)}%</td>
                    <td class="p-2 text-center ${pick.target_upside > 0 ? 'text-green-400' : 'text-red-400'}">${pick.target_upside ? pick.target_upside.toFixed(1) + '%' : 'N/A'}</td>
                `;

                // Add click handler
                tr.addEventListener('click', () => loadUSStockChart(pick, idx));
                table.appendChild(tr);
            });

            // Auto-load first stock
            if (data.top_picks.length > 0) {
                loadUSStockChart(data.top_picks[0], 0);
            }
        }

        function populateUSHistoryDates(dates) {
            const select = document.getElementById('us-history-date-select');
            if (!select) return;

            // Keep first option
            select.innerHTML = '<option value="">📅 최신 분석</option>';

            dates.forEach(date => {
                const option = document.createElement('option');
                option.value = date;
                option.textContent = date;
                select.appendChild(option);
            });
        }

        async function loadUSHistoryByDate(date) {
            if (!date) {
                // Load current/latest data
                updateUSMarketDashboard();
                return;
            }

            try {
                const response = await fetch(`/api/us/history/${date}`);
                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                    return;
                }

                // Update summary with performance
                const summary = document.getElementById('us-smart-money-summary');
                if (summary && data.summary) {
                    const perf = data.summary.avg_performance || 0;
                    const perfClass = perf >= 0 ? 'text-green-400' : 'text-red-400';
                    const perfPrefix = perf >= 0 ? '+' : '';
                    summary.innerHTML = `<span class="text-purple-400">📅 ${date}</span> | <span class="${perfClass}">평균 ${perfPrefix}${perf}%</span>`;
                }

                // Render the historical picks
                renderUSSmartMoneyPicks(data);

            } catch (e) {
                console.error('Error loading history:', e);
            }
        }

        // US Stock Chart instance
        let usStockChart = null;
        let currentChartPick = null;
        let currentChartPeriod = '1y';

        // Period button click handlers
        document.getElementById('us-chart-period-btns')?.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') {
                const period = e.target.dataset.period;
                currentChartPeriod = period;

                // Update button styles
                document.querySelectorAll('#us-chart-period-btns button').forEach(btn => {
                    btn.className = 'px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white transition-colors';
                });
                e.target.className = 'px-2 py-1 text-xs rounded bg-blue-600 text-white';

                // Reload chart with new period
                if (currentChartPick) {
                    loadUSStockChart(currentChartPick, -1, period);
                }
            }
        });

        async function loadUSStockChart(pick, idx, period = null) {
            const chartContainer = document.getElementById('us-stock-chart');
            const tickerEl = document.getElementById('us-chart-ticker');
            const infoEl = document.getElementById('us-chart-info');
            const gradeEl = document.getElementById('us-chart-grade');

            if (!chartContainer) return;

            // Save current pick for period switching
            currentChartPick = pick;
            const usePeriod = period || currentChartPeriod;

            // Update header
            tickerEl.textContent = pick.ticker;
            const score = pick.final_score || pick.composite_score || 0;
            infoEl.textContent = `${pick.name || pick.ticker} | Score: ${score}/100`;

            const aiRec = pick.ai_recommendation || '분석 중';
            gradeEl.textContent = aiRec;
            gradeEl.className = `px-3 py-1 rounded text-sm font-bold ${aiRec.includes('적극') ? 'bg-yellow-600' : 'bg-green-600'} text-white`;

            // Update details panel
            updateUSStockDetails(pick);

            // Highlight selected row
            if (idx >= 0) {
                document.querySelectorAll('#us-smart-money-table tr').forEach(tr => {
                    tr.classList.remove('bg-[#2a2a2a]');
                });
                document.querySelector(`#us-smart-money-table tr[data-idx="${idx}"]`)?.classList.add('bg-[#2a2a2a]');
            }

            // Fetch and render chart
            try {
                chartContainer.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500"><i class="fas fa-spinner fa-spin text-2xl"></i></div>';

                const response = await fetch(`/api/us/stock-chart/${pick.ticker}?period=${usePeriod}`);
                const data = await response.json();

                if (data.error) {
                    chartContainer.innerHTML = `<div class="flex items-center justify-center h-full text-gray-500">${data.error}</div>`;
                    return;
                }

                // Clear container and create chart
                chartContainer.innerHTML = '';

                if (usStockChart) {
                    usStockChart.remove();
                }

                usStockChart = LightweightCharts.createChart(chartContainer, {
                    width: chartContainer.clientWidth,
                    height: 300,
                    layout: { background: { color: '#1a1a1a' }, textColor: '#999' },
                    grid: { vertLines: { color: '#2a2a2a' }, horzLines: { color: '#2a2a2a' } },
                    timeScale: {
                        borderColor: '#2a2a2a',
                        timeVisible: true,
                        rightOffset: 5,
                        fixLeftEdge: true,
                        fixRightEdge: true
                    },
                    rightPriceScale: { borderColor: '#2a2a2a' },
                    handleScroll: false,
                    handleScale: false
                });

                const candleSeries = usStockChart.addCandlestickSeries({
                    upColor: '#22c55e',
                    downColor: '#ef4444',
                    borderUpColor: '#22c55e',
                    borderDownColor: '#ef4444',
                    wickUpColor: '#22c55e',
                    wickDownColor: '#ef4444'
                });

                candleSeries.setData(data.candles);
                usStockChart.timeScale().fitContent();

                // Reset and reload indicators for new stock
                indicatorData = null;
                bbUpperSeries = null; bbMiddleSeries = null; bbLowerSeries = null;
                srLines = [];

                // Hide sub-charts
                document.getElementById('us-rsi-chart')?.classList.add('hidden');
                document.getElementById('us-macd-chart')?.classList.add('hidden');
                if (rsiChart) { rsiChart.remove(); rsiChart = null; }
                if (macdChart) { macdChart.remove(); macdChart = null; }

                // If any indicator is active, reload data and re-render
                if (indicatorState.bb || indicatorState.sr || indicatorState.rsi || indicatorState.macd) {
                    loadTechnicalIndicators(pick.ticker, usePeriod).then(() => {
                        if (indicatorState.bb) renderBollingerBands();
                        if (indicatorState.sr) renderSupportResistance();
                        if (indicatorState.rsi) renderRSI();
                        if (indicatorState.macd) renderMACD();
                    });
                }

                // Load AI Summary
                loadUSAISummary(pick.ticker);

            } catch (e) {
                console.error('Error loading US stock chart:', e);
                chartContainer.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">Failed to load chart</div>';
            }
        }

        // --- Technical Indicators ---
        let indicatorState = { bb: false, sr: false, rsi: false, macd: false };
        let indicatorData = null;
        let bbUpperSeries = null, bbMiddleSeries = null, bbLowerSeries = null;
        let srLines = [];
        let rsiChart = null, rsiSeries = null;
        let macdChart = null, macdLineSeries = null, macdSignalSeries = null, macdHistSeries = null;

        async function loadTechnicalIndicators(ticker, period = '1y') {
            try {
                const response = await fetch(`/api/us/technical-indicators/${ticker}?period=${period}`);
                indicatorData = await response.json();
                console.log('Loaded indicators for', ticker, indicatorData);
            } catch (e) {
                console.error('Error loading technical indicators:', e);
                indicatorData = null;
            }
        }

        function toggleIndicator(type) {
            indicatorState[type] = !indicatorState[type];
            const btn = document.getElementById(`toggle-${type}`);

            if (indicatorState[type]) {
                btn.classList.add('bg-purple-600', 'text-white');
                btn.classList.remove('bg-gray-700', 'text-gray-300');
            } else {
                btn.classList.remove('bg-purple-600', 'text-white');
                btn.classList.add('bg-gray-700', 'text-gray-300');
            }

            // Ensure indicator data is loaded
            if (!indicatorData && currentChartPick) {
                loadTechnicalIndicators(currentChartPick.ticker, currentChartPeriod).then(() => {
                    renderIndicator(type);
                });
            } else {
                renderIndicator(type);
            }
        }

        function renderIndicator(type) {
            if (!indicatorData || !usStockChart) return;

            if (type === 'bb') renderBollingerBands();
            else if (type === 'sr') renderSupportResistance();
            else if (type === 'rsi') renderRSI();
            else if (type === 'macd') renderMACD();
        }

        function renderBollingerBands() {
            // Remove existing BB series
            if (bbUpperSeries) { usStockChart.removeSeries(bbUpperSeries); bbUpperSeries = null; }
            if (bbMiddleSeries) { usStockChart.removeSeries(bbMiddleSeries); bbMiddleSeries = null; }
            if (bbLowerSeries) { usStockChart.removeSeries(bbLowerSeries); bbLowerSeries = null; }

            if (!indicatorState.bb || !indicatorData?.bollinger) return;

            bbUpperSeries = usStockChart.addLineSeries({ color: 'rgba(147, 51, 234, 0.5)', lineWidth: 1 });
            bbMiddleSeries = usStockChart.addLineSeries({ color: 'rgba(147, 51, 234, 0.8)', lineWidth: 1, lineStyle: 2 });
            bbLowerSeries = usStockChart.addLineSeries({ color: 'rgba(147, 51, 234, 0.5)', lineWidth: 1 });

            bbUpperSeries.setData(indicatorData.bollinger.upper);
            bbMiddleSeries.setData(indicatorData.bollinger.middle);
            bbLowerSeries.setData(indicatorData.bollinger.lower);
        }

        function renderSupportResistance() {
            // Remove existing S/R lines
            srLines.forEach(line => usStockChart.removeSeries(line));
            srLines = [];

            if (!indicatorState.sr || !indicatorData?.support_resistance) return;

            const sr = indicatorData.support_resistance;

            // Draw support lines (green)
            sr.support?.forEach(level => {
                const line = usStockChart.addLineSeries({
                    color: '#22c55e',
                    lineWidth: 1,
                    lineStyle: 2,
                    priceLineVisible: true,
                    lastValueVisible: true
                });
                // Create horizontal line data
                if (indicatorData.rsi?.length > 0) {
                    const times = indicatorData.rsi;
                    line.setData([
                        { time: times[0].time, value: level },
                        { time: times[times.length - 1].time, value: level }
                    ]);
                }
                srLines.push(line);
            });

            // Draw resistance lines (red)  
            sr.resistance?.forEach(level => {
                const line = usStockChart.addLineSeries({
                    color: '#ef4444',
                    lineWidth: 1,
                    lineStyle: 2,
                    priceLineVisible: true,
                    lastValueVisible: true
                });
                if (indicatorData.rsi?.length > 0) {
                    const times = indicatorData.rsi;
                    line.setData([
                        { time: times[0].time, value: level },
                        { time: times[times.length - 1].time, value: level }
                    ]);
                }
                srLines.push(line);
            });
        }

        function renderRSI() {
            const container = document.getElementById('us-rsi-chart');

            if (!indicatorState.rsi) {
                container.classList.add('hidden');
                if (rsiChart) { rsiChart.remove(); rsiChart = null; }
                return;
            }

            container.classList.remove('hidden');
            container.innerHTML = '';

            rsiChart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 80,
                layout: { background: { color: '#1a1a1a' }, textColor: '#666' },
                grid: { vertLines: { color: '#2a2a2a' }, horzLines: { color: '#2a2a2a' } },
                timeScale: { visible: false },
                rightPriceScale: { borderColor: '#2a2a2a', scaleMargins: { top: 0.1, bottom: 0.1 } },
                handleScroll: false, handleScale: false
            });

            rsiSeries = rsiChart.addLineSeries({ color: '#22c55e', lineWidth: 1 });
            rsiSeries.setData(indicatorData.rsi || []);

            // Add overbought/oversold lines
            const ovLine = rsiChart.addLineSeries({ color: '#ef4444', lineWidth: 1, lineStyle: 2 });
            const osLine = rsiChart.addLineSeries({ color: '#22c55e', lineWidth: 1, lineStyle: 2 });

            if (indicatorData.rsi?.length > 0) {
                const times = indicatorData.rsi;
                ovLine.setData([{ time: times[0].time, value: 70 }, { time: times[times.length - 1].time, value: 70 }]);
                osLine.setData([{ time: times[0].time, value: 30 }, { time: times[times.length - 1].time, value: 30 }]);
            }

            rsiChart.timeScale().fitContent();
        }

        function renderMACD() {
            const container = document.getElementById('us-macd-chart');

            if (!indicatorState.macd) {
                container.classList.add('hidden');
                if (macdChart) { macdChart.remove(); macdChart = null; }
                return;
            }

            container.classList.remove('hidden');
            container.innerHTML = '';

            macdChart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 80,
                layout: { background: { color: '#1a1a1a' }, textColor: '#666' },
                grid: { vertLines: { color: '#2a2a2a' }, horzLines: { color: '#2a2a2a' } },
                timeScale: { visible: false },
                rightPriceScale: { borderColor: '#2a2a2a' },
                handleScroll: false, handleScale: false
            });

            macdLineSeries = macdChart.addLineSeries({ color: '#3b82f6', lineWidth: 1 });
            macdSignalSeries = macdChart.addLineSeries({ color: '#f59e0b', lineWidth: 1 });
            macdHistSeries = macdChart.addHistogramSeries({
                color: '#22c55e',
                priceFormat: { type: 'volume' },
                priceScaleId: ''
            });
            macdChart.priceScale('').applyOptions({ scaleMargins: { top: 0.7, bottom: 0 } });

            macdLineSeries.setData(indicatorData.macd?.macd_line || []);
            macdSignalSeries.setData(indicatorData.macd?.signal_line || []);

            // Color histogram based on value
            const histData = (indicatorData.macd?.histogram || []).map(d => ({
                time: d.time,
                value: d.value,
                color: d.value >= 0 ? '#22c55e' : '#ef4444'
            }));
            macdHistSeries.setData(histData);

            macdChart.timeScale().fitContent();
        }

        async function loadUSAISummary(ticker) {
            const summaryEl = document.getElementById('us-ai-summary');
            const updatedEl = document.getElementById('us-ai-updated');

            if (!summaryEl) return;

            const loadingText = currentLang === 'en' ? 'Loading AI analysis...' : 'AI 분석 로딩 중...';
            summaryEl.innerHTML = `<span class="text-gray-500"><i class="fas fa-spinner fa-spin"></i> ${loadingText}</span>`;

            try {
                const response = await fetch(`/api/us/ai-summary/${ticker}?lang=${currentLang}`);
                const data = await response.json();

                if (data.error) {
                    summaryEl.innerHTML = `<span class="text-gray-500">${data.error}</span>`;
                    return;
                }

                // Convert markdown-style text to HTML
                let htmlSummary = data.summary
                    .replace(/## (.*)/g, '<strong class="text-white block mb-1">$1</strong>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
                    .replace(/\n/g, '<br>');

                summaryEl.innerHTML = htmlSummary;

                if (updatedEl && data.updated) {
                    const date = new Date(data.updated);
                    updatedEl.textContent = `Updated: ${date.toLocaleDateString()}`;
                }

            } catch (e) {
                console.error('Error loading AI summary:', e);
                summaryEl.innerHTML = '<span class="text-gray-500">AI 분석을 불러올 수 없습니다.</span>';
            }
        }

        function updateUSStockDetails(pick) {
            // Stock Details section was replaced with Sector Heatmap
            // No need to update individual stock details anymore
        }

        function renderUSETFFlows(data) {
            const inflowsContainer = document.getElementById('us-etf-inflows');
            const outflowsContainer = document.getElementById('us-etf-outflows');
            const sentimentEl = document.getElementById('us-etf-sentiment');

            // Update sentiment badge
            if (sentimentEl && data.market_sentiment_score) {
                const score = data.market_sentiment_score;
                let sentiment, bgColor;
                if (score >= 60) { sentiment = 'Bullish'; bgColor = 'bg-green-600'; }
                else if (score >= 50) { sentiment = 'Neutral'; bgColor = 'bg-gray-600'; }
                else { sentiment = 'Bearish'; bgColor = 'bg-red-600'; }
                sentimentEl.textContent = `${sentiment} (${score})`;
                sentimentEl.className = `text-xs px-2 py-1 rounded ${bgColor} text-white`;
            }

            // Render inflows
            if (inflowsContainer && data.top_inflows) {
                inflowsContainer.innerHTML = '';
                data.top_inflows.forEach(etf => {
                    const div = document.createElement('div');
                    div.className = 'flex justify-between items-center text-sm';
                    div.innerHTML = `
                        <span class="text-white">${etf.ticker} <span class="text-gray-500 text-xs">${etf.name}</span></span>
                        <span class="text-green-400 font-bold">${etf.flow_score}</span>
                    `;
                    inflowsContainer.appendChild(div);
                });
            }

            // Render outflows
            if (outflowsContainer && data.top_outflows) {
                outflowsContainer.innerHTML = '';
                data.top_outflows.forEach(etf => {
                    const div = document.createElement('div');
                    div.className = 'flex justify-between items-center text-sm';
                    div.innerHTML = `
                        <span class="text-white">${etf.ticker} <span class="text-gray-500 text-xs">${etf.name}</span></span>
                        <span class="text-red-400 font-bold">${etf.flow_score}</span>
                    `;
                    outflowsContainer.appendChild(div);
                });
            }

            // AI Analysis Button Handler
            const aiBtn = document.getElementById('us-etf-ai-btn');
            const aiContainer = document.getElementById('us-etf-ai-container');
            const aiContent = document.getElementById('us-etf-ai-content');

            if (data.ai_analysis && aiBtn && aiContainer && aiContent) {
                aiBtn.classList.remove('hidden');

                // Remove existing listeners to prevent duplicates
                const newBtn = aiBtn.cloneNode(true);
                aiBtn.parentNode.replaceChild(newBtn, aiBtn);

                newBtn.addEventListener('click', () => {
                    aiContainer.classList.toggle('hidden');
                    // Use textContent for safety, or innerHTML if we trust the source (we generated it)
                    // Since Gemini might generate bolding (**text**), we should ideally parse markdown.
                    // But for now, textContent is safer. Or simple replacement for bold.
                    aiContent.textContent = data.ai_analysis;
                });
            } else if (aiBtn) {
                aiBtn.classList.add('hidden');
            }
        }

        function renderUSMarketIndices(indices) {
            const container = document.getElementById('us-market-indices-container');
            if (!container) return;
            container.innerHTML = '';

            if (!indices || indices.length === 0) {
                container.innerHTML = '<div class="col-span-full text-center text-gray-500 text-sm">Loading market data...</div>';
                return;
            }

            indices.forEach(idx => {
                // US Color Standard: Green = Up, Red = Down
                const colorClass = idx.color === 'green' ? 'text-[#00E396]' : (idx.color === 'red' ? 'text-[#FF4560]' : 'text-gray-400');
                const sign = idx.change_pct > 0 ? '+' : '';

                const div = document.createElement('div');
                div.className = 'bg-[#1a1a1a] border border-[#2a2a2a] rounded p-3 flex flex-col items-center justify-center hover:bg-[#252525] transition-colors';
                div.innerHTML = `
                    <span class="text-xs text-gray-400 mb-1">${idx.name}</span>
                    <span class="text-lg font-bold text-white mb-1">${idx.price}</span>
                    <span class="text-xs font-medium ${colorClass}">${idx.change} (${sign}${idx.change_pct.toFixed(2)}%)</span>
                `;
                container.appendChild(div);
            });
        }

        document.addEventListener('DOMContentLoaded', async () => {
            try {
                // Set date picker max to today
                const today = new Date().toISOString().split('T')[0];
                document.getElementById('history-date').max = today;

                // Date picker event listener
                document.getElementById('history-date').addEventListener('change', (e) => {
                    selectedDate = e.target.value;
                    updateDashboard();
                });

                // Start real-time price updates
                priceUpdateInterval = setInterval(updateRealtimePrices, 20000); // 20 seconds

                updateDashboard(); // Initial dashboard load
            } catch (e) {
                console.error("Initialization Error:", e);
            }
        });

        function resetDate() {
            document.getElementById('history-date').value = '';
            selectedDate = null;
            updateDashboard();
        }

        // --- Dashboard Update Logic ---
        async function updateDashboard() {
            try {
                let url = '/api/portfolio';
                if (selectedDate) {
                    url += `?date=${selectedDate}`;
                }

                const response = await fetch(url);
                const data = await response.json();

                if (data.error) {
                    console.error(data.error);
                    if (selectedDate) alert("No data found for this date.");
                    return;
                }

                // Set default date if not set
                if (!selectedDate && data.latest_date) {
                    const dateInput = document.getElementById('history-date');
                    if (dateInput && !dateInput.value) {
                        dateInput.value = data.latest_date;
                        // Don't trigger change event to avoid loop, just visual indication
                    }
                }

                // Update Key Stats (only if not in history mode, or partial update?)
                // For now, let's update everything returned
                if (data.key_stats) {
                    // Update Market Indices
                    if (data.market_indices) {
                        renderMarketIndices(data.market_indices);
                    }

                    // renderHoldingsChart(data.holdings_distribution); // Removed
                    renderHoldingsTable(data.top_holdings);
                    renderStyleBox(data.style_box);

                    // Load initial chart for the top holding
                    if (data.top_holdings && data.top_holdings.length > 0) {
                        updateSummaryChart(data.top_holdings[0].ticker);
                    }

                    if (data.performance) {
                        renderPerformanceTable(data.performance);
                    }
                }
            } catch (e) {
                console.error("Error updating dashboard:", e);
            }
        }

        function renderMarketIndices(indices) {
            const container = document.getElementById('market-indices-container');
            if (!container) return;
            container.innerHTML = '';

            if (!indices || indices.length === 0) {
                container.innerHTML = '<div class="col-span-full text-center text-gray-500 text-sm">No market data available</div>';
                return;
            }

            indices.forEach(idx => {
                // Korean Color Standard: Red = Up, Blue = Down
                const colorClass = idx.color === 'red' ? 'text-[#FF4560]' : (idx.color === 'blue' ? 'text-[#2962FF]' : 'text-gray-400');
                const sign = idx.change_pct > 0 ? '+' : '';

                const div = document.createElement('div');
                div.className = 'bg-[#1a1a1a] border border-[#2a2a2a] rounded p-3 flex flex-col items-center justify-center hover:bg-[#252525] transition-colors';
                div.innerHTML = `
                    <span class="text-xs text-gray-400 mb-1">${idx.name}</span>
                    <span class="text-lg font-bold text-white mb-1">${idx.price}</span>
                    <span class="text-xs font-medium ${colorClass}">${sign}${idx.change} (${sign}${idx.change_pct.toFixed(2)}%)</span>
                `;
                container.appendChild(div);
            });
        }

        let summaryChart;
        let summaryCandleSeries;
        let summaryVolumeSeries;
        let chartResizeObserver = null;
        let currentChartData = { candles: [], volumes: [] }; // Store full data
        let currentRange = '1Y'; // Default range

        function setChartRange(range) {
            currentRange = range;

            // Update button styles
            document.querySelectorAll('.chart-range-btn').forEach(btn => {
                if (btn.dataset.range === range) {
                    btn.classList.remove('text-gray-400', 'hover:text-white', 'hover:bg-[#333]');
                    btn.classList.add('text-white', 'bg-[#333]');
                } else {
                    btn.classList.add('text-gray-400', 'hover:text-white', 'hover:bg-[#333]');
                    btn.classList.remove('text-white', 'bg-[#333]');
                }
            });

            if (summaryChart && currentChartData.candles.length > 0) {
                const filtered = filterDataByRange(currentChartData.candles, currentChartData.volumes, range);
                summaryCandleSeries.setData(filtered.candles);
                summaryVolumeSeries.setData(filtered.volumes);
                summaryChart.timeScale().fitContent();
            }
        }

        function filterDataByRange(candles, volumes, range) {
            if (!candles || candles.length === 0) return { candles: [], volumes: [] };

            if (range === 'All') return { candles, volumes };

            const lastCandleTime = candles[candles.length - 1].time;
            const lastDate = new Date(lastCandleTime);
            let startDate = new Date(lastDate);

            switch (range) {
                case '1D': startDate.setDate(lastDate.getDate() - 1); break; // Logic might need refinement for trading days
                case '1W': startDate.setDate(lastDate.getDate() - 7); break;
                case '1M': startDate.setMonth(lastDate.getMonth() - 1); break;
                case '3M': startDate.setMonth(lastDate.getMonth() - 3); break;
                case '6M': startDate.setMonth(lastDate.getMonth() - 6); break;
                case '1Y': startDate.setFullYear(lastDate.getFullYear() - 1); break;
                case '5Y': startDate.setFullYear(lastDate.getFullYear() - 5); break;
            }

            const startTime = startDate.getTime();

            const filteredCandles = candles.filter(d => new Date(d.time).getTime() >= startTime);
            const filteredVolumes = volumes.filter(d => new Date(d.time).getTime() >= startTime);

            return { candles: filteredCandles, volumes: filteredVolumes };
        }

        async function updateSummaryChart(ticker) {
            const container = document.getElementById('summary-chart-container');
            if (!container) return;

            // Cleanup previous observer
            if (chartResizeObserver) {
                chartResizeObserver.disconnect();
                chartResizeObserver = null;
            }

            // Cleanup previous chart
            if (summaryChart) {
                summaryChart.remove();
                summaryChart = null;
            }

            // Show loading state
            container.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-gray-500">Loading chart...</div>';
            document.getElementById('summary-chart-ticker').innerText = ticker;

            try {
                const response = await fetch(`/api/stock/${ticker}`);
                const data = await response.json();

                // Clear loading state
                container.innerHTML = '';

                if (data.error) {
                    container.innerHTML = `<div class="absolute inset-0 flex items-center justify-center text-red-500">Error: ${data.error}</div>`;
                    return;
                }

                // Update Title
                const name = data.metrics && data.metrics.name ? data.metrics.name : '';
                document.getElementById('summary-chart-ticker').innerText = name ? `${name} (${ticker})` : ticker;

                // Check for data keys
                const priceData = data.candles || data.price_history;

                if (!priceData || priceData.length === 0) {
                    container.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-gray-500">No price data available</div>';
                    return;
                }

                // Store full data
                currentChartData.candles = priceData.map(d => ({
                    time: d.time, open: d.open, high: d.high, low: d.low, close: d.close
                }));
                currentChartData.volumes = priceData.map(d => ({
                    time: d.time, value: d.volume, color: d.close >= d.open ? 'rgba(239, 68, 68, 0.5)' : 'rgba(59, 130, 246, 0.5)'
                }));

                // Create Chart with Restricted Interaction
                summaryChart = LightweightCharts.createChart(container, {
                    layout: { background: { color: '#121212' }, textColor: '#D1D5DB' },
                    grid: { vertLines: { color: '#2a2a2a' }, horzLines: { color: '#2a2a2a' } },
                    width: container.clientWidth,
                    height: container.clientHeight,
                    timeScale: { timeVisible: true, secondsVisible: false },
                    handleScroll: false, // Disable scroll
                    handleScale: false,  // Disable zoom
                });

                summaryCandleSeries = summaryChart.addCandlestickSeries({
                    upColor: '#EF4444', downColor: '#3B82F6', borderVisible: false, wickUpColor: '#EF4444', wickDownColor: '#3B82F6',
                });

                summaryVolumeSeries = summaryChart.addHistogramSeries({
                    color: '#26a69a',
                    priceFormat: { type: 'volume' },
                    priceScaleId: '',
                });

                summaryChart.priceScale('').applyOptions({
                    scaleMargins: { top: 0.8, bottom: 0 },
                });

                // Handle resize
                chartResizeObserver = new ResizeObserver(entries => {
                    if (entries.length === 0 || !entries[0].contentRect) return;
                    const newRect = entries[0].contentRect;
                    if (summaryChart) {
                        summaryChart.applyOptions({ width: newRect.width, height: newRect.height });
                    }
                });
                chartResizeObserver.observe(container);

                // Apply Initial Filter
                setChartRange(currentRange);

            } catch (e) {
                console.error("Error updating summary chart:", e);
                container.innerHTML = `<div class="absolute inset-0 flex items-center justify-center text-red-500">Error loading chart</div>`;
            }
        }

        function renderHoldingsTable(holdings) {
            const tbody = document.getElementById('holdings-table-body');
            tbody.innerHTML = '';

            holdings.forEach(stock => {
                const tr = document.createElement('tr');
                tr.className = 'border-b border-gray-800 hover:bg-gray-800 transition-colors cursor-pointer';
                tr.onclick = () => updateSummaryChart(stock.ticker);

                // Determine color for YTD
                const ytdVal = typeof stock.ytd === 'number' ? stock.ytd : parseFloat(stock.ytd || 0);
                const ytdColor = ytdVal >= 0 ? 'text-red-400' : 'text-blue-400';
                const ytdSign = ytdVal >= 0 ? '+' : '';
                const ytdDisplay = typeof stock.ytd === 'number' ? stock.ytd.toFixed(1) : stock.ytd;

                // Determine color for Price vs Rec. Price
                const currentPrice = stock.price || 0;
                const recPrice = stock.recommendation_price || 0;
                let priceColor = 'text-gray-300';
                if (currentPrice > recPrice) priceColor = 'text-red-400';
                else if (currentPrice < recPrice) priceColor = 'text-blue-400';

                // Determine color for Return
                const returnPct = stock.return_pct || 0;
                const returnColor = returnPct > 0 ? 'text-red-400' : (returnPct < 0 ? 'text-blue-400' : 'text-gray-400');
                const returnSign = returnPct > 0 ? '+' : '';

                tr.innerHTML = `
                    <td class="py-3 text-left font-mono text-blue-400">${stock.ticker}</td>
                    <td class="py-3 text-left font-medium text-white">${stock.name}</td>
                    <td class="py-3 text-left font-mono text-gray-400">${recPrice.toLocaleString()}</td>
                    <td class="py-3 text-left font-mono ${priceColor}">${currentPrice.toLocaleString()}</td>
                    <td class="py-3 text-left font-mono ${returnColor}">${returnSign}${returnPct.toFixed(2)}%</td>
                    <td class="py-3 text-center font-bold text-yellow-400">${stock.score ? stock.score.toFixed(1) : '-'}</td>
                    <td class="py-3 text-center text-sm text-gray-300">${stock.grade || '-'}</td>
                    <td class="py-3 text-center text-sm text-gray-300">${stock.wave || '-'}</td>
                    <td class="py-3 text-center text-sm text-gray-300">${stock.sd_stage || '-'}</td>
                    <td class="py-3 text-center text-sm text-gray-300">${stock.inst_trend || '-'}</td>
                    <td class="py-3 text-center font-mono ${ytdColor}">${ytdSign}${ytdDisplay}%</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function renderPerformanceTable(performance) {
            const tbody = document.getElementById('performance-table-body');
            if (!tbody) return; // Guard clause if element doesn't exist
            tbody.innerHTML = '';

            if (!performance || performance.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-gray-500">No performance data available yet.</td></tr>';
                return;
            }

            performance.forEach(item => {
                const tr = document.createElement('tr');
                tr.className = 'border-b border-gray-800 hover:bg-gray-800 transition-colors';

                const returnVal = parseFloat(item.return.replace('%', ''));
                const returnClass = returnVal >= 0 ? 'text-green-400' : 'text-red-400';

                tr.innerHTML = `
                    <td class="py-3 font-mono text-gray-400" > ${item.ticker}</td >
                    <td class="py-3 font-medium text-white">${item.name}</td>
                    <td class="py-3 text-gray-300">${item.date}</td>
                    <td class="py-3 text-gray-300">${item.days}</td>
                    <td class="py-3 text-right ${returnClass}">${item.return}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function renderStyleBox(data) {
            // Helper to set text and adjust background opacity based on value
            const setCell = (id, val) => {
                const el = document.getElementById(id);
                if (el) {
                    el.innerText = val + '%';
                    // Dynamic opacity for green background based on value (max ~30%)
                    // Base color is #86bf8e (Green). We'll use rgba for simplicity or just keep the static classes for now.
                    // For this demo, we'll keep the static colors from the HTML template but update text.
                }
            };

            setCell('sb-large-value', data.large_value);
            setCell('sb-large-core', data.large_core);
            setCell('sb-large-growth', data.large_growth);
            setCell('sb-mid-value', data.mid_value);
            setCell('sb-mid-core', data.mid_core);
            setCell('sb-mid-growth', data.mid_growth);
            setCell('sb-small-value', data.small_value);
            setCell('sb-small-core', data.small_core);
            setCell('sb-small-growth', data.small_growth);
        }

        // Real-time Price Updates
        async function updateRealtimePrices() {
            // Collect tickers from the table
            const rows = document.querySelectorAll('#holdings-table-body tr');
            const tickers = [];
            const tickerRowMap = {}; // ticker -> row element

            rows.forEach(row => {
                const tickerCell = row.querySelector('td:first-child');
                if (tickerCell) {
                    const ticker = tickerCell.innerText.trim();
                    tickers.push(ticker);
                    tickerRowMap[ticker] = row;
                }
            });

            if (tickers.length === 0) return;

            try {
                const response = await fetch('/api/realtime-prices', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ tickers: tickers })
                });

                const prices = await response.json();

                // Update table and chart
                for (const [ticker, data] of Object.entries(prices)) {
                    // Handle both simple float (legacy/fallback) and object (new OHLC)
                    let currentPrice = 0;
                    let ohlc = null;

                    if (typeof data === 'number') {
                        currentPrice = data;
                    } else {
                        currentPrice = data.current;
                        ohlc = data;
                    }

                    // 1. Update Table
                    const row = tickerRowMap[ticker];
                    if (row) {
                        const recPriceCell = row.querySelector('td:nth-child(3)');
                        const priceCell = row.querySelector('td:nth-child(4)');

                        if (priceCell && recPriceCell) {
                            const oldPrice = parseFloat(priceCell.innerText.replace(/,/g, ''));
                            const recPrice = parseFloat(recPriceCell.innerText.replace(/,/g, ''));

                            if (oldPrice !== currentPrice) {
                                priceCell.innerText = currentPrice.toLocaleString();

                                // Determine base color based on Rec. Price comparison
                                let baseColor = 'text-gray-300';
                                if (currentPrice > recPrice) baseColor = 'text-red-400';
                                else if (currentPrice < recPrice) baseColor = 'text-blue-400';

                                // Flash effect (Green/Red for up/down tick, then settle to baseColor)
                                const flashColor = currentPrice > oldPrice ? 'text-green-400' : 'text-red-400';

                                priceCell.className = `py-3 text-left font-mono ${flashColor} transition-colors duration-500`;
                                setTimeout(() => {
                                    priceCell.className = `py-3 text-left font-mono ${baseColor} transition-colors duration-500`;
                                }, 1000);
                            }
                        }
                    }

                    // 2. Update Chart if it's the current ticker
                    if (ticker === currentChartTicker && ohlc && summaryCandleSeries) {
                        // Update the last candle
                        // Lightweight charts update() method updates the last bar or adds a new one
                        summaryCandleSeries.update({
                            time: ohlc.date,
                            open: ohlc.open,
                            high: ohlc.high,
                            low: ohlc.low,
                            close: ohlc.current
                        });
                        // Also update volume series if it exists
                        if (summaryVolumeSeries) {
                            summaryVolumeSeries.update({
                                time: ohlc.date,
                                value: ohlc.volume,
                                color: ohlc.current >= ohlc.open ? 'rgba(239, 68, 68, 0.5)' : 'rgba(59, 130, 246, 0.5)'
                            });
                        }
                        console.log(`Chart updated for ${ticker}:`, ohlc.current);
                    }
                }
                console.log("Updated prices for", Object.keys(prices).length, "stocks");

            } catch (e) {
                console.error("Error updating realtime prices:", e);
            }
        }

        // Start polling every 60 seconds
        setInterval(updateRealtimePrices, 60000);

        // Initial call after a short delay to ensure table is rendered
        setTimeout(updateRealtimePrices, 5000);

    </script>
</body>

</html>
```
