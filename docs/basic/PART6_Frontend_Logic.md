# Improved US Stock Analysis System Blueprint - Part 6: Frontend Logic (현재 코드 기준)

이 문서는 실제 JS 구성(`templates/partials/scripts_*.html`)에 맞춘 요약본입니다.

---

## 6.1 JS 파일 구성

| 파일 | 역할 |
|---|---|
| `scripts_base.html` | 상태 관리, i18n 메시지, 공통 유틸 |
| `scripts_us_fetch.html` | 데이터 fetch + 렌더링 오케스트레이션 |
| `scripts_us_charts.html` | 차트 렌더링/지표 토글 |
| `scripts_ai_etf.html` | ETF AI 인사이트 렌더링 |

---

## 6.2 핵심 흐름

- `updateUSMarketDashboard()`가 대시보드 주요 섹션을 병렬로 로드합니다.
- `updateRealtimePrices()`가 테이블 가격을 배치로 갱신합니다.
- `loadUSStockChart()`가 종목 클릭 시 차트/요약을 갱신합니다.
- `reloadMacroAnalysis()`가 매크로 섹션을 주기적으로 갱신합니다.

---

## 6.3 주요 API 호출

| Endpoint | Method | 용도 |
|---|---|---|
| `/api/us/portfolio` | GET | 지수/매크로 스냅샷 |
| `/api/us/smart-money` | GET | 스마트머니 테이블 |
| `/api/us/etf-flows` | GET | ETF 흐름 |
| `/api/us/history-dates` | GET | 히스토리 날짜 목록 |
| `/api/us/stock-chart/<ticker>` | GET | 차트 OHLC |
| `/api/us/technical-indicators/<ticker>` | GET | RSI/MACD/BB |
| `/api/us/macro-analysis` | GET | 매크로 + AI |
| `/api/us/sector-heatmap` | GET | 섹터 히트맵 |
| `/api/us/options-flow` | GET | 옵션 플로우 |
| `/api/us/calendar` | GET | 경제 캘린더 |
| `/api/us/ai-summary/<ticker>` | GET | 종목 AI 요약 |
| `/api/realtime-prices` | POST | 배치 실시간 가격 |

---

## 6.4 모델 선택(Gemini/GPT)

UI에서 `model=gpt` 선택이 가능하지만, 백엔드는 현재 Gemini 결과로 폴백합니다.
