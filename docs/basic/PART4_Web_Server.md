# Improved US Stock Analysis System Blueprint - Part 4: Web Server (현재 코드 기준)

이 문서는 실제 구현(`flask_app.py`)에 맞춘 요약본입니다. 상세 로직은 해당 파일이 소스 오브 트루스입니다.

---

## 핵심 동작 요약

- 실시간/스냅샷 데이터는 FMP API를 통해 조회합니다.
- SQLite(`DATA_DIR/gyuant.db`)에 저장된 문서/스마트머니 결과를 우선 사용합니다.
- PostgreSQL은 시계열 조회에 우선 사용되며, 필요 시 CSV/JSON으로 폴백합니다.
- AI 요약은 캐시/TTL을 적용해 응답을 안정화합니다.

---

## 주요 엔드포인트 (현재 구현)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | 대시보드 렌더링 |
| `/health` | GET | 헬스 체크 |
| `/status` | GET | 모듈 상태/계약 체크 |
| `/api/us/portfolio` | GET | 미국 지수/원자재/크립토 스냅샷 |
| `/api/us/smart-money` | GET | 스마트머니 상위 픽 |
| `/api/us/etf-flows` | GET | ETF 흐름 결과 |
| `/api/us/stock-chart/<ticker>` | GET | 종목 OHLC 시계열 |
| `/api/us/technical-indicators/<ticker>` | GET | RSI/MACD/BB 등 기술지표 |
| `/api/us/history-dates` | GET | 히스토리 스냅샷 날짜 목록 |
| `/api/us/history/<date>` | GET | 특정 날짜 스냅샷 |
| `/api/us/macro-analysis` | GET | 매크로 지표 + AI 요약 |
| `/api/us/sector-heatmap` | GET | 섹터 히트맵 데이터 |
| `/api/us/options-flow` | GET | 옵션 플로우 데이터 |
| `/options/unusual` | GET | 종목별 unusual trades |
| `/api/us/ai-summary/<ticker>` | GET | 종목별 AI 요약 |
| `/api/realtime-prices` | POST | 배치 실시간 가격 업데이트 |
| `/api/us/calendar` | GET | 주간 경제 캘린더 |

추가 페이지:
- `/top-movers`, `/sectors/heatmap`, `/sectors/<sector>/movers`

---

## 데이터 소스/우선순위

1. SQLite (`market_documents`, `market_smart_money_*`)
2. PostgreSQL (`market.daily_prices`, `factors.volume_analysis`)
3. CSV/JSON 폴백 (`DATA_DIR` 하위 산출물)

---

## 참고

- `options_flow`만 yfinance에 의존합니다.
- `model=gpt` 요청은 현재 Gemini 결과로 폴백됩니다.
