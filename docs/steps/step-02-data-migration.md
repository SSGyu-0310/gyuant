# Step 2: Data Migration (yfinance -> FMP 전체 전환)

## 목표
yfinance 의존을 제거하고, 모든 실데이터 수집 흐름을 FMP 기반으로 통일한다. 기존 CSV/JSON 산출물의 스키마는 유지한다.

## 범위
- 포함: `flask_app.py` + `us_market/` 파이프라인의 yfinance 호출 전부 교체.
- 제외: 백테스트 데이터 구조 설계(이는 Step 3).

## 작업 지침 (에이전트용)
### 1) Flask API 실시간/단기 데이터 교체
- `flask_app.py:get_sector` -> FMP `profile`
- `flask_app.py:fetch_price_map` -> FMP `quote` 또는 `batch-quote`
- `flask_app.py:get_us_stock_chart` -> FMP `historical-price-full`
- `flask_app.py:get_technical_indicators` -> FMP `historical-price-full`
- `flask_app.py:get_realtime_prices` -> FMP `historical-chart/1min` 또는 `quote`
- `flask_app.py:get_us_macro_analysis` -> FMP `quote` + `treasury-rates`

요구사항:
- 기존 응답 JSON 스키마 유지
- 오류 시 빈 데이터/에러 구조 유지

### 2) 파이프라인 스크립트 교체 (`us_market/`)
- `create_us_daily_prices.py`: `historical-price-full/{symbol}` 사용
- `sector_heatmap.py`: `batch-quote` + 최근 2~5일 `historical-price-full`
- `analyze_etf_flows.py`: `historical-price-full` (90일)
- `portfolio_risk.py`: `historical-price-full` (6개월)
- `macro_analyzer.py`: `quote` + `treasury-rates`
- `smart_money_screener_v2.py`: `profile`, `key-metrics-ttm`, `ratios-ttm`, `ratings-snapshot`
- `analyze_13f.py`: `institutional-ownership` + `insider-trading`
- `insider_tracker.py`: `insider-trading`
- `options_flow.py`: FMP에 옵션 체인 문서가 없으므로
  - 유지(yfinance) 또는
  - 대체 제공자 사용

### 3) 심볼 매핑 정리
- Yahoo 심볼과 FMP 심볼을 분리 관리
- 예시 매핑:
  - `BTC-USD` -> `BTCUSD`
  - `GC=F` -> `GCUSD`
  - `CL=F` -> `CLUSD`
  - `KRW=X` -> `USDKRW` (FMP 리스트로 확인)
  - 지수는 `index-list` 기반 확인

### 4) 데이터 계약 유지
- 기존 CSV/JSON 컬럼명을 변경하지 않는다.
- `us_daily_prices.csv`: `ticker,date,open,high,low,current_price,volume,change,change_rate`
- `smart_money_picks_v2.csv`, `us_etf_flows.csv` 등 기존 스키마 유지

### 5) 테스트/의존성 정리
- `tests/conftest.py`의 `mock_yfinance`를 FMP mock으로 교체
- yfinance 제거:
  - `requirements.txt`
  - `us_market/requirements.txt`
- 문서 업데이트:
  - `PART1_Data_Collection.md`
  - `PART2_Analysis_Screening.md`
  - `strategy_overview.md`

## 품질 체크리스트
- 파이프라인 전체 실행 (`us_market/update_all.py`) 시 실패 없이 완료
- Flask 주요 엔드포인트 정상 응답
- 캐시/레이트리밋 처리로 429 발생 최소화

## 유의 사항
- KR 지수/환율 심볼이 FMP에 없을 수 있으므로 fallback을 남겨둘 것
- 옵션 체인은 FMP 문서에 없으므로 별도 공급자 검토 필요

## 완료 기준
- yfinance import가 코드베이스에서 제거됨
- 산출물 스키마 불변
- 테스트 통과

## 참고
- FMP 엔드포인트: `docs/fmp-migration-guide.md`
