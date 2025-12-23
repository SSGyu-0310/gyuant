# Step 3: Backtest Data Model (Point-in-Time 중심)

## 목표
과거 시점 재현이 가능한 데이터 구조를 설계하고, 백테스트에 필요한 기본 데이터를 저장한다.

## 범위
- 포함: 가격 데이터, 유니버스, 시그널(알파) 데이터 구조 정의 및 저장 포맷.
- 제외: 백테스트 엔진 구현(이는 Step 4).

## 핵심 원칙
- Look-ahead bias 금지: 모든 시그널/지표는 해당 날짜 이전 데이터로만 계산.
- Survivorship bias 최소화: 가능한 한 과거 시점의 유니버스를 보존.

## 권장 저장 구조 (로컬 기준)
기본 디렉토리: `DATA_DIR/backtest/`

예시:
- `backtest/prices_daily.csv`
- `backtest/universe_snapshots/{YYYY-MM-DD}.csv`
- `backtest/signals/{signal_id}.csv`
- `backtest/runs/{run_id}.json`

## 스키마 제안
### 1) 가격 데이터 (`prices_daily.csv`)
필수 컬럼:
- `date` (YYYY-MM-DD)
- `ticker`
- `open`, `high`, `low`, `close`
- `volume`
- `source` (예: `FMP`)

선택 컬럼:
- `adj_close`
- `as_of` (수집 시점)

### 2) 유니버스 스냅샷 (`universe_snapshots/*.csv`)
필수 컬럼:
- `date`
- `ticker`
- `name`
- `sector`
- `market`

### 3) 시그널 데이터 (`signals/{signal_id}.csv`)
필수 컬럼:
- `as_of_date` (시그널 계산 기준일)
- `ticker`
- `signal_value`
- `signal_version`

## 구현 지침 (에이전트용)
1) `create_us_daily_prices.py`의 산출물을 그대로 재사용하거나, 백테스트 전용 CSV로 복사/정규화.
2) 유니버스 스냅샷 저장:
   - 최소 기준: 현재 `us_stocks_list.csv`를 날짜별 스냅샷으로 저장.
   - 확장 가능: FMP 상장/인덱스 히스토리로 과거 유니버스 확보.
3) 시그널 데이터 저장:
   - 알파 계산 결과를 `as_of_date` 기준으로 저장.
4) 데이터 품질 체크:
   - 결측치 비율, 중복 행, 날짜 정합성 검사.

## 품질 체크리스트
- `date` 컬럼이 ISO 포맷으로 통일됨
- 동일 `ticker,date` 중복 없음
- 시그널 계산 시 `as_of_date` 이후 데이터 미사용 보장

## 완료 기준
- 최소 1년 이상의 `prices_daily.csv` 확보
- 유니버스 스냅샷 최소 1개 생성
- 시그널 파일 1개 이상 생성

