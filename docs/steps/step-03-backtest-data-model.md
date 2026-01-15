# Step 3: Backtest Data Model (SQLite, Point-in-Time 중심)

## 목표
과거 시점 재현이 가능한 데이터 구조를 SQLite로 확정하고, 백테스트에 필요한 최소 테이블을 구축한다.

## 범위
- 포함: 백테스트용 핵심 테이블 설계, point-in-time 규칙 정의, 품질 체크 기준.
- 제외: 백테스트 엔진 로직 구현(이는 Step 4).

## 현재 구현 스냅샷 (코드 기준)
런타임 SQLite 스키마(`backtest/db_schema.py`)에는 아래 테이블만 존재합니다:
- `bt_prices_daily`, `bt_universe_snapshot`
- `bt_signal_definitions`, `bt_signals`
- `bt_runs`, `bt_run_metrics`, `bt_run_equity_curve`, `bt_run_positions`, `bt_run_trades`

`bt_fundamentals` 및 `bt_alpha_*` 계열 테이블은 아직 스키마에 포함되어 있지 않습니다.

## 핵심 원칙
- Look-ahead bias 금지: 모든 지표/시그널은 `as_of_date` 이전 데이터로 계산.
- Survivorship bias 최소화: 과거 시점 유니버스 스냅샷을 보존.
- 단일 소스: `DATA_DIR/gyuant.db`만 사용 (CSV 사용 금지).

## 결정사항
- DB 파일: `DATA_DIR/gyuant.db`
- 백테스트 테이블 접두사: `bt_`
- 날짜/시간 규칙: `YYYY-MM-DD`, `ISO 8601` 타임스탬프

## MVP 알파 정의 (스크리닝 → 포트폴리오)
백테스트 이전에 **스크리닝 결과를 “정량 포트폴리오(알파)”로 변환**하는 최소 규칙을 확정한다.

### 기본 알파 스펙 (예시)
- `signal_id`: `screening_top20_equal_weight`
- `signal_version`: `v1`
- 입력 소스: `market_smart_money_picks` (최신 run 기준)
- 시그널 값: `composite_score`
- 랭크: `rank`
- 포트폴리오 구성 규칙 (config_json에 저장)
  - `top_n`: 20
  - `weight_scheme`: `equal`
  - `target_weight`: 0.05 (20종목 × 5%)
  - `rebalance_freq`: `none` (MVP는 리밸런싱 없음)
  - `hold_period_days`: 252 (1년 보유, 필요 시 조정)
  - `entry_price`: `next_open` 또는 `next_close` 중 하나 고정
  - `cash_handling`: `hold_cash` (선정 종목 부족 시 현금 유지)

### 시그널 적재 규칙 (스크리닝 결과 → bt_signals)
- `bt_signals.signal_value` = `market_smart_money_picks.composite_score`
- `bt_signals.signal_rank` = `market_smart_money_picks.rank`
- `bt_signals.meta_json`에 세부 점수(`sd_score`, `tech_score`, `fund_score`, `analyst_score`, `rs_score`) 저장
- `as_of_date`는 `market_smart_money_runs.analysis_date` 기준

### 공통 포트폴리오 스냅샷 (스크리닝/백테스트 공용 출력)
프론트엔드와 백테스트가 **동일한 포트폴리오 구성 결과**를 공유하도록 문서형 스냅샷을 권장한다.
- 저장 위치: `market_documents` (doc_type = `portfolio_proposal`)
- 페이로드 예시:
  - `as_of_date`, `strategy_id`, `strategy_version`
  - `universe_filter` (유니버스 필터 요약)
  - `positions`: `[ticker, weight, rank, composite_score]`
  - `rules`: `{top_n, weight_scheme, rebalance_freq, hold_period_days, entry_price}`

## 유니버스 확장 계획 (S&P 500 → 1k~3k)
### 목표
현재 S&P 500 고정 리스트를 **유동적 유니버스**로 확장한다.

### 확보 절차 (초기 1회 + 정기 갱신)
1) FMP에서 전체 US 종목 리스트 확보(티커/거래소/타입).
2) 공통 필터:
   - 보통주(common stock)만 포함
   - OTC/불명확 거래소 제외
3) 유동성/시총 필터(예시 기준):
   - `market_cap >= 1B` 또는
   - `avg_dollar_volume_20d >= 5M`
4) 필터 결과를 `market_stocks`에 저장하고 `is_active=1`로 유지.
5) `bt_universe_snapshot`을 월 1회 또는 분기 1회 스냅샷으로 저장.

### 필터링 데이터 소스
- 시총: FMP `profile` or `quote`
- 평균 거래대금: `market_prices_daily`에서 20일 평균 `close * volume`
- 결과는 `bt_signal_definitions.params_schema_json`의 `universe_filter`에 기록

## 관련 문서
- `docs/db/schema.sql`: 백테스트 테이블 DDL
- `docs/db/erd.md`: 테이블 관계 요약

## 핵심 테이블 설계
### 1) `bt_prices_daily`
- 목적: 백테스트 가격 원본(일봉)
- 컬럼: `ticker`, `date`, `open`, `high`, `low`, `close`, `volume`, `source`, `ingested_at`
- PK: (`ticker`, `date`)
- 인덱스: (`date`)

### 2) `bt_universe_snapshot`
- 목적: 특정 시점의 유니버스 보존
- 컬럼: `as_of_date`, `ticker`, `name`, `sector`, `market`, `source`, `ingested_at`
- PK: (`as_of_date`, `ticker`)
- 인덱스: (`as_of_date`)

### 3) `bt_signal_definitions`
- 목적: 시그널(알파) 메타 정보
- 컬럼: `signal_id`, `name`, `description`, `version`, `params_schema_json`, `created_at`, `updated_at`
- PK: (`signal_id`, `version`)

### 4) `bt_signals`
- 목적: 시그널 산출 결과(점수/랭크)
- 컬럼: `signal_id`, `signal_version`, `as_of_date`, `ticker`, `signal_value`, `signal_rank`, `meta_json`
- PK: (`signal_id`, `signal_version`, `as_of_date`, `ticker`)
- 인덱스: (`as_of_date`, `signal_id`)

### 5) `bt_fundamentals` (선택, 권장, **미구현**)
- 목적: 가치 지표의 point-in-time 저장
- 컬럼 예시: `as_of_date`, `ticker`, `pe_ratio`, `pb_ratio`, `revenue_growth`, `roe`, `market_cap`, `source`
- PK: (`as_of_date`, `ticker`)

### 6) `bt_runs`
- 목적: 백테스트 실행 메타
- 컬럼: `run_id`, `signal_id`, `signal_version`, `alpha_id`, `alpha_version`, `config_json`,
  `as_of_date`, `start_date`, `end_date`, `top_n`, `hold_period_days`, `rebalance_freq`,
  `transaction_cost_bps`, `status`, `created_at`, `finished_at`, `error`
- PK: (`run_id`)
- 인덱스: (`as_of_date`), (`status`)

### 7) `bt_run_metrics`
- 목적: 성과 지표
- 컬럼: `run_id`, `cagr`, `volatility`, `sharpe`, `mdd`, `total_return`, `win_rate`, `turnover`
- PK: (`run_id`)

### 8) `bt_run_equity_curve`
- 목적: 일별 포트폴리오 시계열
- 컬럼: `run_id`, `date`, `equity`, `returns`, `drawdown`
- PK: (`run_id`, `date`)

### 9) `bt_run_positions`
- 목적: 포지션 상세
- 컬럼: `run_id`, `ticker`, `entry_date`, `entry_price`, `exit_date`, `exit_price`, `weight`, `shares`
- PK: (`run_id`, `ticker`, `entry_date`)

### 10) `bt_run_trades` (선택)
- 목적: 매매 로그
- 컬럼: `run_id`, `trade_id`, `ticker`, `side`, `trade_date`, `price`, `shares`, `fee`
- PK: (`run_id`, `trade_id`)

## 품질 체크리스트
### 데이터 품질
- `date/as_of_date`는 ISO 포맷으로 통일됨
- 동일 PK 중복 없음
- 가격 데이터 누락 구간/정지 종목 처리 규칙 문서화

### 시점 일관성 (Look-ahead 방지)
- 시그널 산출 시점은 `as_of_date` 이전 데이터만 사용
- 펀더멘털/애널리스트 데이터는 향후 point-in-time 테이블로 이동

### 유니버스 일관성
- `bt_universe_snapshot`이 `as_of_date` 기준으로 존재
- 스냅샷 기반으로 시그널/포트폴리오 구성

## 구현 체크리스트 (남은 작업)
- [ ] `market_smart_money_runs/picks` → `bt_signals` 적재 스크립트 구현
- [ ] `bt_signal_definitions`에 `screening_top20_equal_weight` 정의 등록
- [ ] 스크리닝 결과로 `portfolio_proposal` 문서 생성(공통 포맷)
- [ ] 유니버스 확장 스크립트 구현(1k~3k 필터 기준 포함)
- [ ] `bt_universe_snapshot` 정기 스냅샷 정책 확정(월/분기 단위)

## 완료 기준
- 최소 1년 이상 `bt_prices_daily` 확보
- 유니버스 스냅샷 1개 이상 생성
- 시그널 파일 대신 `bt_signals` 테이블 생성 및 1회 이상 적재
- 스키마/인덱스가 확정되어 팀 기준으로 공유됨
