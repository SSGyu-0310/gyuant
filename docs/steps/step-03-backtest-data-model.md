# Step 3: Backtest Data Model (SQLite, Point-in-Time 중심)

## 목표
과거 시점 재현이 가능한 데이터 구조를 SQLite로 확정하고, 백테스트에 필요한 최소 테이블을 구축한다.

## 범위
- 포함: 백테스트용 핵심 테이블 설계, point-in-time 규칙 정의, 품질 체크 기준.
- 제외: 백테스트 엔진 로직 구현(이는 Step 4).

## 핵심 원칙
- Look-ahead bias 금지: 모든 지표/시그널은 `as_of_date` 이전 데이터로 계산.
- Survivorship bias 최소화: 과거 시점 유니버스 스냅샷을 보존.
- 단일 소스: `DATA_DIR/gyuant.db`만 사용 (CSV 사용 금지).

## 결정사항
- DB 파일: `DATA_DIR/gyuant.db`
- 백테스트 테이블 접두사: `bt_`
- 날짜/시간 규칙: `YYYY-MM-DD`, `ISO 8601` 타임스탬프

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

### 5) `bt_fundamentals` (선택, 권장)
- 목적: 가치 지표의 point-in-time 저장
- 컬럼 예시: `as_of_date`, `ticker`, `pe_ratio`, `pb_ratio`, `revenue_growth`, `roe`, `market_cap`, `source`
- PK: (`as_of_date`, `ticker`)

### 6) `bt_runs`
- 목적: 백테스트 실행 메타
- 컬럼: `run_id`, `signal_id`, `signal_version`, `config_json`, `as_of_date`, `start_date`, `end_date`,
  `top_n`, `hold_period_days`, `rebalance_freq`, `status`, `created_at`, `finished_at`, `error`
- PK: (`run_id`)
- 인덱스: (`as_of_date`), (`status`)

### 7) `bt_run_metrics`
- 목적: 성과 지표
- 컬럼: `run_id`, `cagr`, `volatility`, `sharpe`, `mdd`, `total_return`, `win_rate`, `turnover`
- PK: (`run_id`)

### 8) `bt_run_equity_curve`
- 목적: 일별 포트폴리오 시계열
- 컬럼: `run_id`, `date`, `equity`, `returns`
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
- `date/as_of_date`는 ISO 포맷으로 통일됨
- 동일 PK 중복 없음
- 시그널은 `as_of_date` 이후 데이터를 절대 사용하지 않음
- 유니버스 스냅샷은 과거 시점 기준으로 보존됨

## 완료 기준
- 최소 1년 이상 `bt_prices_daily` 확보
- 유니버스 스냅샷 1개 이상 생성
- 시그널 파일 대신 `bt_signals` 테이블 생성 및 1회 이상 적재
- 스키마/인덱스가 확정되어 팀 기준으로 공유됨
