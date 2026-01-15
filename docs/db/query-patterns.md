# Query Patterns + Index Guide

본 문서는 API/백테스트에서 발생하는 대표 쿼리와 인덱스 설계 기준을 정리합니다. 실제 인덱스는 `docs/db/schema.sql`을 기준으로 합니다.

## 1) Market API 주요 쿼리

### 최신 Smart Money 스냅샷
```sql
SELECT run_id, analysis_date, analysis_timestamp
FROM market_smart_money_runs
ORDER BY analysis_date DESC
LIMIT 1;
```
인덱스: `market_smart_money_runs(analysis_date)` (필요 시 추가)

### 특정 run의 picks 조회
```sql
SELECT *
FROM market_smart_money_picks
WHERE run_id = ?
ORDER BY rank ASC;
```
인덱스: `idx_market_picks_run`

### 티커 가격 시계열
```sql
SELECT date, close, volume
FROM market_prices_daily
WHERE ticker = ? AND date BETWEEN ? AND ?
ORDER BY date ASC;
```
인덱스: PK(`ticker`, `date`) + `idx_market_prices_date`

### 매크로/문서 조회
```sql
SELECT payload_json
FROM market_documents
WHERE doc_type = ? AND lang = ? AND model = ?
ORDER BY as_of_date DESC, updated_at DESC
LIMIT 1;
```
인덱스: `idx_market_docs_type_date`

### ETF 플로우 최신일자 조회
```sql
SELECT *
FROM market_etf_flows
WHERE as_of_date = ?
ORDER BY flow_score DESC;
```
인덱스: `idx_market_etf_asof`

## 2) Backtest 주요 쿼리

### 실행 목록
```sql
SELECT run_id, status, created_at, as_of_date
FROM bt_runs
ORDER BY created_at DESC
LIMIT ?;
```
인덱스: `idx_bt_runs_status`, (필요 시) `bt_runs(created_at)`

### 실행 상세 (메타 + 지표)
```sql
SELECT r.*, m.*
FROM bt_runs r
LEFT JOIN bt_run_metrics m ON r.run_id = m.run_id
WHERE r.run_id = ?;
```
인덱스: PK on `bt_runs`, `bt_run_metrics`

### Equity Curve
```sql
SELECT date, equity, returns, drawdown
FROM bt_run_equity_curve
WHERE run_id = ?
ORDER BY date ASC;
```
인덱스: PK(`run_id`, `date`)

### Positions
```sql
SELECT *
FROM bt_run_positions
WHERE run_id = ?
ORDER BY entry_date ASC;
```
인덱스: PK(`run_id`, `ticker`, `entry_date`)

### 시그널 상위 N개
```sql
SELECT *
FROM bt_signals
WHERE signal_id = ? AND signal_version = ? AND as_of_date = ?
ORDER BY signal_rank ASC
LIMIT ?;
```
인덱스: `idx_bt_signals_rank`

### 유니버스 스냅샷
```sql
SELECT ticker
FROM bt_universe_snapshot
WHERE as_of_date = ?;
```
인덱스: `idx_bt_universe_asof`

## 3) 인덱스 설계 원칙
- **PK 기반 조회 우선**: `(ticker, date)` 등 복합 PK는 범위 조회에 유리합니다.
- **날짜 필터 빈도**가 높은 테이블은 `date/as_of_date` 보조 인덱스를 둡니다.
- **순위/정렬 기반** 조회는 `(..., signal_rank)` 형태 인덱스를 둡니다.
- **JSON 컬럼은 인덱스 제외**: 필요 시 계산/추출 값을 별도 컬럼화합니다.

## 4) 추가 고려사항
- 대용량 테이블(`market_prices_daily`, `bt_prices_daily`)은 과도한 인덱스 추가를 피합니다.
- Backtest 결과 테이블은 `run_id` 기준 조회가 대부분이므로 PK만으로 충분합니다.
- 엔진/파이프라인 변경으로 쿼리가 바뀌면 인덱스도 같이 갱신합니다.
