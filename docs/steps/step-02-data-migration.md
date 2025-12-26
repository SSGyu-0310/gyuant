# Step 2: Storage Migration (CSV/JSON -> SQLite 단일 소스)

## 목표
모든 운영 데이터(가격/분석/스냅샷)를 CSV/JSON이 아닌 SQLite에 저장하고, DB를 단일 소스로 사용한다.

## 범위
- 포함: DB 스키마 설계, 기존 산출물 마이그레이션, 파이프라인/Flask API 읽기-쓰기 전환.
- 제외: 백테스트 엔진 로직 구현(이는 Step 4), 알파 확장(이는 Step 5).

## 결정사항
- SQLite DB 파일: `DATA_DIR/gyuant.db`
- 단일 소스 원칙: CSV/JSON은 더 이상 소스로 사용하지 않으며 필요 시 “export” 용도로만 사용.
- 시간 규칙: 모든 날짜는 `YYYY-MM-DD`, 타임스탬프는 ISO 8601(UTC 권장).
- 동시성: `WAL` 모드 + `busy_timeout` 사용.

## 관련 문서
- `docs/db/schema.sql`: 전체 스키마 DDL (단일 소스)
- `docs/db/erd.md`: 테이블 관계 요약
- `docs/db/query-patterns.md`: 쿼리 패턴 + 인덱스 가이드
- `docs/db/migration-checklist.md`: CSV/JSON -> SQLite 전환 체크리스트

## 테이블 매핑 (운영 데이터)
| 기존 파일 | 새 테이블 | 비고 |
| --- | --- | --- |
| `us_daily_prices.csv` | `market_prices_daily` | `current_price` -> `close`로 매핑 |
| `us_stocks_list.csv` | `market_stocks` | 티커 메타(섹터/시장) |
| `us_volume_analysis.csv` | `market_volume_analysis` | 수급 지표/점수 |
| `us_etf_flows.csv` | `market_etf_flows` | ETF 플로우 데이터 |
| `smart_money_picks_v2.csv` | `market_smart_money_picks` | 분석일 기준 Top N |
| `smart_money_current.json` | `market_smart_money_runs` + `market_smart_money_picks` | 요약/스냅샷 정보 |
| `history/picks_YYYYMMDD.json` | `market_smart_money_runs` + `market_smart_money_picks` | 과거 스냅샷 |
| `macro_analysis*.json` | `market_documents` | `doc_type=macro_analysis`, `lang`, `model` 포함 |
| `sector_heatmap.json` | `market_documents` | `doc_type=sector_heatmap` |
| `options_flow.json` | `market_documents` | `doc_type=options_flow` |
| `insider_moves.json` | `market_documents` | `doc_type=insider_moves` |
| `portfolio_risk.json` | `market_documents` | `doc_type=portfolio_risk` |
| `ai_summaries.json` | `market_documents` | `doc_type=ai_summaries` |
| `weekly_calendar.json` | `market_documents` | `doc_type=calendar` |
| `etf_flow_analysis.json` | `market_documents` | `doc_type=etf_flow_analysis` |

## 권장 스키마 (핵심 테이블 요약)
실제 DDL은 `docs/db/schema.sql`을 기준으로 하며, 아래는 이해를 위한 요약입니다.
### 1) `market_prices_daily`
- `ticker` TEXT, `date` DATE, `open` REAL, `high` REAL, `low` REAL, `close` REAL, `volume` REAL
- `change` REAL, `change_rate` REAL, `source` TEXT, `ingested_at` TEXT
- PK: (`ticker`, `date`)

### 2) `market_volume_analysis`
- `ticker` TEXT, `as_of_date` DATE
- `supply_demand_score` REAL, `supply_demand_stage` TEXT, 기타 지표 컬럼
- PK: (`ticker`, `as_of_date`)

### 3) `market_smart_money_runs` / `market_smart_money_picks`
- `market_smart_money_runs`: `run_id`, `analysis_date`, `analysis_timestamp`, `summary_json`, `created_at`
- `market_smart_money_picks`: `run_id`, `rank`, `ticker`, `composite_score`, `grade`, `current_price`, `target_upside`, 기타 점수 컬럼
- FK: picks.`run_id` -> runs.`run_id`

### 4) `market_documents`
- `doc_type` TEXT, `as_of_date` DATE, `lang` TEXT, `model` TEXT, `payload_json` TEXT, `updated_at` TEXT
- PK: (`doc_type`, `as_of_date`, `lang`, `model`)

## 작업 지침
1) DB 초기화 스크립트 작성
   - 테이블 생성 + 인덱스 + PRAGMA 설정(WAL/foreign_keys/busy_timeout).
2) 기존 CSV/JSON 일괄 마이그레이션 스크립트 작성
   - 1회 import 후 CSV/JSON 의존 제거.
3) 파이프라인 스크립트 수정
   - 각 `us_market/*.py`가 결과를 DB에 저장하도록 변경.
4) Flask API 수정
   - 파일 읽기 대신 DB 조회로 응답 생성.
5) 운영 검증
   - 기존 API 응답 스키마 유지 확인.

## 체크리스트
- [x] SQLite 초기 스키마 확정 및 마이그레이션 스크립트 추가
- [x] CSV/JSON -> DB 1회 백필 완료
- [x] 파이프라인 전 스크립트 DB 쓰기 전환
- [x] Flask API DB 읽기 전환 (응답 스키마 유지)
- [x] 데이터 계약(스키마/존재/최신성) 점검 로직 수정
- [x] CSV/JSON 파일 의존 제거 또는 export-only로 전환

## 완료 기준
- DB가 운영 데이터의 단일 소스가 됨
- 기존 API가 동일한 스키마로 응답
- CSV/JSON이 기본 데이터 경로에서 제거됨
