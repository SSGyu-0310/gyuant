# CSV/JSON -> SQLite 마이그레이션 체크리스트

이 문서는 CSV/JSON 산출물을 SQLite로 일괄 이관하고, 단일 소스로 전환하기 위한 실행 체크리스트입니다.

## 1) 사전 준비
- [ ] `DATA_DIR/gyuant.db` 경로 확정 및 백업 정책 정의
- [ ] 파이프라인/서버 쓰기 중단(데이터 Freeze 시간 확보)
- [ ] `docs/db/schema.sql`로 스키마 생성
- [ ] `PRAGMA foreign_keys=ON`, `journal_mode=WAL`, `busy_timeout` 적용 확인

## 2) 테이블별 이관 순서 (권장)
1. `market_stocks`
   - source: `us_stocks_list.csv`
   - 필드: `ticker`, `name`, `sector`, `market` (+옵션: `industry`)
2. `market_prices_daily`
   - source: `us_daily_prices.csv`
   - 매핑: `current_price` -> `close`
3. `market_volume_analysis`
   - source: `us_volume_analysis.csv`
   - `date` -> `as_of_date`
4. `market_etf_flows`
   - source: `us_etf_flows.csv`
5. `market_smart_money_runs` + `market_smart_money_picks`
   - source: `smart_money_current.json`, `smart_money_picks_v2.csv`, `history/picks_YYYYMMDD.json`
6. `market_documents`
   - source: JSON 산출물 (macro/heatmap/options/insider/risk/calendar/ai summaries)

## 3) JSON -> market_documents 매핑 규칙
- `macro_analysis.json` -> `doc_type=macro_analysis`, `lang=ko`, `model=gemini`
- `macro_analysis_en.json` -> `doc_type=macro_analysis`, `lang=en`, `model=gemini`
- `macro_analysis_gpt.json` -> `doc_type=macro_analysis`, `lang=ko`, `model=gpt`
- `macro_analysis_gpt_en.json` -> `doc_type=macro_analysis`, `lang=en`, `model=gpt`
- `sector_heatmap.json` -> `doc_type=sector_heatmap`
- `options_flow.json` -> `doc_type=options_flow`
- `insider_moves.json` -> `doc_type=insider_moves`
- `portfolio_risk.json` -> `doc_type=portfolio_risk`
- `ai_summaries.json` -> `doc_type=ai_summaries`
- `weekly_calendar.json` -> `doc_type=calendar`
- `etf_flow_analysis.json` -> `doc_type=etf_flow_analysis`

## 4) 검증 체크
- [x] 로우 카운트 비교(CSV vs DB)
- [x] PK 중복 여부 체크 (market_prices_daily)
- [x] 날짜 포맷 ISO 여부 확인 (market_prices_daily)
- [x] 샘플 티커로 시계열 조회 정상 동작 확인
- [x] `market_smart_money_runs` 최신 run_id 기준으로 picks 조인 확인

## 5) 애플리케이션 전환
- [x] 파이프라인 스크립트 DB 쓰기 전환
- [x] Flask API DB 읽기 전환 (응답 스키마 동일)
- [ ] 데이터 계약/검증 로직 DB 기준으로 수정
- [ ] CSV/JSON 파일 의존 제거 (export 용도만 유지)

## 6) 롤백 계획
- [ ] DB 백업 파일 보관
- [ ] 기존 CSV/JSON 읽기 경로 보관
- [ ] 전환 실패 시 복귀 경로 문서화
