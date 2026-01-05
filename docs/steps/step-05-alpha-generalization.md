# Step 5: Alpha 확장 로드맵 (파라미터 -> 사용자 정의 -> LLM)

## 목표
고정 전략을 점진적으로 확장하여
1) 파라미터 튜닝,
2) 사용자 정의 알파,
3) 자연어 알파(LLM 변환)
까지 단계적으로 구현한다.

## 범위
- 포함: 알파 정의/버전 관리, 파라미터 실험, 사용자 정의 알파 빌더, LLM 변환 파이프라인.
- 제외: 대규모 자동 최적화(필요 시 별도 단계).

## 단계별 확장 계획

### Phase A: 파라미터 튜닝 (작은 확장)
**목표**: 기존 고정 전략을 “설정 가능한 알파”로 전환.

- 파라미터 예시
  - `top_n`, `hold_period_days`, `rebalance_freq`, `min_score`, `universe_filter`
  - `weight_scheme` (`equal`, `score_proportional`, `capped_score`)
  - `entry_price` (`next_open`, `next_close`)
  - `transaction_cost_bps`, `slippage_bps`
- 저장 구조
  - `bt_signal_definitions.params_schema_json`에 파라미터 스키마 저장
  - `bt_runs.config_json`에 실제 파라미터 저장
- 결과 비교
  - `bt_run_metrics`로 성과 비교
  - `bt_alpha_runs`(선택) 테이블로 실험 로그 관리

**구현 상세**
- `params_schema_json` 예시:
  - `{"top_n":{"type":"int","min":5,"max":50},"weight_scheme":{"enum":["equal","score_proportional"]}}`
- 파라미터 검증 단계:
  - 입력 타입/범위 확인 → 기본값 보정 → 실행 로그 기록
- 성과 비교 보고서:
  - 최소 3개 파라미터 조합 비교
  - 동일 기간/유니버스로 비교하고 결과 표/그래프 저장

**체크리스트**
- [ ] 파라미터 스키마 정의 (JSON schema)
- [ ] UI에서 파라미터 입력 가능
- [ ] 동일 알파/다른 설정 비교 기능

### Phase B: 사용자 정의 알파 (조금 더 유연한 범위)
**목표**: 사용자가 지표를 조합해 알파를 생성하고 백테스트 가능.

- 알파 정의 테이블(권장)
  - `bt_alpha_defs`: `alpha_id`, `name`, `description`, `expression`, `status`, `created_at`
  - `bt_alpha_versions`: `alpha_id`, `version`, `expression`, `params_schema_json`, `created_at`
- 표현 방식(안전한 DSL/룰 기반)
  - 허용 필드/함수 allow-list
  - 예: `score = 0.5*rank(roe) - 0.5*rank(pe_ratio)`
- 검증 절차
  - 파서/타입 체크 → 샘플 데이터 실행 → 점수 분포 확인

**구현 상세**
- 팩터 계산 파이프라인:
  - `bt_fundamentals`에 point-in-time 팩터 저장(월/분기 업데이트)
  - `rank()`, `zscore()`, `winsorize()` 같은 전처리 함수 제공
- 룰 실행 안전장치:
  - 허용 컬럼/함수 목록 외 접근 차단
  - 연산 복잡도 제한(연산자 수, 함수 깊이 제한)
- 결과 품질 검증:
  - 점수 분포(평균/표준편차/상하위 편차) 자동 리포트
  - 상위/하위 종목 샘플 출력

**체크리스트**
- [ ] 알파 DSL/룰 문법 확정
- [ ] 알파 검증 파이프라인 구축
- [ ] 알파 생성/저장/수정 UI 제공

### Phase C: 자연어 알파 (LLM 변환)
**목표**: 사용자가 자연어로 알파를 설명하면 LLM이 안전한 알파 정의로 변환하고 저장.

- 저장 구조(권장)
  - `bt_alpha_nl_requests`: `request_id`, `user_input`, `llm_output`, `model`, `status`, `created_at`
  - `bt_alpha_defs`에 최종 승인된 알파만 저장
- 안전장치
  - allow-list 필드/함수만 사용
  - 실행 전 규칙 검증 + human approval 단계
  - 복잡도 제한(연산 수, 함수 깊이, 데이터 범위)
- 운영 정책
  - 실패 시 이유를 저장하고 UI에 반환
  - 승인/거절 이력 관리

**구현 상세**
- 자연어 입력 → DSL 변환 로그 저장
- 변환 실패 유형 분류(미지원 지표, 과도한 복잡도, 보안 위반)
- 승인 후 `bt_alpha_defs`에 반영, 거절 사유는 `bt_alpha_nl_requests`에 저장

**체크리스트**
- [ ] 자연어 -> DSL 변환 프롬프트/템플릿 정의
- [ ] 안전성 검증(정적 분석) 구현
- [ ] 승인 워크플로우 및 로그 저장

## 확장 고려사항 (유니버스/팩터 확대)
- 유니버스 확장(1k~3k) 시 필터 기준을 파라미터화
  - 예: `min_market_cap`, `min_avg_dollar_volume_20d`, `exclude_exchanges`
- 가치/퀄리티 팩터 추가
  - `ROE`, `EV/EBITDA`, `FCF Yield`, `Profit Margin` 등
- 알파 간 앙상블(가중 평균/투표) 실험 가능

## 공통 고려사항
- **버전 관리**: 알파는 버전 단위로 고정 재현 가능해야 함
- **재현성**: 동일 입력/데이터에서 동일 결과 보장
- **보안**: 임의 코드 실행 금지, SQL 직접 실행 금지
- **성능**: 지표 계산은 사전 계산 또는 캐시 활용

## 관련 문서
- `docs/db/schema.sql`: `bt_alpha_*` 및 `bt_runs` 테이블 정의

## 완료 기준
- Phase A: 파라미터 변경에 따라 결과가 달라짐을 검증
- Phase B: 사용자 알파 1개 이상 생성 및 백테스트 성공
- Phase C: 자연어 알파 1개 이상 승인/저장 및 백테스트 성공
