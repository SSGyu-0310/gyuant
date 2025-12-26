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
- 저장 구조
  - `bt_signal_definitions.params_schema_json`에 파라미터 스키마 저장
  - `bt_runs.config_json`에 실제 파라미터 저장
- 결과 비교
  - `bt_run_metrics`로 성과 비교
  - `bt_alpha_runs`(선택) 테이블로 실험 로그 관리

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

**체크리스트**
- [ ] 자연어 -> DSL 변환 프롬프트/템플릿 정의
- [ ] 안전성 검증(정적 분석) 구현
- [ ] 승인 워크플로우 및 로그 저장

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
