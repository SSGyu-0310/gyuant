# Step 5: Alpha Generalization (파라미터 + 지표 확장)

## 목표
현재의 고정된 알파를 범용화하여 파라미터 조정 및 지표 확장을 가능하게 한다.

## 범위
- 포함: 알파 인터페이스 정의, 파라미터화, 실험 결과 관리.
- 제외: 대규모 자동 최적화(필요 시 후속 단계).

## 구현 지침 (에이전트용)
### 1) 알파 인터페이스 정의
권장 형태:
- 입력: 가격/재무/지표 데이터프레임 + `config`
- 출력: `signals` (ticker, as_of_date, signal_value)

### 2) 파라미터 구성 파일
권장 구조:
- `alpha_configs/{alpha_id}.json`
- 예: `lookback`, `threshold`, `top_n` 등

### 3) 알파 레지스트리
권장 구조:
- `alpha/registry.py` 에 `alpha_id -> 함수` 매핑
- 기존 알파는 baseline으로 고정 보존

### 4) 실험 관리
저장:
- `backtest/alpha_runs.csv`
- 컬럼 예: `alpha_id`, `config_hash`, `CAGR`, `Sharpe`, `MDD`, `run_id`

### 5) 과최적화 방지
- 최소한의 train/validation 구간 분리
- 성능이 과도하게 튀는 설정은 제외

## 테스트 전략
- 파라미터 변경에 따라 결과가 변화하는지 검증
- baseline 알파 결과가 유지되는지 회귀 테스트

## 완료 기준
- 알파 설정 파일 기반으로 실행 가능
- 최소 3개 설정 결과 비교 가능
- baseline 결과 재현 가능

