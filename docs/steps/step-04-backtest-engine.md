# Step 4: Backtest Engine + Report (장기 보유용)

## 목표
가치투자 스타일의 장기 보유 백테스트를 구현하고, 시각화 및 핵심 지표(Sharpe, MDD 등)를 출력한다.

## 범위
- 포함: 백테스트 로직, 성과 지표, 그래프 출력.
- 제외: 알파 확장/파라미터 최적화(이는 Step 5).

## 기본 규칙 (가치투자 시나리오)
- 진입: 특정 날짜 또는 특정 규칙(예: 분기 첫날)로 매수.
- 보유: N개월/년 고정 보유.
- 리밸런싱: 최소화(필요 시 월/분기 단위).

## 구현 지침 (에이전트용)
### 1) 모듈 구조
권장 파일:
- `backtest/engine.py` (핵심 로직)
- `backtest/metrics.py` (Sharpe, MDD, CAGR 등)
- `backtest/report.py` (JSON/CSV 출력)
- `backtest/plots.py` (그래프 생성)

### 2) 입력/출력 규격
입력:
- `backtest/prices_daily.csv`
- `backtest/signals/{signal_id}.csv`
출력:
- `backtest/runs/{run_id}.json` (설정+메트릭)
- `backtest/runs/{run_id}_equity_curve.csv`
- `backtest/runs/{run_id}_positions.csv`
- `backtest/runs/{run_id}.png` (그래프)

### 3) 지표 계산
필수:
- 누적 수익률
- 연환산 수익률(CAGR)
- 변동성(연환산)
- 샤프(무위험 0~2% 가정 가능)
- 최대 낙폭(MDD)

### 4) 시각화
필수 그래프:
- 누적 수익률 곡선
- Drawdown 곡선
도구:
- `matplotlib` 또는 `plotly` (새 의존성 추가 시 `requirements.txt` 업데이트)

### 5) 신뢰성 확보
- 고정된 시드/입력 기준으로 동일 결과 보장
- 단일 종목 테스트로 sanity check

## 테스트 전략
- 소규모 샘플 데이터로 정답이 예측 가능한 케이스 작성
- 누적 수익률/샤프 계산의 단위 테스트

## 완료 기준
- 백테스트 1회 실행 시 보고서와 그래프가 생성됨
- 샤프/MDD 수치가 재현 가능

