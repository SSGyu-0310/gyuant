# Step 4: Backtest Engine + Report (장기 보유, SQLite 기반)

## 목표
고정된 가치 전략을 기준으로 과거 시점 스크리닝 → 장기 보유 → 성과 지표/그래프를 산출하는 백테스트 엔진을 구현한다.

## 범위
- 포함: 백테스트 실행 로직, 성과 지표 계산, 결과 저장(SQLite), 기본 API 스펙.
- 제외: 알파 일반화/자연어 알파(이는 Step 5).

## 기본 시나리오 (MVP)
- 스크리닝 기준일(`as_of_date`)에 시그널 상위 N개 선정
- 다음 거래일에 매수, 고정 기간 보유(예: 1년)
- 리밸런싱 최소화(기본: 없음)
- 동일가중 포트폴리오

## 포트폴리오 구성 규칙 (MVP 상세)
**스크리닝 = 포트폴리오 제안** 구조를 유지한다.
- 기본 알파: `screening_top20_equal_weight`
- 구성 규칙:
  - `top_n = 20`, `weight_scheme = equal`, `target_weight = 0.05`
  - 리밸런싱 없음(보유기간 종료 시 전량 청산)
  - 진입일: `as_of_date` 다음 거래일
  - 진입가: `open` 또는 `close` 중 하나로 고정
  - 청산일: 보유기간 이후 첫 거래일
  - 현금: 선정 종목 부족 시 현금 비중 유지

## 엔진 모듈 구조 (권장)
> 실제 구현 시 폴더/모듈은 팀 컨벤션에 맞게 조정
- `backtest/engine/calendar.py`: 거래일 계산(다음 거래일/휴일 처리)
- `backtest/engine/signal_loader.py`: `bt_signals` 및 유니버스 조회
- `backtest/engine/portfolio.py`: 포트폴리오 구성(비중, 진입/청산 규칙)
- `backtest/engine/pricing.py`: 가격 로딩/결측 처리
- `backtest/engine/metrics.py`: CAGR/Sharpe/MDD 등 지표 계산
- `backtest/engine/runner.py`: 실행 오케스트레이션 + DB 저장

## 실행 흐름 (세부 단계)
1) **입력 파라미터 확정**
   - `signal_id`, `signal_version`, `as_of_date`
   - `top_n`, `hold_period_days`, `entry_price`, `transaction_cost_bps`
2) **시그널 로드**
   - `bt_signals`에서 `as_of_date` 기준 상위 N개 로드
   - `bt_universe_snapshot`과 조인해 당시 유니버스에 포함된 종목만 선택
3) **진입일 결정**
   - `as_of_date` 다음 거래일을 계산
   - 거래일 계산은 `bt_prices_daily`의 실제 거래일로 확인
4) **포트폴리오 구성**
   - 동일가중 또는 점수 비례 가중치 결정
   - 총합이 1이 되도록 정규화
5) **가격 시계열 생성**
   - 진입일부터 청산일까지 종목별 가격 시계열 로드
   - 결측치/상장폐지 처리 규칙 적용
6) **일별 수익률 계산**
   - 일별 포트폴리오 수익률 및 누적 수익률 산출
7) **성과 지표 계산**
   - CAGR, Sharpe, MDD, Volatility, Total Return
8) **DB 저장**
   - `bt_runs`, `bt_run_metrics`, `bt_run_equity_curve`, `bt_run_positions`

## 결측/슬리피지/거래비용 규칙 (MVP)
- 결측 가격:
  - 당일 가격 없음 → 전일 종가로 대체(가정), 혹은 포지션 제외 옵션 제공
- 슬리피지/거래비용:
  - `transaction_cost_bps` 설정(예: 5~20 bps)
  - 진입/청산 시 비용 차감

## 출력 스냅샷 (공통 포맷 권장)
스크리닝 결과와 백테스트 결과가 동일한 포맷으로 “포트폴리오 구성”을 제공해야 함.
- `portfolio_snapshot` 구조(예시):
  - `as_of_date`, `strategy_id`, `positions[{ticker, weight, rank, score}]`
  - `rules`, `universe_filter`, `generated_at`
- 저장 위치:
  - 스크리닝: `market_documents` (doc_type=`portfolio_proposal`)
  - 백테스트: `bt_runs.config_json` + `bt_run_positions`

## 입력
- `bt_signals`: 스크리닝 결과(점수/랭크)
- `bt_prices_daily`: 가격 데이터
- `bt_universe_snapshot`: 당시 유니버스 보정

## 출력 (SQLite)
- `bt_runs`, `bt_run_metrics`, `bt_run_equity_curve`, `bt_run_positions` (필수)
- `bt_run_trades` (선택)

## 관련 문서
- `docs/db/schema.sql`: bt_* 테이블 정의
- `docs/db/query-patterns.md`: 실행/조회 쿼리 패턴

## 실행 흐름
1) `as_of_date` 기준 스크리닝
   - `bt_signals`에서 `signal_id + as_of_date`로 상위 N개 선택
   - 유니버스 스냅샷과 조인해 당시 존재 종목만 포함
2) 진입일/진입가 결정
   - 진입일: `as_of_date` 다음 거래일
   - 진입가: 진입일 `open` 또는 `close` 중 하나로 고정
3) 보유/청산
   - 보유기간: `hold_period_days` 또는 `hold_period_months`
   - 청산일: 보유기간 이후 첫 거래일
4) 포트폴리오 수익률 계산
   - 동일가중 또는 설정된 가중치 적용
   - 일별 포트폴리오 가치/수익률 생성
5) 지표 계산 및 저장
   - `CAGR`, `Sharpe`, `MDD`, `Volatility`, `Total Return` 등

## 지표 계산 (필수)
- 누적 수익률
- 연환산 수익률(CAGR)
- 변동성(연환산)
- 샤프(무위험 0~2% 가정 가능)
- 최대 낙폭(MDD)

## API 연동 포인트 (초안)
- `POST /api/backtests`
  - body: `as_of_date`, `signal_id`, `top_n`, `hold_period_days`, `rebalance_freq`, `transaction_cost_bps`
  - response: `run_id`, `status`
- `GET /api/backtests`
  - 최근 실행 목록
- `GET /api/backtests/<run_id>`
  - 메타 + 핵심 지표
- `GET /api/backtests/<run_id>/equity`
  - 누적 수익률/드로우다운 시계열
- `GET /api/backtests/<run_id>/positions`
  - 종목별 진입/청산 정보

## 테스트 전략
### 단위 테스트
- 거래일 계산(휴일/주말 처리) 테스트
- 포트폴리오 가중치 합 = 1 검증
- 지표 계산식 단위 테스트(CAGR/Sharpe/MDD)

### 통합 테스트
- 소규모 샘플 데이터로 정답이 예측 가능한 케이스 작성
- 동일 입력에서 동일 결과 재현(결정론 보장)
- 단일 종목/단기 기간 sanity check

## 체크리스트
- [ ] 백테스트 실행 로직 구현 및 SQLite 저장
- [ ] 포트폴리오 구성 규칙(동일가중 MVP) 구현
- [ ] 지표 계산 모듈(Sharpe/MDD/CAGR) 구현
- [ ] 기본 API 엔드포인트 제공
- [ ] 결과 재현성 테스트 추가

## 완료 기준
- 1회 실행 시 `bt_runs`와 `bt_run_*` 테이블이 채워짐
- 프론트엔드에서 equity curve와 요약 지표를 조회 가능
- 동일 입력 재실행 시 결과 일치
