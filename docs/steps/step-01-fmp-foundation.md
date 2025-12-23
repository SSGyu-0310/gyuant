# Step 1: FMP Foundation (Client + Config + Safe Toggle)

## 목표
FMP API를 안정적으로 호출할 수 있는 공통 클라이언트를 만들고, 기존 yfinance 호출을 단계적으로 교체할 수 있는 토글 구조를 준비한다.

## 범위
- 포함: FMP 클라이언트, 환경 변수, 호출 로깅/리트라이, 최소 통합 지점 1~2개.
- 제외: 전면 교체(이는 Step 2에서 진행).

## 사전 조건
- `.env`에 `FMP_API_KEY` 준비.
- 로컬 실행 기준 (`DATA_DIR` 유지).

## 작업 지침 (에이전트용)
### 1) 공통 FMP 클라이언트 추가
권장 위치: `utils/fmp_client.py`
필수 기능:
- `FMP_API_KEY`, `FMP_BASE_URL` 읽기 (기본값: `https://financialmodelingprep.com`)
- `get_json(path, params)` 구현: `apikey` 자동 추가
- 429/5xx 리트라이: 지수 백오프(최소 2회 이상)
- timeout 기본값(예: 10~20초)
- 로깅: endpoint + status + latency

### 2) 호출 레이어 규격화
다음 기본 메서드를 제공:
- `quote(symbols: list[str])` -> 현재가/전일가
- `profile(symbol)` -> sector, company name
- `historical_price_full(symbol, from, to)` -> 일봉
- `historical_chart(symbol, interval)` -> 분봉(필요 시)

### 3) 데이터 소스 토글
환경 변수 예시:
- `DATA_PROVIDER=FMP|YF`
- 기본값은 `FMP`로 설정, 실패 시 `YF` fallback 가능

### 4) 최소 통합 포인트 적용
다음 중 1~2개를 FMP로 전환하여 검증:
- `flask_app.py:fetch_price_map`
- `flask_app.py:get_sector`

### 5) 테스트 기반 확인
`tests/conftest.py`에 FMP client mock 추가:
- `mock_fmp_client` fixture
- `flask_app`에서 FMP client 주입 가능 구조 확보

## 산출물
- `utils/fmp_client.py`
- `docs` 또는 `.env`에 FMP 설정 안내
- 1~2개 API 엔드포인트 FMP 전환

## 유의 사항
- Step 1에서는 yfinance 제거 금지 (Step 2에서 전체 전환 후 제거).
- 기존 산출물 스키마는 바꾸지 않는다.

## 완료 기준
- FMP 기반 호출이 로컬에서 성공 (쿼트/프로필)
- 테스트에서 mock으로 정상 동작
- 기존 기능 영향 없음

## 참고
- FMP 엔드포인트: `docs/fmp-migration-guide.md`

