# US Market Backend Blueprint - Part 1: 데이터 수집 (현재 코드 기준)

이 문서는 실제 구현(`us_market/*.py`)에 맞춘 요약본입니다. 상세 로직은 각 스크립트가 소스 오브 트루스입니다.

---

## 📁 대상 스크립트

| 파일명 | 설명 | 주요 출력 |
|---|---|---|
| `create_us_daily_prices.py` | S&P 500 일봉 수집 (FMP) | `us_daily_prices.csv` (옵션), PostgreSQL `market.daily_prices`, SQLite `bt_universe_snapshot` |
| `analyze_volume.py` | 거래량/수급 분석 | `us_volume_analysis.csv`, PostgreSQL `factors.volume_analysis` |
| `analyze_etf_flows.py` | ETF 자금 흐름 분석 + AI 요약(옵션) | `us_etf_flows.csv`, `etf_flow_analysis.json` |

---

## 📦 공통 의존성

```bash
pip install -r requirements.txt
```

PostgreSQL dual-write를 사용할 경우 `psycopg`/`sqlalchemy`가 필요합니다(이미 requirements.txt 포함).

---

## 🔧 주요 환경 변수

- `FMP_API_KEY` : FMP API 키 (필수)
- `DATA_DIR` : 출력/데이터 폴더 (기본 `us_market`)
- `USE_POSTGRES` : PostgreSQL 사용 여부 (기본 `true`)
- `USE_SQLITE` : SQLite 사용 여부 (기본 `true`)
- `WRITE_CSV` : CSV 출력 여부 (기본 `true`)
- `FMP_DOWNLOAD_WORKERS` : 가격 수집 워커 수 (기본 4)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` : ETF AI 요약(옵션)

---

## 1) `create_us_daily_prices.py`

S&P 500 종목의 일봉(OHLCV) 데이터를 FMP로 수집합니다.

동작 요약:
- FMP 기반 OHLCV 수집 (심볼 매핑 포함)
- `WRITE_CSV=true`면 `us_daily_prices.csv` 생성
- `USE_POSTGRES=true`면 PostgreSQL `market.daily_prices`로 dual-write
- SQLite `bt_universe_snapshot`에 유니버스 스냅샷 저장
- `backtest/universe_snapshots/YYYY-MM-DD.csv` 스냅샷 생성

실행:
```bash
python us_market/create_us_daily_prices.py
```

---

## 2) `analyze_volume.py`

수급/거래량 지표를 계산합니다.

동작 요약:
- PostgreSQL(우선) 또는 CSV 폴백으로 가격 데이터 로드
- OBV, AD line, MFI, 거래량 서지 등을 계산
- `us_volume_analysis.csv` 출력
- `USE_POSTGRES=true`면 `factors.volume_analysis`로 저장

실행:
```bash
python us_market/analyze_volume.py
```

---

## 3) `analyze_etf_flows.py`

ETF 흐름과 모멘텀을 계산합니다.

동작 요약:
- FMP 가격 데이터를 사용해 흐름 점수 계산
- `us_etf_flows.csv`, `etf_flow_analysis.json` 생성
- Gemini AI 요약은 `GEMINI_API_KEY`가 있을 때만 수행

실행:
```bash
python us_market/analyze_etf_flows.py
```
