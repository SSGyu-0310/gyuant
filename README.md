# 📈 US Market Smart Money Dashboard

미국(S&P 500) 시장 데이터를 수집하고, **거래량·ETF 플로우·옵션 플로우** 등을 분석해  
**‘스마트 머니’의 움직임을 추적**하는 대시보드 시스템입니다.

---

## PostgreSQL 로컬 개발 시작하기

PostgreSQL 전환 기준으로 기본값은 `USE_POSTGRES=true`입니다.  
CSV fallback을 사용하려면 `USE_POSTGRES=false`로 설정하세요.

### 1) .env 설정

```bash
cp .env.example .env
```

`.env`에 아래 값을 채웁니다.

```
USE_POSTGRES=true
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=gyuant_market
PG_USER=postgres
PG_PASSWORD=your_postgres_password_here
```

### 2) PostgreSQL 실행

옵션 A: Docker (권장)

```bash
docker compose up -d postgres
```

옵션 B: 로컬 설치

```bash
# 예시 (macOS/Linux)
createdb gyuant_market
createuser postgres
```

### 3) 스키마 초기화 (idempotent)

```bash
python scripts/init_db.py
```

### 4) (옵션) CSV -> PostgreSQL 마이그레이션

```bash
python scripts/migrate_csv_to_postgres.py
```

### 5) 헬스체크 (DB 연결 테스트)

```bash
RUN_DB_TESTS=1 pytest tests/test_db_connection.py -q
```

### Troubleshooting

- 5432 포트가 이미 사용 중이면 `PG_PORT` 또는 docker 포트 매핑을 변경하세요.
- 패스워드/권한 오류는 `PG_USER`, `PG_PASSWORD`, DB 권한을 확인하세요.
- CSV 마이그레이션은 `DATA_DIR`에 CSV가 있을 때만 동작합니다 (`us_market` 기본).

## ✨ Features

- **Smart Money Screener (5-Factor)**: 수급, 기술적 지표, 펀더멘털, 애널리스트 등급, 상대강도 등을 결합해 **S~F 등급** 산출
- **Options Flow**: 비정상 옵션 거래량, Put/Call Ratio 등을 통해 **시장 방향성 신호 감지**
- **AI Analysis**: **Gemini (flash)** 기반으로 수집 데이터를 투자 인사이트로 요약
- **Data Sources**: 시세/펀더멘털은 FMP 기반, 옵션 체인 및 일부 글로벌/환율은 yfinance fallback

---

## 🏗️ Project Structure

```text
.
├── flask_app.py                   # 메인 웹 서버 (Flask)
├── scheduler.py                   # 데이터 업데이트 스케줄러
├── us_market/                     # 미국 시장 핵심 로직 폴더
│   ├── create_us_daily_prices.py  # [Step 1] 가격 데이터 수집
│   ├── analyze_volume.py          # [Step 2] 거래량/수급 분석
│   ├── analyze_etf_flows.py       # [Step 3] ETF 자금 흐름 + AI 분석
│   └── smart_money_screener_v2.py # [Step 4] 종합 스크리닝 (5팩터)
├── utils/                         # 공통 유틸리티 (로거, 성능 최적화 등)
├── templates/                     # 프론트엔드 HTML 템플릿
├── tests/                         # Pytest 유닛 테스트
├── .env.example                   # 환경 변수 설정 샘플
└── requirements.txt               # 필수 패키지 목록
```

---

## 🚀 Quickstart (Local)

### Linux/macOS (WSL 포함)

```bash
# 1) 가상환경 생성
python3 -m venv .venv

# 2) 가상환경 활성화
source .venv/bin/activate

# 3) 패키지 설치
pip install -r requirements.txt

# 4) 환경 변수 파일 준비
cp .env.example .env
```

`.env`에 아래 값들을 입력하세요.

- `GOOGLE_API_KEY` : Gemini AI 분석에 필요
- `FMP_API_KEY` : FMP 기반 시장 데이터 수집에 필요
- `DATA_DIR` : 데이터가 저장될 폴더 (기본값: `us_market`)

```bash
# 예시: 편한 에디터로 열기
nano .env
```

```bash
# 5) 파이프라인 실행 (최초 1회)
python us_market/update_all.py
```

```bash
# 6) 웹 서버 실행
python flask_app.py
```

브라우저에서 접속:
- 기본값: http://localhost:5001
- `.env`의 `HOST`/`PORT`를 변경했다면 해당 값으로 접속

---

### Windows (CMD/PowerShell)

```bat
:: 1) 가상환경 생성
python -m venv .venv

:: 2) 가상환경 활성화 (CMD)
.\.venv\Scripts\activate

:: PowerShell 사용 시
.\.venv\Scripts\Activate.ps1

:: 3) 패키지 설치
pip install -r requirements.txt

:: 4) 환경 변수 파일 준비
copy .env.example .env
```

WSL(리눅스 터미널)에서는 위 Windows 명령이 동작하지 않습니다.  
WSL은 **Linux/macOS 섹션**을 그대로 따라야 합니다.

`.env`에 `GOOGLE_API_KEY`, `FMP_API_KEY`, `DATA_DIR`를 입력합니다.

```bat
:: 5) 파이프라인 실행 (최초 1회)
python us_market\update_all.py

:: 6) 웹 서버 실행
python flask_app.py
```

브라우저에서 `http://localhost:5001` 접속.

---

## 🧱 Data Pipeline (재실행/부분 실행)

전체 파이프라인:
```bash
python us_market/update_all.py
```

분석만 재실행 (가격 데이터는 이미 있을 때):
```bash
python us_market/update_all.py --analysis-only
```

개별 스크립트 실행:
```bash
python us_market/create_us_daily_prices.py
python us_market/analyze_volume.py
python us_market/analyze_etf_flows.py
python us_market/smart_money_screener_v2.py
```

스크리닝이 오래 걸리면 `SMART_MONEY_LIMIT`로 분석 대상 수를 제한할 수 있습니다.

```bash
SMART_MONEY_LIMIT=200 python us_market/update_all.py --analysis-only
```

API 호출이 많아 느릴 경우 병렬 워커를 늘릴 수 있습니다.

```bash
SMART_MONEY_WORKERS=4 python us_market/update_all.py --analysis-only
```

---

## 🖥️ Run Web Server (프론트엔드 확인)

```bash
python flask_app.py
```

브라우저에서:
- `http://localhost:5001`
- 원격 서버라면 `http://<서버IP>:5001` (또는 `.env`의 `HOST`/`PORT`)

Windows에서는 아래 배치 파일도 사용 가능합니다.
```bat
start_server.bat
```

---

## 🧪 Tests

```bash
pytest
```

---

## 🤝 Collaboration Guidelines

### ✅ 데이터 파일 업로드 금지 (중요)
- **분석 결과물/데이터 파일은 Git에 올리지 않습니다.**
- 예: `*.csv`, `*.json`, `*.parquet` 등  
  (대용량 파일로 인해 GitHub push가 막힐 수 있습니다)

> 권장: 데이터는 로컬에서 생성/캐시하고, 재현 가능한 스크립트로 파이프라인을 유지합니다.

### ✅ 의존성 관리
새 패키지를 설치했다면 반드시 `requirements.txt`를 업데이트해주세요.

```bash
pip freeze > requirements.txt
```

### ✅ 환경 변수 관리
새로운 API 키/설정이 추가되면 `.env.example`에도 동일하게 반영해야 합니다.

---

## 📌 Notes

- 데이터 소스/수집 정책, 스케줄러 운영 방식은 `scheduler.py` 및 `us_market/` 파이프라인 스크립트에 정의되어 있습니다.
- FMP Premium 제한으로 옵션 체인 및 일부 글로벌/환율은 yfinance를 유지합니다.
- 한국 시장 관련 기능은 제거되었습니다.
- 대시보드 UI는 `templates/` 기반으로 렌더링됩니다.

---
