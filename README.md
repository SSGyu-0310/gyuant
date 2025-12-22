# 📈 US/KO Market Smart Money Dashboard

미국(S&P 500) 및 한국 시장 데이터를 수집하고, **거래량·기관 보유량(13F)·ETF 플로우·옵션 플로우** 등을 분석해  
**‘스마트 머니’의 움직임을 추적**하는 대시보드 시스템입니다.

---

## ✨ Features

- **Smart Money Screener (6-Factor)**: 수급, 기술적 지표, 펀더멘털, 애널리스트 등급, 상대강도 등을 결합해 **S~F 등급** 산출
- **Institutional Support (13F)**: 13F 공시 기반 **기관 매집/보유 패턴 분석**
- **Options Flow**: 비정상 옵션 거래량, Put/Call Ratio 등을 통해 **시장 방향성 신호 감지**
- **AI Analysis**: **Gemini (flash)** 기반으로 수집 데이터를 투자 인사이트로 요약

---

## 🏗️ Project Structure

```text
.
├── flask_app.py                   # 메인 웹 서버 (Flask)
├── scheduler.py                   # 데이터 업데이트 스케줄러
├── us_market/                     # 미국 시장 핵심 로직 폴더
│   ├── create_us_daily_prices.py  # [Step 1] 가격 데이터 수집
│   ├── analyze_volume.py          # [Step 2] 거래량/수급 분석
│   ├── analyze_13f.py             # [Step 3] 기관 보유 분석 (13F)
│   ├── analyze_etf_flows.py       # [Step 4] ETF 자금 흐름 + AI 분석
│   └── smart_money_screener_v2.py # [Step 5] 종합 스크리닝 (6팩터)
├── utils/                         # 공통 유틸리티 (로거, 성능 최적화 등)
├── templates/                     # 프론트엔드 HTML 템플릿
├── tests/                         # Pytest 유닛 테스트
├── .env.example                   # 환경 변수 설정 샘플
└── requirements.txt               # 필수 패키지 목록
```

---

## 🚀 Quickstart (Local)

### 1) 가상환경 생성 및 활성화

```bash
python -m venv .venv
```

- **Mac/Linux**
```bash
source .venv/bin/activate
```

- **Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
```

- **Windows (CMD)**
```bat
.\.venv\Scripts\activate
```

### 2) 패키지 설치

```bash
pip install -r requirements.txt
```

### 3) 환경 변수 설정

`.env.example` 파일을 복사해 `.env`를 만들고 필요한 값을 입력합니다.

```bash
# Mac/Linux
cp .env.example .env
```

```bat
:: Windows (CMD)
copy .env.example .env
```

필수 환경 변수:

- `GOOGLE_API_KEY` : Gemini AI 분석에 필요
- `DATA_DIR` : 데이터가 저장될 폴더 (기본값: `us_market`)

> ⚠️ `.env`는 **절대 Git에 커밋하지 마세요.** (API 키 포함)

---

## 🧱 Data Pipeline (최초 1회)

최초 1회 아래 순서대로 실행해 기본 데이터/분석 산출물을 생성합니다.

```bash
python us_market/create_us_daily_prices.py
python us_market/analyze_volume.py
python us_market/analyze_13f.py
python us_market/analyze_etf_flows.py
python us_market/smart_money_screener_v2.py
```

---

## 🖥️ Run Web Server

### Option A) Windows 배치 실행

```bat
start_server.bat
```

### Option B) 직접 실행 (예: Flask)

프로젝트 구성에 맞춰 실행 커맨드가 다를 수 있습니다.  
일반적으로 아래 형태로 실행합니다.

```bash
python flask_app.py
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
- 대시보드 UI는 `templates/` 기반으로 렌더링됩니다.

---
