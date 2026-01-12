# US Market Backend Blueprint - Part 2: 분석 및 스크리닝 (현재 코드 기준)

이 문서는 실제 구현(`us_market/*.py`)에 맞춘 요약본입니다. 상세 로직은 각 스크립트가 소스 오브 트루스입니다.

---

## 📁 대상 스크립트

| 파일명 | 설명 | 주요 출력 |
|---|---|---|
| `smart_money_screener_v2.py` | 5팩터 스크리닝 (FMP 기반) | `smart_money_picks_v2.csv`, SQLite `market_smart_money_*` |
| `sector_heatmap.py` | 섹터 퍼포먼스 히트맵 | `sector_heatmap.json`, SQLite `market_documents` |
| `options_flow.py` | 옵션 플로우 분석 | `options_flow.json`, SQLite `market_documents` |
| `insider_tracker.py` | 인사이더 매매 추적 | `insider_moves.json`, SQLite `market_documents` |
| `portfolio_risk.py` | 리스크 분석 | `portfolio_risk.json`, SQLite `market_documents` |

---

## 📦 공통 의존성

```bash
pip install -r requirements.txt
```

옵션 플로우 분석만 `yfinance`를 사용합니다. (기타 데이터는 FMP 기반)

---

## 🔧 주요 환경 변수

- `FMP_API_KEY` : FMP API 키 (필수)
- `DATA_DIR` : 출력/데이터 폴더 (기본 `us_market`)
- `USE_POSTGRES` : PostgreSQL 사용 여부 (기본 `true`)
- `USE_SQLITE` : SQLite 사용 여부 (기본 `true`)
- `SMART_MONEY_LIMIT` : 스크리닝 대상 상한(analysis-only 시 사용)
- `SMART_MONEY_WORKERS` : 스크리닝 병렬 워커 수
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` : 일부 AI 요약(옵션)

---

## 1) `smart_money_screener_v2.py`

FMP 기반으로 기술/펀더멘털/애널리스트/상대강도/수급 점수를 계산합니다.

동작 요약:
- 입력: `us_volume_analysis.csv` (또는 PostgreSQL) + FMP 시세/지표
- 출력: `smart_money_picks_v2.csv`
- SQLite `market_smart_money_runs` / `market_smart_money_picks` 저장
- 가중치(코드 기준): 수급 31.25%, 기술 25%, 펀더멘털 18.75%, 애널리스트 12.5%, 상대강도 12.5%

실행:
```bash
python us_market/smart_money_screener_v2.py
```

---

## 2) `sector_heatmap.py`

FMP 시세 데이터를 이용해 섹터 퍼포먼스를 계산합니다.

출력:
- `sector_heatmap.json`
- SQLite `market_documents` (`doc_type=sector_heatmap`)

실행:
```bash
python us_market/sector_heatmap.py
```

---

## 3) `options_flow.py` (yfinance 사용)

옵션 체인/IV/Put-Call Ratio 분석은 yfinance에 의존합니다.

출력:
- `options_flow.json`
- SQLite `market_documents` (`doc_type=options_flow`)

실행:
```bash
python us_market/options_flow.py
```

---

## 4) `insider_tracker.py`

FMP insider trading 데이터를 활용합니다.

출력:
- `insider_moves.json`
- SQLite `market_documents` (`doc_type=insider_moves`)

실행:
```bash
python us_market/insider_tracker.py
```

---

## 5) `portfolio_risk.py`

FMP 가격 데이터를 기반으로 포트폴리오 리스크를 계산합니다.

출력:
- `portfolio_risk.json`
- SQLite `market_documents` (`doc_type=portfolio_risk`)

실행:
```bash
python us_market/portfolio_risk.py
```
