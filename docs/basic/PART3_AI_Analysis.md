# US Market Backend Blueprint - Part 3: AI 분석 (현재 코드 기준)

이 문서는 실제 구현(`us_market/*.py`)에 맞춘 요약본입니다. 상세 로직은 각 스크립트가 소스 오브 트루스입니다.

---

## 📁 대상 스크립트

| 파일명 | 설명 | 주요 출력 |
|---|---|---|
| `macro_analyzer.py` | 매크로 지표 수집 + Gemini 요약 | `macro_analysis.json`, `macro_analysis_en.json` |
| `ai_summary_generator.py` | 종목별 AI 요약 | `ai_summaries.json` |
| `final_report_generator.py` | Top 10 리포트 | `final_top10_report.json`, `smart_money_current.json` |
| `economic_calendar.py` | 경제 캘린더 + AI 요약(옵션) | `weekly_calendar.json` |

---

## 📦 공통 의존성

```bash
pip install -r requirements.txt
```

---

## 🔧 주요 환경 변수

- `GOOGLE_API_KEY` or `GEMINI_API_KEY` : Gemini API 키 (필수)
- `FMP_API_KEY` : FMP API 키 (매크로/캘린더 데이터 수집)
- `DATA_DIR` : 출력/데이터 폴더 (기본 `us_market`)

참고:
- `OPENAI_API_KEY`는 현재 코드에서 사용하지 않습니다(향후 확장용).

---

## 1) `macro_analyzer.py`

FMP로 매크로 지표를 수집하고 Gemini(HTTP 호출, `gemini-2.0-flash`)로 요약합니다.

출력:
- `macro_analysis.json` (ko)
- `macro_analysis_en.json` (en)
- SQLite `market_documents` (`doc_type=macro_analysis`, `model=gemini`)

실행:
```bash
python us_market/macro_analyzer.py
```

---

## 2) `ai_summary_generator.py`

스마트머니 상위 종목을 대상으로 Gemini 요약을 생성합니다.

출력:
- `ai_summaries.json`
- SQLite `market_documents` (`doc_type=ai_summaries`)

실행:
```bash
python us_market/ai_summary_generator.py
```

---

## 3) `final_report_generator.py`

정량 스코어 + AI 요약을 결합해 최종 Top 10 리포트를 생성합니다.

출력:
- `final_top10_report.json`
- `smart_money_current.json` (대시보드용 스냅샷)
- SQLite `market_documents` (`doc_type=final_top10_report`, `doc_type=smart_money_current`)

실행:
```bash
python us_market/final_report_generator.py
```

---

## 4) `economic_calendar.py`

FMP 경제 캘린더를 수집하고, 고충격 이벤트에 한해 Gemini 요약을 생성합니다.

출력:
- `weekly_calendar.json`
- SQLite `market_documents` (`doc_type=calendar`)

실행:
```bash
python us_market/economic_calendar.py
```

---

## GPT 관련 안내

현재 백엔드는 Gemini 결과만 생성합니다. UI에서 `model=gpt` 요청이 들어오면 Gemini 결과로 폴백됩니다.
