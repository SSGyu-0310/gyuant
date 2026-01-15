# Improved US Stock Analysis System Blueprint - Part 5: Frontend UI (현재 코드 기준)

이 문서는 실제 템플릿 구조(`templates/`)에 맞춘 요약본입니다. 상세 구조는 템플릿 파일이 소스 오브 트루스입니다.

---

## 5.1 템플릿 구조

기본 템플릿은 `templates/index.html`이며, 아래 partials로 구성됩니다.

| 파일 | 역할 |
|---|---|
| `templates/partials/head.html` | 메타/라이브러리 로드 + 테마/기본 스타일 |
| `templates/partials/header.html` | 상단 헤더/탭 |
| `templates/partials/content_us.html` | 미국 시장 대시보드 본문 |
| `templates/partials/content_cals.html` | 경제 캘린더 섹션 |
| `templates/partials/scripts_base.html` | 공통 JS 유틸/상태 관리 |
| `templates/partials/scripts_us_fetch.html` | 데이터 fetch/렌더링 |
| `templates/partials/scripts_us_charts.html` | 차트/지표 로직 |
| `templates/partials/scripts_ai_etf.html` | ETF AI 인사이트 렌더링 |

---

## 5.2 테마/스타일

- 기본 폰트/컬러는 `head.html`의 CSS 변수(`:root`)로 관리됩니다.
- `data-theme`는 `localStorage('ui.theme')` 또는 시스템 테마를 따라 설정됩니다.
- Tailwind CDN을 사용합니다.

---

## 5.3 주요 UI 섹션

- Market Indices (상단 지수 카드)
- Smart Money Picks (좌측 테이블)
- Stock Chart + AI Summary (우측 상단)
- Sector Heatmap (우측 하단)
- Options Flow / Unusual Trades (하단 그리드)
- Economic Calendar (별도 탭)
- Ticker Drawer (오른쪽 슬라이드 패널)

---

## 변경이 필요한 경우

- 레이아웃/DOM 수정: `templates/partials/content_*.html`
- 테마/색상: `templates/partials/head.html`
- 인터랙션/렌더링: `templates/partials/scripts_*.html`
