# US/KO Market Dashboard

## Quickstart (local)
1. `python -m venv .venv && .venv/Scripts/activate` (Windows) 또는 `source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `.env.example`를 참고해 `.env` 작성 (DATA_DIR 기본 `us_market`)
4. `python flask_app.py`로 개발 서버 실행 (기본 포트 5001)

## Tests
```
pytest
```

## Production (Gunicorn)
```
gunicorn -w 2 -b 0.0.0.0:8000 "flask_app:app"
```

## Scheduler / Snapshots
```
python scripts/daily_snapshot.py
```

### AI summary env vars
- `AI_SUMMARY_MAX_CHARS` (default 1500): max characters for AI summary (truncates safely)
- `AI_SUMMARY_TTL_SEC` (default 1800): TTL (seconds) for ticker+lang AI summary cache
- `AI_SUMMARY_REGEN_COOLDOWN_SEC` (default 30): cooldown (seconds) for forced regenerate calls
- `PERF_ENABLE_THREADS` (default 0) / `PERF_MAX_WORKERS` (default 4): control optional safe threading for network-bound work
- Options scoring thresholds (F1-A):
  - `OPT_PREMIUM_HIGH_USD` (default 1000000)
  - `OPT_PREMIUM_MID_USD` (default 300000)
  - `OPT_VOL_OI_RATIO_HIGH` (default 5), `OPT_VOL_OI_RATIO_MID` (default 3)
  - `OPT_VOLUME_LARGE` (default 5000), `OPT_IV_HIGH` (default 80)
