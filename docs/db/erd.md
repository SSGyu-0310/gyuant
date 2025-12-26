# ERD / 관계도 (Draft)

이 문서는 `docs/db/schema.sql`을 기준으로 한 관계 요약입니다. 실제 구현과 차이가 생기면 스키마가 우선입니다.

## 1) Market (운영 데이터)
```
market_stocks (ticker)
  1 ──── * market_prices_daily (ticker)
  1 ──── * market_volume_analysis (ticker)

market_smart_money_runs (run_id)
  1 ──── * market_smart_money_picks (run_id)

market_documents (doc_type, as_of_date, lang, model)
  (독립 테이블)
```

## 2) Backtest (bt_*)
```
bt_signal_definitions (signal_id, version)
  1 ──── * bt_signals (signal_id, signal_version)
  1 ──── * bt_runs (signal_id, signal_version)

bt_runs (run_id)
  1 ──── 1 bt_run_metrics (run_id)
  1 ──── * bt_run_equity_curve (run_id)
  1 ──── * bt_run_positions (run_id)
  1 ──── * bt_run_trades (run_id)

bt_alpha_defs (alpha_id)
  1 ──── * bt_alpha_versions (alpha_id)

bt_alpha_versions (alpha_id, version)
  0 ──── * bt_runs (alpha_id, alpha_version)
  1 ──── * bt_alpha_runs (alpha_id, alpha_version)
```

## 3) 비-FK 참조 (로직 의존)
- `bt_prices_daily`, `bt_universe_snapshot`, `bt_fundamentals`는 백테스트 로직에서 참조하지만, 대용량/적재 순서 이슈를 고려해 FK를 두지 않습니다.
- `market_documents`는 doc_type별 JSON payload로 저장되며, 별도 FK 없이 독립적으로 조회됩니다.

## 4) 정합성 규칙 요약
- `bt_signals`는 반드시 `bt_signal_definitions`에 대응하는 버전이 있어야 합니다.
- `bt_runs`는 존재하는 signal/alpha 버전만 참조합니다.
- `market_smart_money_picks`는 항상 `market_smart_money_runs`에 종속됩니다.
