#!/usr/bin/env python3
"""
Daily snapshot utility (M1)
Usage:
    python scripts/daily_snapshot.py
Notes:
    - Uses DATA_DIR/LOG_DIR/HISTORY_DIR env vars if set, defaults to repo/us_market and logs/.
    - Generates history/picks_YYYYMMDD.json and updates run_state.json.
    - Attempts to run us_market/update_all.py if present (errors are tolerated and logged).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import subprocess

from utils.pipeline_utils import (
    copy_atomic,
    ensure_contracts,
    resolve_paths,
    write_json_atomic,
)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def build_snapshot_payload(data_dir: Path) -> Dict[str, Any]:
    src = data_dir / "smart_money_current.json"
    if src.exists():
        payload = _load_json(src, {})
        if isinstance(payload, dict) and "picks" in payload:
            return payload
    # Fallback minimal structure
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "analysis_date": now.split("T")[0],
        "analysis_timestamp": now,
        "picks": [],
        "summary": {"total_analyzed": 0, "avg_score": 0},
    }


def run_pipeline_if_available(data_dir: Path) -> None:
    script = data_dir / "update_all.py"
    if not script.exists():
        return
    try:
        subprocess.run([sys.executable, str(script)], cwd=str(data_dir), check=True)
    except Exception:
        # tolerate failure; gaps will be logged by validators
        pass


def update_run_state(run_state_path: Path, success: bool, error: str = "") -> None:
    state = _load_json(run_state_path, {})
    now = datetime.utcnow().astimezone().isoformat()
    if success:
        state["last_success_at"] = now
    else:
        state["last_failure_at"] = now
        state["error_summary"] = error
    write_json_atomic(run_state_path, state)


def main() -> int:
    paths = resolve_paths()
    data_dir = paths["data_dir"]
    history_dir = paths["history_dir"]
    run_state_path = paths["run_state"]

    ensure_contracts()  # validates and mocks if needed

    try:
        run_pipeline_if_available(data_dir)
        ensure_contracts()  # validate again after pipeline run

        payload = build_snapshot_payload(data_dir)
        snapshot_name = f"picks_{datetime.now().strftime('%Y%m%d')}.json"
        snapshot_path = history_dir / snapshot_name
        write_json_atomic(snapshot_path, payload)
        update_run_state(run_state_path, success=True)
        return 0
    except Exception as e:
        update_run_state(run_state_path, success=False, error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
