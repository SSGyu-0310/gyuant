from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from .contracts import get_contracts
from backtest.db_schema import get_connection, init_db

ALLOW_FILE_FALLBACK = os.getenv("ALLOW_FILE_FALLBACK", "false").lower() == "true"


def resolve_paths() -> Dict[str, Path]:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(os.getenv("DATA_DIR", base_dir / "us_market")).resolve()
    log_dir = Path(os.getenv("LOG_DIR", base_dir / "logs")).resolve()
    history_dir = Path(os.getenv("HISTORY_DIR", data_dir / "history")).resolve()
    run_state = data_dir / "run_state.json"
    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "log_dir": log_dir,
        "history_dir": history_dir,
        "run_state": run_state,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_json_atomic(path: Path, data: Any) -> None:
    content = json.dumps(data, ensure_ascii=False, indent=2)
    write_text_atomic(path, content)


def write_csv_atomic(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _log_gap(record: Dict[str, Any]) -> None:
    paths = resolve_paths()
    ensure_dir(paths["log_dir"])
    log_path = paths["log_dir"] / "pipeline_gaps.jsonl"
    record["detected_at"] = datetime.now().astimezone().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mock_json(contract: Dict[str, Any], path: Path) -> None:
    obj = contract.get("minimal_mock_object") or contract.get("ui_safe_default") or {}
    write_json_atomic(path, obj)
    _log_gap(
        {
            "module": contract["module"],
            "file": str(path),
            "reason": "MOCK_CREATED",
            "details": "json mock created",
        }
    )


def _mock_csv(contract: Dict[str, Any], path: Path) -> None:
    cols = contract.get("required_columns", [])
    rows = contract.get("minimal_mock_rows", [])
    write_csv_atomic(path, rows, cols)
    _log_gap(
        {
            "module": contract["module"],
            "file": str(path),
            "reason": "MOCK_CREATED",
            "details": "csv mock created",
        }
    )


def _validate_db_contract(contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    table = contract.get("db_table")
    if not table:
        return None
    if not init_db():
        return {"status": "db_unavailable", "source": "db", "db_table": table}

    where = contract.get("db_where")
    params = contract.get("db_params") or []
    query = f"SELECT 1 FROM {table}"
    if where:
        query += f" WHERE {where}"
    query += " LIMIT 1"

    conn = get_connection()
    try:
        row = conn.execute(query, params).fetchone()
        if row:
            return {"status": "ok", "source": "db", "db_table": table}
        return {"status": "missing", "source": "db", "db_table": table}
    except Exception as e:
        return {"status": "db_error", "source": "db", "db_table": table, "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def validate_contract(contract: Dict[str, Any], paths: Optional[Dict[str, Path]] = None) -> Dict[str, Any]:
    """
    Validate a data contract file.
    
    CRITICAL: This function will NEVER overwrite existing files.
    Mock files are only created when the file is completely missing.
    All other issues are logged but existing data is preserved.
    """
    paths = paths or resolve_paths()
    data_dir = paths["data_dir"]
    path = data_dir / contract["file_path"]
    ensure_dir(path.parent)

    result = {
        "module": contract["module"],
        "file": str(path),
        "status": "ok",
        "mocked": False,
        "source": "file",
    }

    db_result = _validate_db_contract(contract)
    if db_result:
        result.update(db_result)
        if db_result["status"] == "ok":
            return result
        if not ALLOW_FILE_FALLBACK:
            _log_gap(
                {
                    "module": contract["module"],
                    "file": str(path),
                    "reason": "DB_MISSING",
                    "details": {
                        "table": db_result.get("db_table"),
                        "status": db_result.get("status"),
                        "error": db_result.get("error"),
                    },
                }
            )
            return result
        result["source"] = "file"
        result["status"] = "ok"
        result["mocked"] = False
        result.pop("error", None)

    # ONLY create mock file when file is completely missing
    if not path.exists():
        _log_gap({"module": contract["module"], "file": str(path), "reason": "MISSING_FILE", "details": "Creating mock file"})
        if contract["type"] == "json":
            _mock_json(contract, path)
        else:
            _mock_csv(contract, path)
        result.update({"status": "mocked", "mocked": True})
        return result

    # File exists - from here on, we NEVER overwrite, only log issues
    if contract["type"] == "json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            # Log parse error but DO NOT overwrite - preserve existing file
            _log_gap({
                "module": contract["module"], 
                "file": str(path), 
                "reason": "JSON_PARSE_ERROR", 
                "details": {"error": str(e), "action": "preserved_existing"}
            })
            result.update({"status": "parse_error", "mocked": False, "error": str(e)})
            return result

        required_keys = contract.get("required_keys") or []
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            # Log schema mismatch but DO NOT overwrite
            _log_gap({
                "module": contract["module"],
                "file": str(path),
                "reason": "BAD_SCHEMA",
                "details": {"missing_keys": missing_keys, "action": "preserved_existing"},
            })
            result.update({"status": "schema_warning", "mocked": False, "missing_keys": missing_keys})
            return result

    elif contract["type"] == "csv":
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            # Log parse error but DO NOT overwrite - preserve existing file
            _log_gap({
                "module": contract["module"], 
                "file": str(path), 
                "reason": "CSV_PARSE_ERROR", 
                "details": {"error": str(e), "action": "preserved_existing"}
            })
            result.update({"status": "parse_error", "mocked": False, "error": str(e)})
            return result

        if not rows:
            # Log empty file but DO NOT overwrite - let smart_update.py handle refresh
            _log_gap({
                "module": contract["module"], 
                "file": str(path), 
                "reason": "EMPTY_FILE", 
                "details": {"action": "preserved_existing", "note": "smart_update will refresh"}
            })
            result.update({"status": "empty_file", "mocked": False})
            return result

        header = rows[0]
        required_columns = contract.get("required_columns") or []
        missing_cols = [c for c in required_columns if c not in header]
        if missing_cols:
            # Log schema mismatch but DO NOT overwrite
            _log_gap({
                "module": contract["module"],
                "file": str(path),
                "reason": "BAD_SCHEMA",
                "details": {"missing_columns": missing_cols, "action": "preserved_existing"},
            })
            result.update({"status": "schema_warning", "mocked": False, "missing_columns": missing_cols})
            return result

    return result


def ensure_contracts(modules: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    paths = resolve_paths()
    ensure_dir(paths["data_dir"])
    ensure_dir(paths["log_dir"])
    ensure_dir(paths["history_dir"])

    results = []
    for contract in get_contracts(modules):
        res = validate_contract(contract, paths)
        results.append(res)
    return results


def copy_atomic(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    with NamedTemporaryFile("wb", delete=False, dir=str(dest.parent)) as tmp:
        shutil.copyfileobj(open(src, "rb"), tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)
