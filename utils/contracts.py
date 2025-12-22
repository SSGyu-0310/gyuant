from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Contract definitions used for validation and mock generation.
# Paths are relative to the resolved DATA_DIR (see pipeline_utils.resolve_paths).

Contract = Dict[str, Any]


CONTRACTS: List[Contract] = [
    {
        "module": "smart_money_current",
        "file_path": Path("smart_money_current.json"),
        "type": "json",
        "required_keys": ["picks", "analysis_date", "summary"],
        "minimal_mock_object": {
            "analysis_date": "",
            "analysis_timestamp": "",
            "picks": [],
            "summary": {"total_analyzed": 0, "avg_score": 0},
        },
        "ui_safe_default": {"picks": []},
    },
    {
        "module": "smart_money_csv",
        "file_path": Path("smart_money_picks_v2.csv"),
        "type": "csv",
        # Columns that are actually output by smart_money_screener_v2.py
        "required_columns": [
            "ticker",
            "name",
            "composite_score",
            "grade",
            "current_price",
        ],
        "minimal_mock_rows": [],
        "ui_safe_default": [],
    },
    {
        "module": "etf_flows",
        "file_path": Path("us_etf_flows.csv"),
        "type": "csv",
        "required_columns": [
            "ticker",
            "name",
            "category",
            "current_price",
            "price_1w_pct",
            "price_1m_pct",
            "vol_ratio_5d_20d",
            "obv_change_20d_pct",
            "avg_volume_20d",
            "flow_score",
            "flow_status",
        ],
        "minimal_mock_rows": [],
        "ui_safe_default": [],
    },
    {
        "module": "etf_flow_analysis",
        "file_path": Path("etf_flow_analysis.json"),
        "type": "json",
        "required_keys": ["ai_analysis"],
        "minimal_mock_object": {"ai_analysis": "", "data_summary": {}, "timestamp": ""},
        "ui_safe_default": {"ai_analysis": ""},
    },
    {
        "module": "macro",
        "file_path": Path("macro_analysis.json"),
        "type": "json",
        "required_keys": ["macro_indicators", "ai_analysis"],
        "minimal_mock_object": {"macro_indicators": {}, "ai_analysis": "", "timestamp": ""},
        "ui_safe_default": {"macro_indicators": {}, "ai_analysis": ""},
    },
    {
        "module": "macro_en",
        "file_path": Path("macro_analysis_en.json"),
        "type": "json",
        "required_keys": ["macro_indicators", "ai_analysis"],
        "minimal_mock_object": {"macro_indicators": {}, "ai_analysis": "", "timestamp": ""},
        "ui_safe_default": {"macro_indicators": {}, "ai_analysis": ""},
    },
    {
        "module": "macro_gpt",
        "file_path": Path("macro_analysis_gpt.json"),
        "type": "json",
        "required_keys": ["macro_indicators", "ai_analysis"],
        "minimal_mock_object": {"macro_indicators": {}, "ai_analysis": "", "timestamp": ""},
        "ui_safe_default": {"macro_indicators": {}, "ai_analysis": ""},
    },
    {
        "module": "macro_gpt_en",
        "file_path": Path("macro_analysis_gpt_en.json"),
        "type": "json",
        "required_keys": ["macro_indicators", "ai_analysis"],
        "minimal_mock_object": {"macro_indicators": {}, "ai_analysis": "", "timestamp": ""},
        "ui_safe_default": {"macro_indicators": {}, "ai_analysis": ""},
    },
    {
        "module": "heatmap",
        "file_path": Path("sector_heatmap.json"),
        "type": "json",
        "required_keys": ["series"],
        "minimal_mock_object": {
            "data_date": "",
            "series": [],
        },
        "ui_safe_default": {"series": []},
    },
    {
        "module": "options",
        "file_path": Path("options_flow.json"),
        "type": "json",
        "required_keys": ["options_flow"],
        "minimal_mock_object": {"options_flow": [], "summary": {}, "timestamp": ""},
        "ui_safe_default": {"options_flow": []},
    },
    {
        "module": "ai_summaries",
        "file_path": Path("ai_summaries.json"),
        "type": "json",
        "required_keys": [],
        "minimal_mock_object": {},
        "ui_safe_default": {},
    },
    {
        "module": "calendar",
        "file_path": Path("weekly_calendar.json"),
        "type": "json",
        "required_keys": ["events"],
        "minimal_mock_object": {"week_start": "", "week_end": "", "events": []},
        "ui_safe_default": {"events": []},
    },
    {
        "module": "us_volume",
        "file_path": Path("us_volume_analysis.csv"),
        "type": "csv",
        "required_columns": [
            "ticker",
            "name",
            "supply_demand_score",
            "supply_demand_stage",
        ],
        "minimal_mock_rows": [],
        "ui_safe_default": [],
    },
    {
        "module": "us_13f",
        "file_path": Path("us_13f_holdings.csv"),
        "type": "csv",
        "required_columns": [
            "ticker",
            "institutional_pct",
            "institutional_score",
        ],
        "minimal_mock_rows": [],
        "ui_safe_default": [],
    },
    {
        "module": "us_stocks",
        "file_path": Path("us_stocks_list.csv"),
        "type": "csv",
        "required_columns": ["ticker", "name", "sector", "market"],
        "minimal_mock_rows": [],
        "ui_safe_default": [],
    },
    {
        "module": "us_prices",
        "file_path": Path("us_daily_prices.csv"),
        "type": "csv",
        "required_columns": ["ticker", "date", "close"],
        "minimal_mock_rows": [],
        "ui_safe_default": [],
    },
]


def get_contracts(modules: Optional[List[str]] = None) -> List[Contract]:
    if modules is None:
        return CONTRACTS
    module_set = set(modules)
    return [c for c in CONTRACTS if c["module"] in module_set]
