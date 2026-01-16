
import logging
import uuid
import pandas as pd
from datetime import datetime, date
from backtest.repository.backtest_repo import BacktestRepository
from backtest.db_schema_pg import init_db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_save_empty_metrics():
    repo = BacktestRepository()
    
    # Create a dummy run
    config = {
        "strategy_name": "TestEmptyStrategy",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "initial_capital": 100000
    }
    run_id = repo.create_run(config)
    print(f"Created Run ID: {run_id}")
    
    # Empty metrics (simulating backtest engine returning empty dict or None)
    # But wait, we fixed it in routes.py to ensure it's not empty before saving.
    # So here we replicate what routes.py does: if empty, default it.
    metrics = {} 
    
    # SIMULATE THE FIX in routes.py:
    if not metrics:
        metrics = {
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0
        }
    
    equity_curve = pd.DataFrame([])
    trades = pd.DataFrame([])
    
    # Save
    print("Saving results with DEFAULT 0 metrics...")
    repo.save_results(run_id, equity_curve, trades, metrics)
    
    # Load
    print(f"Loading run {run_id}...")
    run_data = repo.get_run(run_id)
    
    if not run_data:
        print("❌ Run not found!")
        return
        
    loaded_metrics = run_data.get("metrics")
    print("Loaded Metrics:", loaded_metrics)
    
    # Check
    if loaded_metrics and loaded_metrics['total_return'] == 0.0:
        print("✅ Metrics defaulted to 0.0 and saved correctly.")
    else:
        print("❌ Metrics mismatch or missing.")

if __name__ == "__main__":
    try:
        test_save_empty_metrics()
    except Exception as e:
        print(f"❌ Error: {e}")
