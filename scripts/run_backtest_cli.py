import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
from datetime import datetime, date
import logging
import pandas as pd
from backtest.engine.backtest_engine import BacktestEngine
from backtest.strategies.value_ranker import ValueRankerStrategy
from backtest.strategies.momentum import MomentumStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_backtest(strategy_name: str, start_date: date, end_date: date, capital: float):
    logger.info(f"Running backtest for {strategy_name}...")

    # Select Strategy
    if strategy_name.lower() == "value":
        strategy = ValueRankerStrategy("ValueRanker", {"top_n": 10})
    elif strategy_name.lower() == "momentum":
        strategy = MomentumStrategy("Momentum", {"top_n": 10})
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Initialize Engine
    engine = BacktestEngine(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        rebalance_freq="quarterly",
    )

    # Run
    results = engine.run()

    # Print Report
    metrics = results["metrics"]
    print("\n" + "=" * 50)
    print(f"Backtest Results: {strategy.name}")
    print("=" * 50)
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"CAGR:       {metrics.get('cagr', 0):.2%}")
    print(f"Vol:        {metrics.get('volatility', 0):.2%}")
    print(f"Sharpe:     {metrics.get('sharpe', 0):.2f}")
    print(f"MDD:        {metrics.get('max_drawdown', 0):.2%}")
    print(f"Total Ret:  {metrics.get('total_return', 0):.2%}")
    print("=" * 50 + "\n")

    # Optional: turnover calculation could be added to metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["value", "momentum"],
        help="Strategy to run",
    )
    parser.add_argument(
        "--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=str, default="2023-12-31", help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--capital", type=float, default=100000.0, help="Initial capital"
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    run_backtest(args.strategy, start_date, end_date, args.capital)
