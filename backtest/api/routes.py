from flask import Blueprint, jsonify, request, render_template
from backtest.repository.backtest_repo import BacktestRepository
from backtest.engine.backtest_engine import BacktestEngine
from backtest.strategies.smart_money_topn import SmartMoneyTopN
from backtest.strategies.base import StrategyContext
from datetime import datetime, date
import threading
import logging
import traceback

logger = logging.getLogger(__name__)
bp = Blueprint("backtest_api", __name__, url_prefix="/api/backtest")
repo = BacktestRepository()


# Background runner
def run_backtest_async(run_id, config):
    try:
        print(f"\n{'='*60}")
        print(f"🚀 [BACKTEST] Starting run: {run_id[:8]}...")
        print(f"   Strategy: {config.get('strategy_name', 'SmartMoneyTopN')}")
        print(f"   Period: {config.get('start_date')} ~ {config.get('end_date')}")
        print(f"   Capital: ${float(config.get('initial_capital', 100000)):,.0f}")
        print(f"{'='*60}")

        start_date = datetime.strptime(config["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(config["end_date"], "%Y-%m-%d").date()
        initial_capital = float(config.get("initial_capital", 100000))

        # Strategy Factory (Simplified)
        strategy_name = config.get("strategy_name", "SmartMoneyTopN")
        if strategy_name == "SmartMoneyTopN":
            strategy = SmartMoneyTopN(strategy_name, params=config)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        print(f"📊 [BACKTEST] Initializing engine...")

        engine = BacktestEngine(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_freq=config.get("rebalance_freq", "quarterly"),
        )

        print(f"⏳ [BACKTEST] Running simulation...")
        results = engine.run()

        print(f"💾 [BACKTEST] Saving results...")
        repo.save_results(
            run_id=run_id,
            equity_curve=results["equity_curve"],
            trades=results["trades"],
            metrics=results["metrics"],
        )

        metrics = results.get("metrics", {})
        if not metrics:
            metrics = {
                "cagr": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0
            }
            
        print(f"\n{'='*60}")
        print(f"✅ [BACKTEST] Run {run_id[:8]} COMPLETED!")
        print(f"   Total Return: {metrics.get('total_return', 0)*100:.2f}%")
        print(f"   CAGR: {metrics.get('cagr', 0)*100:.2f}%")
        print(f"   Sharpe: {metrics.get('sharpe', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"   Trades: {len(results.get('trades', []))}")
        print(f"{'='*60}\n")

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Backtest run {run_id} failed: {e}\n{error_detail}")
        print(f"\n{'='*60}")
        print(f"❌ [BACKTEST ERROR] Run {run_id[:8]} FAILED!")
        print(f"   Error: {str(e)}")
        print(f"{'='*60}")
        print(f"Traceback:\n{error_detail}")
        repo.update_status(run_id, "failed", str(e))


@bp.route("/run", methods=["POST"])
def create_run():
    config = request.json
    run_id = repo.create_run(config)

    # Run in background thread
    thread = threading.Thread(target=run_backtest_async, args=(run_id, config))
    thread.start()

    return jsonify({"run_id": run_id, "status": "queued"})


@bp.route("/runs/<run_id>", methods=["GET"])
def get_run_status(run_id):
    run = repo.get_run(run_id)
    if run:
        return jsonify(run)
    return jsonify({"error": "Run not found"}), 404


@bp.route("/runs", methods=["GET"])
def list_runs():
    runs = repo.get_all_runs(limit=50)
    return jsonify({"runs": runs})
