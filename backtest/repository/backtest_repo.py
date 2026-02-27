import json
import logging
import uuid
import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

from backtest.db_schema_pg import (
    get_engine,
    get_session,
    BacktestRun,
    BacktestRunMetrics,
    BacktestEquityCurve,
    BacktestTrade,
    BacktestPosition,
)

logger = logging.getLogger(__name__)


class BacktestRepository:
    """
    Repository for saving backtest results to PostgreSQL.
    """

    _schema_checked = False

    def __init__(self):
        if not BacktestRepository._schema_checked:
            self._ensure_schema_and_tables()
            self._ensure_run_columns()
            BacktestRepository._schema_checked = True

    def _ensure_schema_and_tables(self):
        """Ensure all required schemas and tables exist."""
        from backtest.db_schema_pg import (
            create_schemas,
            Base,
            get_engine as get_schema_engine,
        )

        try:
            engine = get_schema_engine()
            create_schemas(engine)
            Base.metadata.create_all(engine)
            logger.info("✅ Ensured all backtest schemas and tables exist")
        except Exception as e:
            logger.warning(f"Could not auto-create schemas/tables: {e}")

    def _ensure_run_columns(self):
        engine = get_engine()
        with engine.begin() as conn:
            # Check which tables exist before trying to ALTER them
            existing_tables = set()
            check_query = text("""
                SELECT table_schema || '.' || table_name as full_name
                FROM information_schema.tables
                WHERE table_schema IN ('backtest', 'direct_indexing', 'factors', 'market')
            """)
            result = conn.execute(check_query)
            for row in result:
                existing_tables.add(row[0])

            # Backtest runs table columns - add ALL required columns
            if "backtest.runs" in existing_tables:
                run_statements = [
                    # Core columns that might be missing
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS as_of_date DATE;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS start_date DATE;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS end_date DATE;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS universe VARCHAR(50);",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS config_json TEXT;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS rebalance_freq VARCHAR(20) DEFAULT 'quarterly';",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'queued';",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS error_message TEXT;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS finished_at TIMESTAMP;",
                    # Additional columns
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS signal_id VARCHAR(50);",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS signal_version VARCHAR(20);",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS alpha_id VARCHAR(50);",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS alpha_version VARCHAR(20);",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS top_n INTEGER;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS hold_period_days INTEGER;",
                    "ALTER TABLE backtest.runs ADD COLUMN IF NOT EXISTS transaction_cost_bps NUMERIC(6, 2) DEFAULT 10;",
                ]
                for stmt in run_statements:
                    conn.execute(text(stmt))

            # Equity curve table - add cash column if table exists
            if "backtest.equity_curve" in existing_tables:
                conn.execute(
                    text(
                        "ALTER TABLE backtest.equity_curve ADD COLUMN IF NOT EXISTS cash NUMERIC(15, 2);"
                    )
                )

        logger.info("✅ Ensured postgres backtest schema columns exist")

    def create_run(self, config: Dict[str, Any]) -> str:
        """
        Create a new backtest run record. Returns run_id.
        """
        session: Session = get_session()
        try:
            run_id = str(uuid.uuid4())

            config_json = json.dumps(config, default=str)

            run = BacktestRun(
                run_id=run_id,
                strategy_name=config.get("strategy_name", "Unknown"),
                start_date=config.get("start_date"),
                end_date=config.get("end_date"),
                as_of_date=datetime.now().date(),
                universe=config.get("universe", "US_ALL"),
                config_json=config_json,
                rebalance_freq=config.get("rebalance_freq", "quarterly"),
                status="running",
                created_at=datetime.now(),
            )
            session.add(run)
            session.commit()
            return run_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create backtest run: {e}")
            raise
        finally:
            session.close()

    def update_status(self, run_id: str, status: str, error_msg: str = None):
        """
        Update run status.
        """
        session: Session = get_session()
        try:
            run = session.query(BacktestRun).filter_by(run_id=run_id).first()
            if run:
                run.status = status
                run.error_message = error_msg
                if status in ["finished", "failed"]:
                    run.finished_at = datetime.now()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update status for {run_id}: {e}")
        finally:
            session.close()

    def save_results(
        self,
        run_id: str,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: Dict[str, float],
    ):
        """
        Save all results (metrics, equity curve, trades).
        """
        session: Session = get_session()
        try:
            # 1. Save Metrics
            run_metrics = BacktestRunMetrics(
                run_id=run_id,
                cagr=metrics.get("cagr"),
                volatility=metrics.get("volatility"),
                sharpe=metrics.get("sharpe"),
                max_drawdown=metrics.get("max_drawdown"),
                total_return=metrics.get("total_return"),
                created_at=datetime.now(),
            )
            session.add(run_metrics)

            # 2. Save Equity Curve
            # Batch insert
            curves = []
            for _, row in equity_curve.iterrows():
                curves.append(
                    BacktestEquityCurve(
                        run_id=run_id,
                        date=row["date"],
                        equity=row["equity"],
                        cash=row["cash"],
                    )
                )
            session.bulk_save_objects(curves)

            # 3. Save Trades
            batch_trades = []
            for _, row in trades.iterrows():
                batch_trades.append(
                    BacktestTrade(
                        run_id=run_id,
                        trade_id=row.get("trade_id", str(uuid.uuid4())),
                        ticker=row["ticker"],
                        side=row["side"],
                        trade_date=row["trade_date"],
                        price=row["price"],
                        shares=row["shares"],
                        fee=row["fee"],
                    )
                )
            session.bulk_save_objects(batch_trades)

            run = session.query(BacktestRun).filter_by(run_id=run_id).first()
            if run:
                run.status = "finished"
                run.finished_at = datetime.now()

            session.commit()
            logger.info(f"Successfully saved results for run {run_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save results for {run_id}: {e}")
            self.update_status(run_id, "failed", str(e))
        finally:
            session.close()

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get a single backtest run with all results.
        Returns run info, metrics, equity curve, and trades.
        """
        session: Session = get_session()
        try:
            run = session.query(BacktestRun).filter_by(run_id=run_id).first()
            if not run:
                return None

            run_data = {
                "run_id": run.run_id,
                "strategy_name": run.strategy_name,
                "start_date": str(run.start_date) if run.start_date else None,
                "end_date": str(run.end_date) if run.end_date else None,
                "as_of_date": str(run.as_of_date) if run.as_of_date else None,
                "universe": run.universe,
                "rebalance_freq": run.rebalance_freq,
                "status": run.status,
                "error_message": run.error_message,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }

            metrics = session.query(BacktestRunMetrics).filter_by(run_id=run_id).first()
            if metrics:
                run_data["metrics"] = {
                    "cagr": float(metrics.cagr) if metrics.cagr is not None else None,
                    "volatility": float(metrics.volatility)
                    if metrics.volatility is not None
                    else None,
                    "sharpe": float(metrics.sharpe)
                    if metrics.sharpe is not None
                    else None,
                    "max_drawdown": float(metrics.max_drawdown)
                    if metrics.max_drawdown is not None
                    else None,
                    "total_return": float(metrics.total_return)
                    if metrics.total_return is not None
                    else None,
                    "win_rate": float(metrics.win_rate)
                    if metrics.win_rate is not None
                    else None,
                    "turnover": float(metrics.turnover)
                    if metrics.turnover is not None
                    else None,
                }
            else:
                run_data["metrics"] = None

            equity_curves = (
                session.query(BacktestEquityCurve)
                .filter_by(run_id=run_id)
                .order_by(BacktestEquityCurve.date)
                .all()
            )
            run_data["equity_curve"] = [
                {
                    "date": str(ec.date),
                    "equity": float(ec.equity) if ec.equity is not None else None,
                    "cash": float(ec.cash) if ec.cash is not None else None,
                    "returns": float(ec.returns) if ec.returns is not None else None,
                    "drawdown": float(ec.drawdown) if ec.drawdown is not None else None,
                }
                for ec in equity_curves
            ]

            trades = (
                session.query(BacktestTrade)
                .filter_by(run_id=run_id)
                .order_by(BacktestTrade.trade_date)
                .all()
            )
            run_data["trades"] = [
                {
                    "trade_id": t.trade_id,
                    "ticker": t.ticker,
                    "side": t.side,
                    "trade_date": str(t.trade_date),
                    "price": float(t.price) if t.price is not None else None,
                    "shares": float(t.shares) if t.shares is not None else None,
                    "fee": float(t.fee) if t.fee is not None else None,
                }
                for t in trades
            ]

            return run_data
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None
        finally:
            session.close()

    def get_all_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get all backtest runs (summary only).
        """
        session: Session = get_session()
        try:
            runs = (
                session.query(BacktestRun)
                .order_by(BacktestRun.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "run_id": r.run_id,
                    "strategy_name": r.strategy_name,
                    "start_date": str(r.start_date) if r.start_date else None,
                    "end_date": str(r.end_date) if r.end_date else None,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                }
                for r in runs
            ]
        except Exception as e:
            logger.error(f"Failed to get all runs: {e}")
            return []
        finally:
            session.close()
