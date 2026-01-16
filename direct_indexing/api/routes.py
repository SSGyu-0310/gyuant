from flask import Blueprint, jsonify, request
from direct_indexing.portfolio_builder import PortfolioBuilder
from direct_indexing.rebalance_engine import RebalanceEngine
from utils.data_access import get_prices  # Reuse existing price fetcher
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)
bp = Blueprint("direct_indexing_api", __name__, url_prefix="/api/direct-indexing")

builder = PortfolioBuilder()
engine = RebalanceEngine()


@bp.route("/portfolios", methods=["POST"])
def create_portfolio():
    try:
        data = request.json
        name = data.get("name")
        benchmark = data.get("benchmark", "SPY")
        initial_capital = float(data.get("initial_capital", 100000))
        holdings = data.get("holdings", {})  # {ticker: weight}
        as_of_date_str = data.get("as_of_date", datetime.now().strftime("%Y-%m-%d"))
        as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()

        tickers = list(holdings.keys())
        current_prices = data.get("current_prices", {})

        pid = builder.create_portfolio(
            name=name,
            benchmark=benchmark,
            initial_capital=initial_capital,
            holdings=holdings,
            as_of_date=as_of_date,
            current_prices=current_prices,
        )
        return jsonify({"portfolio_id": pid, "status": "created"})

    except Exception as e:
        logger.error(f"Create portfolio failed: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/portfolios/<portfolio_id>/rebalance", methods=["POST"])
def rebalance_portfolio(portfolio_id):
    try:
        data = request.json
        target_weights = data.get("target_weights", {})
        as_of_date_str = data.get("as_of_date", datetime.now().strftime("%Y-%m-%d"))
        as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
        current_prices = data.get("current_prices", {})

        actions = engine.rebalance_portfolio(
            portfolio_id=portfolio_id,
            target_weights=target_weights,
            as_of_date=as_of_date,
            current_prices=current_prices,
        )

        return jsonify({"actions": actions, "status": "rebalanced"})

    except Exception as e:
        logger.error(f"Rebalance failed: {e}")
        return jsonify({"error": str(e)}), 500
