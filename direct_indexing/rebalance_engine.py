import json
import logging
import uuid
from typing import Dict, List, Any
from datetime import date, datetime
from sqlalchemy.orm import Session

from backtest.db_schema_pg import (
    get_session,
    DirectIndexingPortfolio,
    DirectIndexingPosition,
    DirectIndexingTaxLot,
    DirectIndexingRebalanceLog,
    DirectIndexingTlhEvent,
)
from direct_indexing.tax_lot_tracker import TaxLotTracker

logger = logging.getLogger(__name__)


class RebalanceEngine:
    """
    Executes rebalance for Direct Indexing portfolios with tax awareness.
    """

    def rebalance_portfolio(
        self,
        portfolio_id: str,
        target_weights: Dict[str, float],
        as_of_date: date,
        current_prices: Dict[str, float],
        tax_method: str = "HIFO",
    ):
        """
        Rebalance portfolio to target weights.
        """
        session: Session = get_session()
        tax_tracker = TaxLotTracker(session)

        try:
            # 1. Load Portfolio
            portfolio = session.get(DirectIndexingPortfolio, portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")

            # 2. Load Current Positions
            positions = (
                session.query(DirectIndexingPosition)
                .filter_by(portfolio_id=portfolio_id, as_of_date=as_of_date)
                .all()
            )

            current_holdings = {}
            for pos in positions:
                current_holdings[pos.ticker] = float(pos.shares)

            # 3. Calculate Trades
            total_equity = 0.0
            for ticker, shares in current_holdings.items():
                price = current_prices.get(ticker, 0.0)
                total_equity += shares * price

            trades = []
            target_shares_map = {}

            for ticker, weight in target_weights.items():
                price = current_prices.get(ticker, 0.0)
                if price > 0:
                    target_val = total_equity * weight
                    target_shares = target_val / price
                    target_shares_map[ticker] = target_shares

            all_tickers = set(current_holdings.keys()) | set(target_shares_map.keys())

            for ticker in all_tickers:
                curr = current_holdings.get(ticker, 0.0)
                tgt = target_shares_map.get(ticker, 0.0)
                price = current_prices.get(ticker, 0.0)

                diff = tgt - curr

                if abs(diff) < 0.0001:
                    continue

                side = "buy" if diff > 0 else "sell"
                trades.append(
                    {
                        "ticker": ticker,
                        "side": side,
                        "shares": abs(diff),
                        "price": price,
                    }
                )

            # 4. Execute Trades & Update Lots
            rebalance_actions = []

            for trade in trades:
                ticker = trade["ticker"]
                shares = trade["shares"]
                price = trade["price"]
                side = trade["side"]

                if side == "sell":
                    selected_lots = tax_tracker.select_lots_to_sell(
                        portfolio_id, ticker, shares, method=tax_method
                    )

                    realized_pl = 0.0

                    for lot, lot_shares in selected_lots:
                        cost_per_share = float(lot.cost_basis) / float(lot.shares)
                        proceeds = lot_shares * price
                        cost = lot_shares * cost_per_share
                        pl = proceeds - cost
                        realized_pl += pl

                        lot.shares = float(lot.shares) - lot_shares
                        lot.cost_basis = float(lot.cost_basis) - cost

                        if float(lot.shares) < 0.0001:
                            session.delete(lot)

                        if pl < 0:
                            tlh_event = DirectIndexingTlhEvent(
                                event_id=str(uuid.uuid4()),
                                portfolio_id=portfolio_id,
                                ticker=ticker,
                                loss_amount=abs(pl),
                                sale_date=as_of_date,
                            )
                            session.add(tlh_event)

                    rebalance_actions.append(
                        {
                            "ticker": ticker,
                            "action": "sell",
                            "shares": shares,
                            "price": price,
                            "realized_pl": realized_pl,
                        }
                    )

                elif side == "buy":
                    cost = shares * price
                    new_lot = DirectIndexingTaxLot(
                        lot_id=str(uuid.uuid4()),
                        portfolio_id=portfolio_id,
                        ticker=ticker,
                        acquisition_date=as_of_date,
                        shares=shares,
                        cost_basis=cost,
                    )
                    session.add(new_lot)

                    rebalance_actions.append(
                        {
                            "ticker": ticker,
                            "action": "buy",
                            "shares": shares,
                            "price": price,
                        }
                    )

            # 5. Log Rebalance
            log = DirectIndexingRebalanceLog(
                rebalance_id=str(uuid.uuid4()),
                portfolio_id=portfolio_id,
                as_of_date=as_of_date,
                action_json=json.dumps(rebalance_actions),
            )
            session.add(log)

            # 6. Update Positions Snapshot
            session.query(DirectIndexingPosition).filter_by(
                portfolio_id=portfolio_id, as_of_date=as_of_date
            ).delete()

            remaining_lots = (
                session.query(DirectIndexingTaxLot)
                .filter_by(portfolio_id=portfolio_id)
                .all()
            )

            ticker_agg = {}
            for lot in remaining_lots:
                if lot.ticker not in ticker_agg:
                    ticker_agg[lot.ticker] = {"shares": 0.0, "cost": 0.0}
                ticker_agg[lot.ticker]["shares"] += float(lot.shares)
                ticker_agg[lot.ticker]["cost"] += float(lot.cost_basis)

            new_positions = []
            total_port_val = 0.0

            for ticker, agg in ticker_agg.items():
                shares = agg["shares"]
                cost = agg["cost"]
                price = current_prices.get(ticker, 0.0)
                mkt_val = shares * price
                total_port_val += mkt_val

                pos = DirectIndexingPosition(
                    portfolio_id=portfolio_id,
                    as_of_date=as_of_date,
                    ticker=ticker,
                    shares=shares,
                    cost_basis=cost,
                    market_value=mkt_val,
                    weight=0,
                )
                new_positions.append(pos)

            if total_port_val > 0:
                for pos in new_positions:
                    pos.weight = float(pos.market_value) / total_port_val

            session.bulk_save_objects(new_positions)

            session.commit()
            logger.info(
                f"Rebalanced portfolio {portfolio_id}. Actions: {len(rebalance_actions)}"
            )
            return rebalance_actions

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to rebalance portfolio: {e}")
            raise
        finally:
            session.close()
