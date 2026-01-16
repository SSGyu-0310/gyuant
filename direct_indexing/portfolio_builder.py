import uuid
import logging
from typing import Dict, Any, List
from datetime import date, datetime
from sqlalchemy.orm import Session

from backtest.db_schema_pg import (
    get_session,
    DirectIndexingPortfolio,
    DirectIndexingPosition,
    DirectIndexingTaxLot,
)

logger = logging.getLogger(__name__)


class PortfolioBuilder:
    """
    Builds and initializes Direct Indexing portfolios.
    """

    def create_portfolio(
        self,
        name: str,
        benchmark: str,
        initial_capital: float,
        holdings: Dict[str, float],
        as_of_date: date,
        current_prices: Dict[str, float],
        rebalance_freq: str = "quarterly",
    ) -> str:
        """
        Create a new portfolio and initialize positions based on target weights.
        Returns portfolio_id.
        """
        session: Session = get_session()
        try:
            portfolio_id = str(uuid.uuid4())

            portfolio = DirectIndexingPortfolio(
                portfolio_id=portfolio_id,
                name=name,
                benchmark=benchmark,
                base_universe=benchmark,
                weighting_method="custom",
                rebalance_freq=rebalance_freq,
                initial_capital=initial_capital,
                created_at=datetime.now(),
            )
            session.add(portfolio)

            positions = []
            lots = []

            for ticker, weight in holdings.items():
                if weight <= 0:
                    continue

                price = current_prices.get(ticker, 100.0)
                market_value = initial_capital * weight
                shares = market_value / price if price > 0 else 0

                if shares == 0:
                    continue

                pos = DirectIndexingPosition(
                    portfolio_id=portfolio_id,
                    as_of_date=as_of_date,
                    ticker=ticker,
                    shares=shares,
                    cost_basis=market_value,
                    market_value=market_value,
                    weight=weight,
                )
                positions.append(pos)

                lot = DirectIndexingTaxLot(
                    lot_id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    ticker=ticker,
                    acquisition_date=as_of_date,
                    shares=shares,
                    cost_basis=market_value,
                )
                lots.append(lot)

            session.bulk_save_objects(positions)
            session.bulk_save_objects(lots)

            session.commit()
            logger.info(
                f"Created Direct Indexing Portfolio {portfolio_id} with {len(positions)} positions"
            )
            return portfolio_id

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create portfolio: {e}")
            raise
        finally:
            session.close()
