from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from datetime import date
from sqlalchemy.orm import Session
from backtest.db_schema_pg import DirectIndexingTaxLot


class TaxLotTracker:
    """
    Manages tax lots and implements tax-aware selling (HIFO, LIFO, FIFO).
    """

    def __init__(self, session: Session):
        self.session = session

    def get_lots(self, portfolio_id: str, ticker: str) -> List[DirectIndexingTaxLot]:
        """
        Get all open tax lots for a position, sorted by acquisition date.
        """
        return (
            self.session.query(DirectIndexingTaxLot)
            .filter_by(portfolio_id=portfolio_id, ticker=ticker)
            .order_by(DirectIndexingTaxLot.acquisition_date.asc())
            .all()
        )

    def select_lots_to_sell(
        self,
        portfolio_id: str,
        ticker: str,
        shares_to_sell: float,
        method: str = "HIFO",
    ) -> List[Tuple[DirectIndexingTaxLot, float]]:
        """
        Select lots to sell based on tax optimization method.
        Returns list of (Lot, Shares_from_Lot).
        """
        lots = self.get_lots(portfolio_id, ticker)

        if not lots:
            return []

        lots_with_cost = []
        for lot in lots:
            shares = float(lot.shares)
            if shares <= 0:
                continue
            cost_per_share = float(lot.cost_basis) / shares
            lots_with_cost.append((lot, cost_per_share))

        if method == "FIFO":
            sorted_lots = lots_with_cost
        elif method == "LIFO":
            sorted_lots = sorted(
                lots_with_cost, key=lambda x: x[0].acquisition_date, reverse=True
            )
        elif method == "HIFO":
            # Sell highest cost basis first (minimize gain / maximize loss)
            sorted_lots = sorted(lots_with_cost, key=lambda x: x[1], reverse=True)
        else:
            sorted_lots = lots_with_cost

        selected_lots = []
        remaining_shares = shares_to_sell

        for lot, cost_per_share in sorted_lots:
            lot_shares = float(lot.shares)

            if lot_shares >= remaining_shares:
                selected_lots.append((lot, remaining_shares))
                remaining_shares = 0
                break
            else:
                selected_lots.append((lot, lot_shares))
                remaining_shares -= lot_shares

        return selected_lots
