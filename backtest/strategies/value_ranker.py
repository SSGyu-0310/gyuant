from typing import Dict, List, Any, Optional
from datetime import date
import pandas as pd
from backtest.strategies.base import StrategyBase, StrategyContext


class ValueRankerStrategy(StrategyBase):
    """
    Value Ranker Strategy:
    - Universe: Provided snapshot
    - Signal: EV/EBITDA (Lower is better)
    - Allocation: Top N equal weight
    """

    def initialize(self, context: StrategyContext):
        self.top_n = self.params.get("top_n", 20)

    def compute_targets(
        self,
        as_of_date: date,
        snapshot: pd.DataFrame,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        context: StrategyContext,
    ) -> Dict[str, float]:
        if fundamentals.empty:
            return {}

        # Filter for tickers in snapshot and valid EV/EBITDA
        valid_data = fundamentals[
            (fundamentals["ticker"].isin(snapshot["ticker"]))
            & (fundamentals["ev_ebitda"] > 0)
        ].copy()

        if valid_data.empty:
            return {}

        # Rank by EV/EBITDA ascending
        valid_data["rank"] = valid_data["ev_ebitda"].rank(ascending=True)

        # Select top N
        top_picks = valid_data.nsmallest(self.top_n, "rank")

        if top_picks.empty:
            return {}

        # Equal weight
        weight = 1.0 / len(top_picks)
        return {ticker: weight for ticker in top_picks["ticker"]}
