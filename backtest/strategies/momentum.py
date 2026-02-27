from typing import Dict, List, Any, Optional
from datetime import date
import pandas as pd
import numpy as np
from backtest.strategies.base import StrategyBase, StrategyContext


class MomentumStrategy(StrategyBase):
    """
    Momentum Strategy:
    - Signal: 12-1 Month Momentum (Return of last 12 months excluding most recent month)
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
        if prices.empty:
            return {}

        # Calculate momentum for each ticker
        # prices dataframe: ticker, date, close, ...
        # We need prices ~12 months ago and ~1 month ago

        # Pivot to wide format: index=date, columns=ticker, values=close
        closes = prices.pivot(index="date", columns="ticker", values="close")

        # Check if we have enough history
        # 12 months ~ 252 trading days
        # 1 month ~ 21 trading days
        if len(closes) < 252:
            return {}

        # Calculate 12-1 momentum
        # Return from t-252 to t-21
        # Mom = P(t-21) / P(t-252) - 1

        try:
            # Get price at t-21 (approx 1 month ago)
            p_recent = closes.iloc[-21]
            # Get price at t-252 (approx 1 year ago)
            p_old = closes.iloc[-252]

            momentum = (p_recent / p_old) - 1.0

            # Filter for tickers in snapshot
            valid_tickers = snapshot["ticker"].unique()
            momentum = momentum[momentum.index.isin(valid_tickers)]

            # Remove NaN
            momentum = momentum.dropna()

            # Select top N
            top_picks = momentum.nlargest(self.top_n)

            if top_picks.empty:
                return {}

            weight = 1.0 / len(top_picks)
            return {ticker: weight for ticker in top_picks.index}

        except IndexError:
            # Not enough data
            return {}
