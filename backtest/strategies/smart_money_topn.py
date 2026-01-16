from typing import Dict, Any, List
from datetime import date
import pandas as pd
from backtest.strategies.base import StrategyBase, StrategyContext


class SmartMoneyTopN(StrategyBase):
    """
    Strategy that buys the top N stocks from the Smart Money Screener.
    """

    def initialize(self, context: StrategyContext):
        self.top_n = self.params.get("top_n", 10)
        self.weight_scheme = self.params.get(
            "weight_scheme", "equal"
        )  # equal, rank_weighted

    def compute_targets(
        self,
        as_of_date: date,
        snapshot: pd.DataFrame,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        context: StrategyContext,
    ) -> Dict[str, float]:
        """
        Compute target weights using Smart Money rankings.
        """
        # Access data loader from context (injected by engine)
        if not hasattr(context, "data_loader") or context.data_loader is None:
            return {}

        # Get Smart Money Ranks
        current_date_str = str(as_of_date)
        ranks_df = context.data_loader.get_smart_money_ranks(
            current_date_str, limit=self.top_n
        )

        if ranks_df.empty:
            return {}

        # Select tickers
        selected_tickers = ranks_df["ticker"].tolist()

        # Calculate weights
        target_weights = {}
        count = len(selected_tickers)

        if count == 0:
            return {}

        if self.weight_scheme == "equal":
            weight = 1.0 / count
            for ticker in selected_tickers:
                target_weights[ticker] = weight

        return target_weights

    def rebalance(self, context: StrategyContext, data: Any) -> Dict[str, float]:
        """
        Legacy interface - delegates to compute_targets.
        """
        return self.compute_targets(
            as_of_date=context.current_date,
            snapshot=pd.DataFrame(),  # Not used in this strategy
            prices=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            context=context,
        )

