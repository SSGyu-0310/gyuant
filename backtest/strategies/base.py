from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import date


class StrategyContext:
    """
    Context object passed to strategy methods.
    Holds state, portfolio, universe, and configuration.
    """

    def __init__(self):
        self.portfolio: Dict[str, Any] = {}
        self.universe: List[str] = []
        self.params: Dict[str, Any] = {}
        self.current_date: Optional[date] = None
        self.cash: float = 0.0
        self.positions: Dict[str, float] = {}
        self.data_loader: Any = None  # To be injected by engine


class StrategyBase(ABC):
    """
    Abstract base class for all backtesting strategies.

    New Interface (v2):
    - select_universe(snapshot, market_data) -> List[str]
    - compute_targets(as_of_date, snapshot, prices, fundamentals, portfolio_state) -> Dict[str, float]
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.context = StrategyContext()
        self.context.params = self.params

    def initialize(self, context: StrategyContext):
        """
        Called once at the start of the backtest.
        Set up parameters, static universe, etc.
        """
        pass

    def before_trading_start(self, context: StrategyContext, data: Any):
        """
        Called before the market opens each day.
        Update universe, load daily data, etc.
        """
        pass

    def select_universe(self, snapshot: pd.DataFrame) -> List[str]:
        """
        Selects tickers from the available universe snapshot.
        Default: Select all tickers in the snapshot.

        Args:
            snapshot: DataFrame with 'ticker' column.

        Returns:
            List of selected tickers.
        """
        if not snapshot.empty and "ticker" in snapshot.columns:
            return snapshot["ticker"].tolist()
        return []

    @abstractmethod
    def compute_targets(
        self,
        as_of_date: date,
        snapshot: pd.DataFrame,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        context: StrategyContext,
    ) -> Dict[str, float]:
        """
        Compute target weights for the portfolio.

        Args:
            as_of_date: Current rebalancing date.
            snapshot: Universe snapshot DataFrame.
            prices: Historical prices DataFrame (PIT).
            fundamentals: Fundamental data DataFrame (PIT).
            context: Strategy context containing current positions/cash.

        Returns:
            Target weights {ticker: weight}. Sum should be <= 1.0.
        """
        pass

    def on_end(self, context: StrategyContext):
        """
        Called at the end of the backtest.
        """
        pass
