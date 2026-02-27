import logging
import pandas as pd
from datetime import date
from typing import Dict, Any, List, Optional
from backtest.engine.backtest_engine import BacktestEngine
from backtest.strategies.base import StrategyBase, StrategyContext

# Configure logging
logging.basicConfig(level=logging.INFO)


class MockDataLoader:
    """
    Mock data loader for testing.
    """

    def get_universe_as_of(self, as_of_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "name": "Apple Inc.",
                    "sector": "TECH",
                    "market": "US",
                    "weight": 0.5,
                },
                {
                    "ticker": "GOOGL",
                    "name": "Alphabet Inc.",
                    "sector": "TECH",
                    "market": "US",
                    "weight": 0.5,
                },
            ]
        )

    def get_prices_as_of(
        self,
        as_of_date: str,
        ticker_list: Optional[List[str]] = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        end_dt = pd.to_datetime(as_of_date)
        start_dt = end_dt - pd.Timedelta(days=lookback_days)

        dates = pd.date_range(start=start_dt, end=end_dt)
        data = []

        if lookback_days < 10:
            dates = pd.date_range(end=end_dt, periods=lookback_days + 1)

        for d in dates:
            for ticker in ticker_list or ["AAPL", "GOOGL"]:
                if ticker in ["AAPL", "GOOGL"]:
                    price = 150.0 + (d.day % 10)
                    data.append(
                        {
                            "date": d,
                            "ticker": ticker,
                            "open": price,
                            "high": price + 1,
                            "low": price - 1,
                            "close": price,
                            "volume": 1000000,
                        }
                    )
        return pd.DataFrame(data)

    def get_fundamentals_as_of(
        self, as_of_date: str, ticker_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return pd.DataFrame()


class BuyAndHoldStrategy(StrategyBase):
    """
    Simple strategy for testing: Buy AAPL and hold.
    """

    def initialize(self, context: StrategyContext):
        self.ticker = self.params.get("ticker", "AAPL")

    def compute_targets(
        self,
        as_of_date: date,
        snapshot: pd.DataFrame,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        context: StrategyContext,
    ) -> Dict[str, float]:
        return {self.ticker: 1.0}


def test_prototype():
    start_date = date(2023, 1, 1)
    end_date = date(2023, 3, 31)

    strategy = BuyAndHoldStrategy("BuyHold_AAPL", params={"ticker": "AAPL"})

    engine = BacktestEngine(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        rebalance_freq="monthly",
    )

    mock_loader = MockDataLoader()
    engine.data_loader = mock_loader
    engine.strategy.context.data_loader = mock_loader

    print("Starting Backtest Prototype...")
    try:
        results = engine.run()

        equity_curve = results["equity_curve"]
        trades = results["trades"]
        metrics = results["metrics"]

        print("\nBacktest Results:")
        print(f"Metrics: {metrics}")
        print(f"\nTrades ({len(trades)}):")
        print(trades)

        if not equity_curve.empty:
            print(f"\nEquity Curve (Tail 5):")
            print(equity_curve.tail())

        if not trades.empty:
            print("\nBacktest generated trades successfully.")
        else:
            print("\nNo trades generated.")

    except Exception as e:
        print(f"\nBacktest Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_prototype()
