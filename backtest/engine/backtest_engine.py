from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from backtest.strategies.base import StrategyBase
from backtest.engine.data_loader_pg import PostgresDataLoader
from backtest.engine.rebalance_calendar import RebalanceCalendar
from backtest.engine.execution_model import ExecutionModel
from backtest.metrics.performance import calculate_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main execution engine for event-driven backtesting.
    Supports Strategy Interface v2 (Universe Snapshot + Plugin).
    """

    def __init__(
        self,
        strategy: StrategyBase,
        start_date: date,
        end_date: date,
        initial_capital: float = 100000.0,
        rebalance_freq: str = "quarterly",
    ):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.data_loader = PostgresDataLoader()
        self.strategy.context.data_loader = self.data_loader
        self.rebalance_calendar = RebalanceCalendar(rebalance_freq)
        self.execution_model = ExecutionModel()

        self.current_date = start_date
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    def run(self):
        """
        Execute the backtest simulation.
        """
        print(f"   📅 Period: {self.start_date} ~ {self.end_date}")
        logger.info(
            f"Starting backtest for {self.strategy.name} from {self.start_date} to {self.end_date}"
        )

        # Initialize strategy
        self.strategy.context.cash = self.initial_capital
        self.strategy.initialize(self.strategy.context)

        # Get trading days (simplified: all weekdays for now, or query DB)
        trading_days = self._get_trading_days()
        total_days = len(trading_days)
        print(f"   📆 Trading days: {total_days}")
        
        rebalance_count = 0
        progress_interval = max(1, total_days // 10)  # Show progress every 10%

        for i, current_date in enumerate(trading_days):
            self.current_date = current_date.date()
            self.strategy.context.current_date = self.current_date

            # Show progress
            if i > 0 and i % progress_interval == 0:
                pct = (i / total_days) * 100
                print(f"   ⏳ Progress: {pct:.0f}% ({self.current_date})")

            # 1. Before Trading Start
            self._before_trading_start()

            # 2. Check Rebalance
            if self.rebalance_calendar.is_rebalance_date(self.current_date):
                self._handle_rebalance()
                rebalance_count += 1

            # 3. Process Daily Data
            self._process_daily_updates()

            # 4. Record Equity
            self._record_metrics()

        self.strategy.on_end(self.strategy.context)
        print(f"   🔄 Total rebalances: {rebalance_count}")
        logger.info("Backtest completed")

        return self._get_results()

    def _get_trading_days(self) -> pd.DatetimeIndex:
        """
        Get list of trading days from DB or generate.
        """
        return pd.date_range(start=self.start_date, end=self.end_date, freq="B")

    def _before_trading_start(self):
        """
        Update universe and context before trading.
        """
        # Ensure universe is populated on the first day if empty
        if not self.strategy.context.universe:
            snapshot = self.data_loader.get_universe_as_of(str(self.current_date))
            self.strategy.context.universe = self.strategy.select_universe(snapshot)

        self.strategy.before_trading_start(self.strategy.context, None)

    def _handle_rebalance(self):
        """
        Execute rebalance logic.
        """
        logger.info(f"Rebalancing on {self.current_date}")

        # 1. Get Universe Snapshot
        snapshot = self.data_loader.get_universe_as_of(str(self.current_date))

        # 2. Select Universe
        selected_tickers = self.strategy.select_universe(snapshot)
        self.strategy.context.universe = selected_tickers

        if not selected_tickers:
            logger.warning(f"No tickers selected for {self.current_date}")
            return

        # 3. Get Data (Prices, Fundamentals)
        # Default lookback 365 days for momentum/trends
        # Increased to 400 to ensure we have enough trading days (252+)
        prices = self.data_loader.get_prices_as_of(
            str(self.current_date), ticker_list=selected_tickers, lookback_days=400
        )

        fundamentals = self.data_loader.get_fundamentals_as_of(
            str(self.current_date), ticker_list=selected_tickers
        )

        # 4. Compute Targets
        target_weights = self.strategy.compute_targets(
            self.current_date, snapshot, prices, fundamentals, self.strategy.context
        )

        # 5. Calculate & Execute Orders
        orders = self.execution_model.calculate_orders(
            self.strategy.context.positions,
            target_weights,
            self.strategy.context.cash,
            self._get_current_prices(),
        )

        executed_trades = self.execution_model.execute_orders(
            orders, self._get_current_prices(), self.current_date
        )

        for trade in executed_trades:
            self._update_portfolio(trade)
            self.trades.append(trade)

    def _process_daily_updates(self):
        """
        Placeholder for daily position updates.

        Portfolio valuation (mark-to-market) is handled in _record_metrics().
        Override this method to implement custom daily logic such as:
        - Risk monitoring alerts
        - Position limit checks
        - Custom calculations
        """

    def _get_current_prices(self) -> Dict[str, float]:
        """
        Helper to get today's prices for held/universe assets.
        """
        tickers = list(
            set(
                self.strategy.context.universe
                + list(self.strategy.context.positions.keys())
            )
        )
        if not tickers:
            return {}

        df = self.data_loader.get_prices_as_of(
            str(self.current_date), ticker_list=tickers, lookback_days=1
        )
        if df.empty:
            return {}

        # Get latest close price per ticker
        latest = df.sort_values("date").groupby("ticker").last()
        return latest["close"].to_dict()

    def _update_portfolio(self, trade: Dict[str, Any]):
        """
        Update cash and positions from a trade.
        """
        ticker = trade["ticker"]
        amount = trade["shares"]
        price = trade["price"]
        cost = amount * price
        fee = trade.get("fee", 0.0)

        if trade["side"] == "buy":
            self.strategy.context.cash -= cost + fee
            self.strategy.context.positions[ticker] = (
                self.strategy.context.positions.get(ticker, 0) + amount
            )
        elif trade["side"] == "sell":
            self.strategy.context.cash += cost - fee
            self.strategy.context.positions[ticker] = (
                self.strategy.context.positions.get(ticker, 0) - amount
            )

            if abs(self.strategy.context.positions[ticker]) < 1e-6:
                del self.strategy.context.positions[ticker]

    def _record_metrics(self):
        """
        Record daily equity and stats.
        """
        portfolio_value = self.strategy.context.cash

        prices = self._get_current_prices()
        for ticker, shares in self.strategy.context.positions.items():
            price = prices.get(ticker, 0.0)
            portfolio_value += shares * price

        self.equity_curve.append(
            {
                "date": self.current_date,
                "equity": portfolio_value,
                "cash": self.strategy.context.cash,
            }
        )

    def _get_results(self) -> Dict[str, Any]:
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        metrics = calculate_metrics(equity_df)
        
        # Ensure default metrics if empty (e.g. no data or too short)
        if not metrics:
            metrics = {
                "cagr": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
            }

        return {"equity_curve": equity_df, "trades": trades_df, "metrics": metrics}
