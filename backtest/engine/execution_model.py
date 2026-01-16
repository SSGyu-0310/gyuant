import uuid
from typing import Dict, List, Any
from datetime import date


class ExecutionModel:
    """
    Handles order generation and execution simulation.
    """

    def __init__(self, transaction_cost_bps: float = 10.0):
        self.transaction_cost_bps = transaction_cost_bps

    def calculate_orders(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        current_cash: float,
        current_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Generate orders to move from current positions to target weights.
        """
        # Calculate total portfolio value
        portfolio_value = current_cash
        for ticker, shares in current_positions.items():
            price = current_prices.get(ticker, 0.0)
            portfolio_value += shares * price

        orders = []

        # Calculate target value for each asset
        for ticker, weight in target_weights.items():
            target_val = portfolio_value * weight
            price = current_prices.get(ticker, 0.0)

            if price <= 0:
                continue

            current_shares = current_positions.get(ticker, 0.0)
            target_shares = target_val / price

            diff_shares = target_shares - current_shares

            if abs(diff_shares) > 0:  # minimal threshold
                orders.append(
                    {
                        "ticker": ticker,
                        "side": "buy" if diff_shares > 0 else "sell",
                        "shares": abs(diff_shares),
                        "price": price,
                    }
                )

        # Handle sell-offs for assets not in target
        for ticker in current_positions:
            if ticker not in target_weights:
                shares = current_positions[ticker]
                price = current_prices.get(ticker, 0.0)
                if shares > 0 and price > 0:
                    orders.append(
                        {
                            "ticker": ticker,
                            "side": "sell",
                            "shares": shares,
                            "price": price,
                        }
                    )

        return orders

    def execute_orders(
        self,
        orders: List[Dict[str, Any]],
        current_prices: Dict[str, float],
        execution_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Simulate execution of orders.
        """
        executed_trades = []

        for order in orders:
            # Simple execution at current price
            # Real model would consider slippage, volume limits, etc.
            price = order["price"]
            value = price * order["shares"]
            fee = value * (self.transaction_cost_bps / 10000.0)

            executed_trades.append(
                {
                    "trade_id": str(uuid.uuid4()),
                    "ticker": order["ticker"],
                    "side": order["side"],
                    "shares": order["shares"],
                    "price": price,
                    "fee": fee,
                    "trade_date": execution_date,
                }
            )

        return executed_trades
