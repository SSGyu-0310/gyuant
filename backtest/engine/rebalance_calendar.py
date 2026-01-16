from datetime import date
from typing import List, Optional
import pandas as pd


class RebalanceCalendar:
    """
    Manages rebalance schedules (e.g., quarterly, monthly).
    """

    def __init__(self, freq: str = "quarterly"):
        self.freq = freq
        self._last_rebalance_month: Optional[int] = None
        self._first_trading_day = True

    def is_rebalance_date(self, current_date: date) -> bool:
        """
        Check if today is a rebalance date.
        Rebalances on the first trading day of the period.
        """
        # Always rebalance on the first trading day
        if self._first_trading_day:
            self._first_trading_day = False
            self._last_rebalance_month = current_date.month
            return True

        if self.freq == "monthly":
            # Rebalance on the first trading day of each month
            if current_date.month != self._last_rebalance_month:
                self._last_rebalance_month = current_date.month
                return True

        elif self.freq == "quarterly":
            # Rebalance on the first trading day of Jan, Apr, Jul, Oct
            quarter_months = [1, 4, 7, 10]
            if current_date.month in quarter_months:
                if self._last_rebalance_month != current_date.month:
                    self._last_rebalance_month = current_date.month
                    return True

        return False

    def reset(self):
        """Reset the calendar state for a new backtest run."""
        self._last_rebalance_month = None
        self._first_trading_day = True
