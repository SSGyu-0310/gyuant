import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate CAGR, Volatility, Sharpe Ratio, Max Drawdown.
    Expects equity_curve to have 'date' and 'equity' columns.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {}

    # Ensure date column is datetime type for proper calculations
    equity_curve = equity_curve.copy()
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])

    # Calculate daily returns
    equity_curve["returns"] = equity_curve["equity"].pct_change().fillna(0)

    # 1. CAGR
    start_value = equity_curve["equity"].iloc[0]
    end_value = equity_curve["equity"].iloc[-1]
    days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
    years = days / 365.25

    cagr = 0.0
    if years > 0:
        cagr = (end_value / start_value) ** (1 / years) - 1

    # 2. Volatility (Annualized)
    daily_vol = equity_curve["returns"].std()
    annual_vol = daily_vol * np.sqrt(252)

    # 3. Sharpe Ratio (Annualized, assuming risk-free = 0 for simplicity)
    mean_return = equity_curve["returns"].mean()
    sharpe = 0.0
    if daily_vol > 0:
        sharpe = (mean_return / daily_vol) * np.sqrt(252)

    # 4. Max Drawdown
    cum_max = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - cum_max) / cum_max
    max_dd = drawdown.min()

    return {
        "cagr": float(cagr),
        "volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "total_return": float((end_value / start_value) - 1),
    }
