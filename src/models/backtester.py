"""Backtesting engine for OctaCryptoOracle."""

from typing import List

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


_TRADING_DAYS_PER_YEAR = 252  # approximate number of trading days in a year


class Backtester:
    """Simulates a simple long-only trading strategy based on model signals."""

    def __init__(self, initial_capital: float = 10_000.0, fee_pct: float = 0.001) -> None:
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct

    def run(self, df: pd.DataFrame, signals: List[int]) -> dict:
        """
        Simulate trading based on *signals* (1=buy/hold, 0=sell/stay out).

        Returns performance metrics and the equity curve.
        """
        prices = df["close"].values
        n = min(len(prices), len(signals))
        prices = prices[:n]
        signals_arr = np.array(signals[:n])

        capital = self.initial_capital
        position = 0.0  # units held
        entry_value = 0.0  # portfolio value at last buy
        equity_curve: List[float] = [capital]
        completed_trades = 0
        wins = 0

        for i in range(1, n):
            price = prices[i]
            prev_signal = signals_arr[i - 1]

            if prev_signal == 1 and position == 0.0:
                # Buy: convert all cash to position
                fee = capital * self.fee_pct
                spend = capital - fee
                position = spend / price
                entry_value = capital
                capital = 0.0
            elif prev_signal == 0 and position > 0.0:
                # Sell: liquidate position
                proceeds = position * price
                fee = proceeds * self.fee_pct
                capital = proceeds - fee
                if capital > entry_value:
                    wins += 1
                completed_trades += 1
                position = 0.0

            portfolio_value = capital + position * price
            equity_curve.append(portfolio_value)

        # Close open position at last price
        if position > 0.0:
            fee = position * prices[-1] * self.fee_pct
            final = position * prices[-1] - fee
            if final > entry_value:
                wins += 1
            completed_trades += 1
            capital = final
            position = 0.0
            equity_curve[-1] = capital

        equity_arr = np.array(equity_curve)
        total_return = (equity_arr[-1] - self.initial_capital) / self.initial_capital * 100.0
        n_years = max(n / _TRADING_DAYS_PER_YEAR, 1 / _TRADING_DAYS_PER_YEAR)
        annualized = ((equity_arr[-1] / self.initial_capital) ** (1 / n_years) - 1) * 100.0

        # Sharpe ratio (daily returns)
        daily_returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-9)
        sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(_TRADING_DAYS_PER_YEAR))

        # Max drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / (peak + 1e-9)
        max_drawdown = float(drawdown.min() * 100.0)

        win_rate = (wins / max(completed_trades, 1)) * 100.0

        # Buy-and-hold
        buy_hold = (prices[-1] - prices[0]) / (prices[0] + 1e-9) * 100.0

        return {
            "total_return_pct": float(total_return),
            "annualized_return": float(annualized),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": float(win_rate),
            "total_trades": completed_trades,
            "equity_curve": equity_arr.tolist(),
            "buy_hold_return": float(buy_hold),
        }

    def plot_equity_curve(self, results: dict) -> object:
        """Return a Plotly figure of the equity curve vs buy-and-hold."""
        if not _PLOTLY_AVAILABLE:
            raise RuntimeError("plotly is required for plotting.")
        equity = results["equity_curve"]
        n = len(equity)
        x = list(range(n))
        bh_final = self.initial_capital * (1 + results["buy_hold_return"] / 100.0)
        buy_hold_curve = np.linspace(self.initial_capital, bh_final, n)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=equity, name="Strategy", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=x, y=buy_hold_curve, name="Buy & Hold", line=dict(color="orange", dash="dash")))
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Days",
            yaxis_title="Portfolio Value (USD)",
            template="plotly_dark",
        )
        return fig
