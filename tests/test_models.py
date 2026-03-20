"""Tests for model and backtester modules."""

import numpy as np
import pandas as pd
import pytest


def _make_price_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 1000 + np.cumsum(rng.normal(0, 10, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"timestamp": dates, "close": close, "open": close * 0.99,
                          "high": close * 1.01, "low": close * 0.98, "volume": rng.uniform(1e6, 1e8, n)})


def test_backtester_run():
    from src.models.backtester import Backtester
    df = _make_price_df()
    signals = [1] * 50 + [0] * 50
    bt = Backtester()
    result = bt.run(df, signals)
    assert isinstance(result, dict)
    for key in ["total_return_pct", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades", "equity_curve", "buy_hold_return"]:
        assert key in result, f"Missing key: {key}"


def test_backtester_buy_hold():
    from src.models.backtester import Backtester
    df = _make_price_df()
    signals = [1] * len(df)
    bt = Backtester()
    result = bt.run(df, signals)
    assert isinstance(result["total_return_pct"], float)
    assert isinstance(result["equity_curve"], list)


def test_lstm_model_init():
    from src.models.lstm_model import LSTMModel
    model = LSTMModel(sequence_length=30, n_features=1, units=64, dropout=0.1)
    assert model.sequence_length == 30
    assert model.units == 64
    assert model.model is None


def test_xgboost_signal_model_init():
    from src.models.xgboost_model import XGBoostSignalModel
    model = XGBoostSignalModel(n_estimators=100, max_depth=4, learning_rate=0.1)
    assert model.n_estimators == 100
    assert model.max_depth == 4
    assert model._model is None
