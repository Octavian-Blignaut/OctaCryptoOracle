"""Tests for technical indicator computation."""

import numpy as np
import pandas as pd
import pytest


def _sample_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close * rng.uniform(0.98, 1.02, n)
    high = np.maximum(open_, close) * rng.uniform(1.0, 1.03, n)
    low = np.minimum(open_, close) * rng.uniform(0.97, 1.0, n)
    volume = rng.uniform(1e6, 1e8, n)
    return pd.DataFrame({"timestamp": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_add_all_indicators_returns_dataframe():
    from src.features.technical_indicators import add_all_indicators
    df = _sample_ohlcv()
    result = add_all_indicators(df)
    assert isinstance(result, pd.DataFrame)


def test_add_all_indicators_has_rsi():
    from src.features.technical_indicators import add_all_indicators
    df = _sample_ohlcv()
    result = add_all_indicators(df)
    assert "RSI_14" in result.columns


def test_add_all_indicators_has_macd():
    from src.features.technical_indicators import add_all_indicators
    df = _sample_ohlcv()
    result = add_all_indicators(df)
    assert "MACD" in result.columns


def test_add_all_indicators_reduces_rows():
    from src.features.technical_indicators import add_all_indicators
    df = _sample_ohlcv(300)
    result = add_all_indicators(df)
    assert len(result) < len(df)
