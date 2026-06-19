"""CCXT exchange data fetcher for OctaCryptoOracle."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _CCXT_AVAILABLE = False
    warnings.warn("ccxt not installed. CCXTFetcher will use mock data.")


def _mock_ohlcv(symbol: str, limit: int) -> pd.DataFrame:
    price_map = {"BTC/USDT": (55000, 75000), "SOL/USDT": (55, 85), "ETH/USDT": (1500, 2500)}
    lo, hi = price_map.get(symbol, (100, 1000))
    rng = np.random.default_rng(42)
    closes = rng.uniform(lo, hi, limit)
    opens = closes * rng.uniform(0.97, 1.03, limit)
    highs = np.maximum(opens, closes) * rng.uniform(1.0, 1.05, limit)
    lows = np.minimum(opens, closes) * rng.uniform(0.95, 1.0, limit)
    volumes = rng.uniform(1e8, 1e10, limit)
    timestamps = pd.date_range(end=pd.Timestamp.now("UTC"), periods=limit, freq="D")
    return pd.DataFrame({"timestamp": timestamps, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes})


class CCXTFetcher:
    """Fetches OHLCV and ticker data via CCXT."""

    def __init__(self, exchange_id: str = "binance") -> None:
        self.exchange_id = exchange_id
        self._exchange = None
        if _CCXT_AVAILABLE:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                self._exchange = exchange_class({"enableRateLimit": True})
            except Exception as exc:
                warnings.warn(f"Could not initialize CCXT exchange '{exchange_id}': {exc}")

    def get_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """Return OHLCV DataFrame for *symbol*."""
        if self._exchange is not None:
            try:
                raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df
            except Exception as exc:
                warnings.warn(f"CCXT get_ohlcv failed: {exc}. Falling back to mock data.")
        return _mock_ohlcv(symbol, limit)

    def get_ticker(self, symbol: str) -> dict:
        """Return current ticker info for *symbol*."""
        if self._exchange is not None:
            try:
                return self._exchange.fetch_ticker(symbol)
            except Exception as exc:
                warnings.warn(f"CCXT get_ticker failed: {exc}. Returning mock ticker.")
        price_map = {"BTC/USDT": 64000.0, "SOL/USDT": 70.0, "ETH/USDT": 1700.0}
        price = price_map.get(symbol, 100.0)
        rng = np.random.default_rng(42)
        return {
            "symbol": symbol,
            "last": price * rng.uniform(0.99, 1.01),
            "bid": price * 0.999,
            "ask": price * 1.001,
            "volume": float(rng.uniform(1e8, 1e10)),
        }
