"""CoinGecko data fetcher for OctaCryptoOracle."""

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.utils.config import get_config

BASE_URL = "https://api.coingecko.com/api/v3"
RATE_LIMIT_SLEEP = 1.2

# Mock price ranges per coin
MOCK_PRICE_RANGES = {
    "bitcoin": (40000, 70000),
    "solana": (80, 200),
    "ethereum": (2000, 4000),
}


def _mock_ohlcv(coin_id: str, days: int) -> pd.DataFrame:
    """Generate realistic-looking mock OHLCV data."""
    rng = np.random.default_rng(42)
    lo, hi = MOCK_PRICE_RANGES.get(coin_id, (100, 200))
    n = days
    prices = rng.uniform(lo, hi, n)
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="D")
    opens = prices
    closes = prices * rng.uniform(0.97, 1.03, n)
    highs = np.maximum(opens, closes) * rng.uniform(1.0, 1.05, n)
    lows = np.minimum(opens, closes) * rng.uniform(0.95, 1.0, n)
    volumes = rng.uniform(1e8, 1e10, n)
    return pd.DataFrame(
        {"timestamp": timestamps, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


class CoinGeckoFetcher:
    """Fetches market data from CoinGecko API."""

    def __init__(self) -> None:
        self._config = get_config()
        self._headers: dict = {}
        if self._config.COINGECKO_API_KEY:
            self._headers["x-cg-demo-api-key"] = self._config.COINGECKO_API_KEY

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{BASE_URL}{endpoint}"
        time.sleep(RATE_LIMIT_SLEEP)
        resp = requests.get(url, headers=self._headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_price(self, coin_id: str, vs_currency: str = "usd") -> dict:
        """Get current price for a coin."""
        try:
            data = self._get("/simple/price", {"ids": coin_id, "vs_currencies": vs_currency})
            return data.get(coin_id, {})
        except Exception as exc:
            warnings.warn(f"CoinGecko get_price failed: {exc}. Using mock data.")
            lo, hi = MOCK_PRICE_RANGES.get(coin_id, (100, 200))
            return {vs_currency: float(np.random.default_rng(42).uniform(lo, hi))}

    def get_historical_ohlc(self, coin_id: str, days: int = 365, vs_currency: str = "usd") -> pd.DataFrame:
        """Get historical OHLC data as DataFrame."""
        try:
            data = self._get(f"/coins/{coin_id}/ohlc", {"vs_currency": vs_currency, "days": days})
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            # CoinGecko OHLC endpoint doesn't return volume; fetch separately
            market = self._get(
                f"/coins/{coin_id}/market_chart",
                {"vs_currency": vs_currency, "days": days, "interval": "daily"},
            )
            volumes = market.get("total_volumes", [])
            if volumes:
                vol_df = pd.DataFrame(volumes, columns=["ts", "volume"])
                vol_df["ts"] = pd.to_datetime(vol_df["ts"], unit="ms", utc=True).dt.normalize()
                df["date"] = df["timestamp"].dt.normalize()
                df = df.merge(vol_df, left_on="date", right_on="ts", how="left").drop(columns=["date", "ts"])
            else:
                df["volume"] = np.nan
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as exc:
            warnings.warn(f"CoinGecko get_historical_ohlc failed: {exc}. Using mock data.")
            return _mock_ohlcv(coin_id, days)

    def get_market_data(self, coin_id: str) -> dict:
        """Get market cap, volume, and price change data."""
        try:
            data = self._get(f"/coins/{coin_id}", {"localization": "false", "tickers": "false", "community_data": "false"})
            mkt = data.get("market_data", {})
            return {
                "market_cap": mkt.get("market_cap", {}).get("usd"),
                "total_volume": mkt.get("total_volume", {}).get("usd"),
                "price_change_24h": mkt.get("price_change_percentage_24h"),
                "price_change_7d": mkt.get("price_change_percentage_7d"),
                "price_change_30d": mkt.get("price_change_percentage_30d"),
            }
        except Exception as exc:
            warnings.warn(f"CoinGecko get_market_data failed: {exc}. Using mock data.")
            rng = np.random.default_rng(42)
            return {
                "market_cap": float(rng.uniform(1e10, 1e12)),
                "total_volume": float(rng.uniform(1e9, 1e11)),
                "price_change_24h": float(rng.uniform(-5, 5)),
                "price_change_7d": float(rng.uniform(-10, 10)),
                "price_change_30d": float(rng.uniform(-20, 20)),
            }
