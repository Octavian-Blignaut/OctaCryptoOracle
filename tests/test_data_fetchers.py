"""Tests for data fetcher modules."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def test_coingecko_fetcher_get_price():
    with patch("src.data.coingecko_fetcher.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"bitcoin": {"usd": 45000.0}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        from src.data.coingecko_fetcher import CoinGeckoFetcher
        fetcher = CoinGeckoFetcher()
        with patch("src.data.coingecko_fetcher.time.sleep"):
            result = fetcher.get_price("bitcoin")
    assert isinstance(result, dict)
    assert "usd" in result


def test_coingecko_fetcher_historical():
    from src.data.coingecko_fetcher import CoinGeckoFetcher
    fetcher = CoinGeckoFetcher()
    with patch.object(fetcher, "_get", side_effect=Exception("API unavailable")):
        df = fetcher.get_historical_ohlc("bitcoin", days=30)
    assert isinstance(df, pd.DataFrame)
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"


def test_ccxt_fetcher_init():
    from src.data.ccxt_fetcher import CCXTFetcher
    fetcher = CCXTFetcher(exchange_id="binance")
    assert fetcher.exchange_id == "binance"


def test_solana_fetcher_on_chain_metrics():
    from src.data.solana_fetcher import SolanaFetcher
    fetcher = SolanaFetcher()
    metrics = fetcher.get_on_chain_metrics()
    assert isinstance(metrics, dict)
    for key in ["tps", "active_validators", "stake_rate", "tvl_defi"]:
        assert key in metrics, f"Missing key: {key}"
