"""Solana on-chain data fetcher for OctaCryptoOracle."""

import warnings

import numpy as np
import requests

from src.utils.config import get_config

_config = get_config()


class SolanaFetcher:
    """Fetches Solana price and on-chain metrics."""

    def get_sol_price(self) -> float:
        """Return current SOL/USD price from CoinGecko."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            resp = requests.get(url, params={"ids": "solana", "vs_currencies": "usd"}, timeout=10)
            resp.raise_for_status()
            return float(resp.json()["solana"]["usd"])
        except Exception as exc:
            warnings.warn(f"SolanaFetcher.get_sol_price failed: {exc}. Using mock price.")
            return float(np.random.default_rng(42).uniform(55, 85))

    def get_on_chain_metrics(self, address: str = None) -> dict:
        """Return on-chain metrics; falls back to mock data on failure."""
        config = get_config()
        if config.HELIUS_API_KEY:
            try:
                url = f"https://api.helius.xyz/v0/network/stats?api-key={config.HELIUS_API_KEY}"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "tps": data.get("tps", 0),
                    "active_validators": data.get("active_validators", 0),
                    "stake_rate": data.get("stake_rate", 0.0),
                    "tvl_defi": data.get("tvl_defi", 0.0),
                }
            except Exception as exc:
                warnings.warn(f"Helius API failed: {exc}. Trying Birdeye.")

        if config.BIRDEYE_API_KEY:
            try:
                headers = {"X-API-KEY": config.BIRDEYE_API_KEY}
                url = "https://public-api.birdeye.so/defi/networks"
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "tps": data.get("tps", 0),
                    "active_validators": data.get("validators", 0),
                    "stake_rate": data.get("stakeRate", 0.0),
                    "tvl_defi": data.get("tvl", 0.0),
                }
            except Exception as exc:
                warnings.warn(f"Birdeye API failed: {exc}. Using mock metrics.")

        rng = np.random.default_rng(42)
        return {
            "tps": float(rng.uniform(3000, 65000)),
            "active_validators": int(rng.integers(1500, 2000)),
            "stake_rate": float(rng.uniform(0.6, 0.8)),
            "tvl_defi": float(rng.uniform(5e8, 2e9)),
        }

    def get_token_holders(self, token_address: str) -> int:
        """Return estimated holder count for *token_address*."""
        config = get_config()
        if config.HELIUS_API_KEY:
            try:
                url = (
                    f"https://api.helius.xyz/v0/token-metadata"
                    f"?api-key={config.HELIUS_API_KEY}"
                )
                resp = requests.get(url, params={"mintAccounts": token_address}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data and isinstance(data, list):
                    return int(data[0].get("numHolders", 0))
            except Exception as exc:
                warnings.warn(f"get_token_holders failed: {exc}. Returning mock value.")
        return int(np.random.default_rng(42).integers(10000, 500000))
