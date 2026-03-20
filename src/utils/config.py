"""Configuration module for OctaCryptoOracle."""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_COINS = ["bitcoin", "solana", "ethereum"]
SUPPORTED_SYMBOLS = ["BTC/USDT", "SOL/USDT", "ETH/USDT"]


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    COINGECKO_API_KEY: Optional[str] = field(default=None)
    HELIUS_API_KEY: Optional[str] = field(default=None)
    BIRDEYE_API_KEY: Optional[str] = field(default=None)
    BINANCE_API_KEY: Optional[str] = field(default=None)
    BINANCE_SECRET: Optional[str] = field(default=None)
    TELEGRAM_BOT_TOKEN: Optional[str] = field(default=None)
    TELEGRAM_CHAT_ID: Optional[str] = field(default=None)


def get_config() -> Config:
    """Return a Config instance populated from environment variables."""
    return Config(
        COINGECKO_API_KEY=os.getenv("COINGECKO_API_KEY"),
        HELIUS_API_KEY=os.getenv("HELIUS_API_KEY"),
        BIRDEYE_API_KEY=os.getenv("BIRDEYE_API_KEY"),
        BINANCE_API_KEY=os.getenv("BINANCE_API_KEY"),
        BINANCE_SECRET=os.getenv("BINANCE_SECRET"),
        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN"),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID"),
    )
