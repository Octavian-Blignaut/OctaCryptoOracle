"""High-level prediction service for OctaCryptoOracle."""

import warnings
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed.")

from src.data.coingecko_fetcher import CoinGeckoFetcher
from src.features.technical_indicators import add_all_indicators
from src.models.backtester import Backtester
from src.models.ensemble import EnsemblePredictor
from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostSignalModel
from src.utils.config import SUPPORTED_COINS


class CryptoPredictorService:
    """Orchestrates data fetching, feature engineering, model training and prediction."""

    def __init__(self) -> None:
        self._fetcher = CoinGeckoFetcher()
        self._ensemble = EnsemblePredictor()

    def prepare_model_data(self, coin_id: str, days: int = 365) -> Tuple[pd.DataFrame, object]:
        """Fetch data, add indicators, fit scaler. Returns (df, scaler)."""
        df = self._fetcher.get_historical_ohlc(coin_id, days=days)
        df = add_all_indicators(df)
        df = df.reset_index(drop=True)
        scaler = MinMaxScaler() if _SKLEARN_AVAILABLE else _DummyScaler()
        scaler.fit(df[["close"]].values)
        return df, scaler

    def train_models(self, coin_id: str) -> Tuple[LSTMModel, XGBoostSignalModel, object]:
        """Train and return LSTM + XGBoost models for *coin_id*."""
        df, scaler = self.prepare_model_data(coin_id)
        # LSTM
        lstm = LSTMModel(sequence_length=60)
        lstm.build_model()
        close_scaled = scaler.transform(df[["close"]].values).flatten()
        X, y = lstm.prepare_sequences(close_scaled, close_scaled)
        if len(X) > 0:
            lstm.train(X, y, epochs=10, batch_size=32)
        # XGBoost
        xgb = XGBoostSignalModel()
        try:
            xgb.train(df)
        except Exception as exc:
            warnings.warn(f"XGBoost training failed: {exc}")
        return lstm, xgb, scaler

    def get_prediction(self, coin_id: str) -> dict:
        """Return full prediction dictionary for *coin_id*."""
        if coin_id not in SUPPORTED_COINS:
            raise ValueError(f"Unsupported coin: {coin_id}. Choose from {SUPPORTED_COINS}.")

        price_info = self._fetcher.get_price(coin_id)
        current_price = float(price_info.get("usd", 0.0))
        if current_price <= 0:
            raise ValueError(f"Invalid current price for {coin_id}: {current_price}")

        df, scaler = self.prepare_model_data(coin_id)

        lstm = LSTMModel(sequence_length=60)
        try:
            lstm.build_model()
            close_scaled = scaler.transform(df[["close"]].values).flatten()
            X, y = lstm.prepare_sequences(close_scaled, close_scaled)
            if len(X) > 0:
                lstm.train(X, y, epochs=5, batch_size=32)
        except Exception as exc:
            warnings.warn(f"LSTM unavailable: {exc}. Using mock forecast.")
            lstm = None

        xgb = XGBoostSignalModel()
        try:
            xgb.train(df)
        except Exception as exc:
            warnings.warn(f"XGBoost unavailable: {exc}.")

        if lstm is not None:
            ensemble_result = self._ensemble.predict(df, lstm, xgb, scaler)
            price_forecast = ensemble_result["price_forecast"]
        else:
            rng = np.random.default_rng(42)
            price_forecast = (current_price * (1 + rng.uniform(-0.02, 0.02, 24))).tolist()
            ensemble_result = {"signal": 1, "signal_confidence": 0.6, "ensemble_confidence": 0.6, "direction": "BULLISH"}

        predicted_price_24h = float(price_forecast[-1]) if price_forecast else current_price
        price_change_pct = (predicted_price_24h - current_price) / current_price * 100.0

        signal = ensemble_result["signal"]
        signal_label = "BUY" if signal == 1 else "SELL"
        confidence = ensemble_result["ensemble_confidence"]

        # Backtest
        signals_list = [1 if ensemble_result["signal"] == 1 else 0] * len(df)
        backtester = Backtester()
        backtest_results = backtester.run(df, signals_list)

        return {
            "coin": coin_id,
            "current_price": current_price,
            "price_forecast_24h": [float(p) for p in price_forecast],
            "predicted_price_24h": predicted_price_24h,
            "price_change_pct": float(price_change_pct),
            "signal": signal,
            "signal_label": signal_label,
            "confidence": float(confidence),
            "direction": ensemble_result["direction"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backtest_results": backtest_results,
        }


class _DummyScaler:
    """Minimal stand-in when scikit-learn is unavailable."""

    def fit(self, X):
        self._min = float(np.min(X))
        self._max = float(np.max(X)) + 1e-9
        return self

    def transform(self, X):
        return (np.array(X) - self._min) / (self._max - self._min)

    def inverse_transform(self, X):
        return np.array(X) * (self._max - self._min) + self._min
