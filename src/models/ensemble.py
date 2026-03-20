"""Ensemble predictor combining LSTM and XGBoost for OctaCryptoOracle."""

import numpy as np
import pandas as pd

from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostSignalModel


class EnsemblePredictor:
    """Weighted ensemble of LSTM price forecast and XGBoost signal classifier."""

    def __init__(self, lstm_weight: float = 0.6, xgb_weight: float = 0.4) -> None:
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight

    def predict(
        self,
        df: pd.DataFrame,
        lstm_model: LSTMModel,
        xgb_model: XGBoostSignalModel,
        scaler,
        sequence_length: int = 60,
    ) -> dict:
        """
        Return ensemble prediction dictionary.

        Keys: price_forecast, signal, signal_confidence, ensemble_confidence, direction.
        """
        # LSTM forecast
        close_vals = df["close"].values.reshape(-1, 1)
        scaled = scaler.transform(close_vals).flatten()
        last_seq = scaled[-sequence_length:]
        price_forecast = lstm_model.predict_next_n(last_seq, n_steps=24, scaler=scaler).tolist()

        # XGBoost signal
        signal, xgb_conf = xgb_model.predict_signal(df)

        # Ensemble confidence
        lstm_conf = min(1.0, abs(price_forecast[-1] - df["close"].iloc[-1]) / (df["close"].iloc[-1] + 1e-9))
        lstm_conf = 1.0 - min(lstm_conf, 0.5)  # closer to current price -> higher confidence
        ensemble_conf = self.lstm_weight * lstm_conf + self.xgb_weight * xgb_conf

        direction = "BULLISH" if signal == 1 else "BEARISH"

        return {
            "price_forecast": price_forecast,
            "signal": signal,
            "signal_confidence": float(xgb_conf),
            "ensemble_confidence": float(ensemble_conf),
            "direction": direction,
        }
