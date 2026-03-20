"""XGBoost signal classification model for OctaCryptoOracle."""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn("xgboost not installed. XGBoostSignalModel will raise on train.")

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

_FEATURE_COLS = [
    "SMA_20", "SMA_50", "EMA_12", "EMA_26", "MACD", "MACD_signal",
    "RSI_14", "Stoch_K", "Stoch_D", "Williams_R", "ROC_10", "MOM_10",
    "BB_upper", "BB_lower", "BB_pct", "ATR_14", "OBV", "MFI_14",
    "volume_sma_20", "price_momentum_5", "price_momentum_10",
    "volume_momentum", "high_low_spread", "body_to_range",
]


class XGBoostSignalModel:
    """Binary signal classifier: 1 = price up next day, 0 = price down."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: object = None
        self._feature_cols: list = _FEATURE_COLS

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and binary target from *df*."""
        available = [c for c in self._feature_cols if c in df.columns]
        X = df[available].values
        target = (df["close"].shift(-1) > df["close"]).astype(int).values
        # Drop last row (no next-day price)
        X = X[:-1]
        target = target[:-1]
        return X, target

    def train(self, df: pd.DataFrame) -> None:
        """Train the XGBoost classifier on *df*."""
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost is required to train XGBoostSignalModel.")
        X, y = self.prepare_features(df)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="logloss",
            random_state=42,
        )
        self._model.fit(X, y)
        if _MLFLOW_AVAILABLE:
            try:
                mlflow.log_params({
                    "xgb_n_estimators": self.n_estimators,
                    "xgb_max_depth": self.max_depth,
                    "xgb_lr": self.learning_rate,
                })
            except Exception:
                pass

    def predict_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Return (signal, confidence) for the latest row in *df*."""
        if self._model is None:
            warnings.warn("XGBoostSignalModel not trained. Returning neutral signal.")
            return 0, 0.5
        available = [c for c in self._feature_cols if c in df.columns]
        X_last = df[available].iloc[[-1]].values
        signal = int(self._model.predict(X_last)[0])
        proba = self._model.predict_proba(X_last)[0]
        confidence = float(max(proba))
        return signal, confidence

    def get_feature_importance(self) -> dict:
        """Return feature importance mapping."""
        if self._model is None:
            return {}
        scores = self._model.feature_importances_
        available = self._feature_cols[: len(scores)]
        return dict(zip(available, scores.tolist()))
