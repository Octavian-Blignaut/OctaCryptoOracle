"""LSTM price-forecasting model for OctaCryptoOracle."""

import warnings
from typing import Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    warnings.warn("TensorFlow not installed. LSTMModel will raise on build/train.")

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class LSTMModel:
    """Sequence-to-one LSTM model for crypto close-price prediction."""

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 1,
        units: int = 128,
        dropout: float = 0.2,
    ) -> None:
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.dropout = dropout
        self.model: Optional[object] = None

    def build_model(self) -> None:
        """Build the Keras LSTM model."""
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required to build the LSTM model.")
        inp = keras.Input(shape=(self.sequence_length, self.n_features))
        x = keras.layers.LSTM(self.units, return_sequences=True)(inp)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.LSTM(64)(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(1)(x)
        self.model = keras.Model(inp, out)
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def prepare_sequences(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding-window sequences from *data*."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            if target is not None:
                y.append(target[i])
        X = np.array(X)
        y_arr = np.array(y) if target is not None else None
        if X.ndim == 2:
            X = X[..., np.newaxis]
        return X, y_arr

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
    ) -> object:
        """Train the model and optionally log params to MLflow."""
        if self.model is None:
            self.build_model()
        if _MLFLOW_AVAILABLE:
            try:
                mlflow.log_params(
                    {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "units": self.units,
                        "dropout": self.dropout,
                        "sequence_length": self.sequence_length,
                    }
                )
            except Exception:
                pass
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
        )
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return model predictions for input sequences *X*."""
        if self.model is None:
            raise RuntimeError("Model not built/trained yet.")
        return self.model.predict(X, verbose=0).flatten()

    def predict_next_n(
        self,
        last_sequence: np.ndarray,
        n_steps: int = 24,
        scaler=None,
    ) -> np.ndarray:
        """Auto-regressive forecast for the next *n_steps* steps."""
        if self.model is None:
            raise RuntimeError("Model not built/trained yet.")
        seq = last_sequence.copy().astype(float)
        if seq.ndim == 1:
            seq = seq[:, np.newaxis]
        predictions = []
        for _ in range(n_steps):
            x_input = seq[-self.sequence_length :][np.newaxis, ...]
            pred = self.model.predict(x_input, verbose=0)[0, 0]
            predictions.append(pred)
            seq = np.vstack([seq, [[pred]]])
        preds = np.array(predictions)
        if scaler is not None:
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        return preds

    def save(self, path: str) -> None:
        """Save the Keras model to *path*."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a Keras model from *path*."""
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required to load the LSTM model.")
        self.model = keras.models.load_model(path)
