"""Technical indicator computation for OctaCryptoOracle."""

import warnings

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    warnings.warn("pandas-ta not installed. Some indicators may be unavailable.")


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 30+ technical indicators to *df* and return the enriched DataFrame.

    Expects columns: open, high, low, close, volume.
    NaN rows caused by indicator warm-up are dropped.
    """
    df = df.copy()

    if not _TA_AVAILABLE:
        warnings.warn("pandas-ta unavailable; returning df as-is.")
        return df

    # --- Trend ---
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_200"] = ta.sma(df["close"], length=200)
    df["EMA_12"] = ta.ema(df["close"], length=12)
    df["EMA_26"] = ta.ema(df["close"], length=26)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"] = macd.iloc[:, 0]
        df["MACD_hist"] = macd.iloc[:, 1]
        df["MACD_signal"] = macd.iloc[:, 2]
    else:
        df["MACD"] = np.nan
        df["MACD_hist"] = np.nan
        df["MACD_signal"] = np.nan

    # --- Momentum ---
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None and not stoch.empty:
        df["Stoch_K"] = stoch.iloc[:, 0]
        df["Stoch_D"] = stoch.iloc[:, 1]
    else:
        df["Stoch_K"] = np.nan
        df["Stoch_D"] = np.nan
    df["Williams_R"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    df["ROC_10"] = ta.roc(df["close"], length=10)
    df["MOM_10"] = ta.mom(df["close"], length=10)

    # --- Volatility ---
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        df["BB_lower"] = bb.iloc[:, 0]
        df["BB_middle"] = bb.iloc[:, 1]
        df["BB_upper"] = bb.iloc[:, 2]
        df["BB_pct"] = bb.iloc[:, 4] if bb.shape[1] > 4 else np.nan
    else:
        df["BB_lower"] = df["BB_middle"] = df["BB_upper"] = df["BB_pct"] = np.nan
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["NATR"] = ta.natr(df["high"], df["low"], df["close"], length=14)

    # --- Volume ---
    df["OBV"] = ta.obv(df["close"], df["volume"])
    df["MFI_14"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    # VWAP requires a DatetimeIndex; set timestamp as index temporarily
    try:
        df_ts = df.set_index("timestamp")
        vwap_series = ta.vwap(df_ts["high"], df_ts["low"], df_ts["close"], df_ts["volume"])
        df["VWAP"] = vwap_series.values
    except Exception:
        df["VWAP"] = np.nan
    df["volume_sma_20"] = ta.sma(df["volume"], length=20)

    # --- Custom ---
    df["price_momentum_5"] = df["close"].pct_change(5)
    df["price_momentum_10"] = df["close"].pct_change(10)
    df["volume_momentum"] = df["volume"].pct_change(5)
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    df["body_to_range"] = body / candle_range.replace(0, np.nan)

    df = df.dropna()
    return df
