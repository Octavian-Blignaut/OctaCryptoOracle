"""Streamlit dashboard for OctaCryptoOracle."""

import warnings

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from src.data.coingecko_fetcher import CoinGeckoFetcher
from src.features.technical_indicators import add_all_indicators
from src.models.backtester import Backtester
from src.prediction.predictor import CryptoPredictorService

COIN_MAP = {"BTC": "bitcoin", "SOL": "solana", "ETH": "ethereum"}
TIMEFRAME_DAYS = {"7 days": 7, "30 days": 30, "90 days": 90, "365 days": 365}

if _ST_AVAILABLE:
    st.set_page_config(page_title="🔮 OctaCryptoOracle", layout="wide", page_icon="🔮")


@st.cache_data(ttl=300)
def fetch_ohlcv(coin_id: str, days: int) -> pd.DataFrame:
    fetcher = CoinGeckoFetcher()
    return fetcher.get_historical_ohlc(coin_id, days=days)


@st.cache_data(ttl=300)
def fetch_price(coin_id: str) -> float:
    fetcher = CoinGeckoFetcher()
    info = fetcher.get_price(coin_id)
    return float(info.get("usd", 0.0))


@st.cache_data(ttl=300)
def fetch_market_data(coin_id: str) -> dict:
    fetcher = CoinGeckoFetcher()
    return fetcher.get_market_data(coin_id)


@st.cache_data(ttl=600)
def get_prediction(coin_id: str) -> dict:
    service = CryptoPredictorService()
    return service.get_prediction(coin_id)


def _candlestick_fig(df: pd.DataFrame) -> object:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price", increasing_line_color="lime", decreasing_line_color="red",
        ),
        row=1, col=1,
    )
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SMA_20"], name="SMA 20", line=dict(color="orange", width=1)), row=1, col=1)
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SMA_50"], name="SMA 50", line=dict(color="cyan", width=1)), row=1, col=1)
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color="rgba(100,100,255,0.5)"),
        row=2, col=1,
    )
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def _forecast_fig(current_price: float, forecast: list) -> object:
    hours = list(range(1, len(forecast) + 1))
    fig = go.Figure()
    fig.add_hline(y=current_price, line_dash="dash", line_color="white", annotation_text="Current")
    fig.add_trace(go.Scatter(x=hours, y=forecast, mode="lines+markers", name="Forecast", line=dict(color="cyan")))
    fig.update_layout(template="plotly_dark", title="24h Price Forecast", xaxis_title="Hours Ahead", yaxis_title="Price (USD)", height=350)
    return fig


def _rsi_fig(df: pd.DataFrame) -> object:
    fig = go.Figure()
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI_14"], name="RSI 14", line=dict(color="purple")))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(template="plotly_dark", title="RSI (14)", height=300)
    return fig


def _macd_fig(df: pd.DataFrame) -> object:
    fig = go.Figure()
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD"], name="MACD", line=dict(color="cyan")))
    if "MACD_signal" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD_signal"], name="Signal", line=dict(color="orange")))
    if "MACD_hist" in df.columns:
        colors = ["lime" if v >= 0 else "red" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["MACD_hist"], name="Histogram", marker_color=colors))
    fig.update_layout(template="plotly_dark", title="MACD", height=300)
    return fig


def _bb_fig(df: pd.DataFrame) -> object:
    fig = go.Figure()
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BB_upper"], name="Upper", line=dict(color="red", dash="dot")))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BB_middle"], name="Middle", line=dict(color="white")))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BB_lower"], name="Lower", line=dict(color="green", dash="dot"),
                                  fill="tonexty", fillcolor="rgba(100,200,100,0.05)"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close", line=dict(color="cyan", width=1)))
    fig.update_layout(template="plotly_dark", title="Bollinger Bands", height=350)
    return fig


def main() -> None:
    """Main Streamlit application entry point."""
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("# 🔮 OctaCryptoOracle")
        st.markdown("*AI-Powered Crypto Predictions*")
        st.divider()
        coin_symbol = st.selectbox("Select Coin", list(COIN_MAP.keys()), index=0)
        timeframe_label = st.selectbox("Timeframe", list(TIMEFRAME_DAYS.keys()), index=3)
        st.divider()
        predict_btn = st.button("🔮 Predict Next 24h", use_container_width=True)
        backtest_btn = st.button("📊 Run Backtest", use_container_width=True)
        auto_refresh = st.toggle("Auto-Refresh (5 min)", value=False)
        st.divider()
        st.info("💡 **DISCLAIMER:** For educational purposes only. Not financial advice.")

    coin_id = COIN_MAP[coin_symbol]
    days = TIMEFRAME_DAYS[timeframe_label]

    # Fetch base data
    with st.spinner("Loading market data..."):
        df_raw = fetch_ohlcv(coin_id, days)
        current_price = fetch_price(coin_id)
        market_data = fetch_market_data(coin_id)
        try:
            df_ind = add_all_indicators(df_raw.copy())
        except Exception:
            df_ind = df_raw.copy()

    # --- Header metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{coin_symbol} Price", f"${current_price:,.2f}", f"{market_data.get('price_change_24h', 0):.2f}%")
    col2.metric("Market Cap", f"${market_data.get('market_cap', 0)/1e9:.1f}B")
    col3.metric("24h Volume", f"${market_data.get('total_volume', 0)/1e9:.2f}B")
    col4.metric("7d Change", f"{market_data.get('price_change_7d', 0):.2f}%")

    # --- Tabs ---
    tab_price, tab_pred, tab_bt, tab_ind = st.tabs(
        ["📈 Price Chart", "🔮 Predictions", "📊 Backtest", "⚙️ Technical Indicators"]
    )

    with tab_price:
        st.subheader(f"{coin_symbol} Price Chart ({timeframe_label})")
        if _PLOTLY_AVAILABLE:
            df_plot = df_ind if len(df_ind) > 0 else df_raw
            st.plotly_chart(_candlestick_fig(df_plot), use_container_width=True)
        else:
            st.line_chart(df_raw.set_index("timestamp")["close"])

    with tab_pred:
        st.subheader("🔮 AI Price Prediction")
        if predict_btn or st.session_state.get("prediction"):
            with st.spinner("Running AI models..."):
                try:
                    pred = get_prediction(coin_id)
                    st.session_state["prediction"] = pred
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred = None

            pred = pred or st.session_state.get("prediction")
            if pred:
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${pred['current_price']:,.2f}")
                c2.metric("24h Prediction", f"${pred['predicted_price_24h']:,.2f}", f"{pred['price_change_pct']:.2f}%")
                signal_color = "🟢" if pred["signal_label"] == "BUY" else "🔴" if pred["signal_label"] == "SELL" else "🟡"
                c3.metric("Signal", f"{signal_color} {pred['signal_label']}")

                st.metric("Confidence", f"{pred['confidence']*100:.1f}%")
                st.metric("Direction", pred["direction"])

                if _PLOTLY_AVAILABLE and pred.get("price_forecast_24h"):
                    st.plotly_chart(_forecast_fig(pred["current_price"], pred["price_forecast_24h"]), use_container_width=True)
        else:
            st.info("Click **🔮 Predict Next 24h** in the sidebar to run the AI models.")

    with tab_bt:
        st.subheader("📊 Backtesting Results")
        if backtest_btn or st.session_state.get("backtest"):
            with st.spinner("Running backtest..."):
                try:
                    pred_bt = get_prediction(coin_id)
                    bt_res = pred_bt.get("backtest_results", {})
                    st.session_state["backtest"] = bt_res
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    bt_res = None

            bt_res = bt_res or st.session_state.get("backtest")
            if bt_res:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{bt_res.get('total_return_pct', 0):.1f}%")
                m2.metric("Sharpe Ratio", f"{bt_res.get('sharpe_ratio', 0):.2f}")
                m3.metric("Max Drawdown", f"{bt_res.get('max_drawdown', 0):.1f}%")
                m4.metric("Win Rate", f"{bt_res.get('win_rate', 0):.1f}%")
                st.caption(f"Buy & Hold Return: {bt_res.get('buy_hold_return', 0):.1f}%  |  Total Trades: {bt_res.get('total_trades', 0)}")
                if _PLOTLY_AVAILABLE and bt_res.get("equity_curve"):
                    bt = Backtester()
                    fig = bt.plot_equity_curve(bt_res)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **📊 Run Backtest** in the sidebar to analyse historical performance.")

    with tab_ind:
        st.subheader("⚙️ Technical Indicators")
        if len(df_ind) > 0 and _PLOTLY_AVAILABLE:
            st.plotly_chart(_rsi_fig(df_ind), use_container_width=True)
            st.plotly_chart(_macd_fig(df_ind), use_container_width=True)
            st.plotly_chart(_bb_fig(df_ind), use_container_width=True)
        else:
            st.warning("Technical indicators unavailable (install pandas-ta and plotly).")

    if auto_refresh:
        import time
        time.sleep(300)
        st.rerun()


if __name__ == "__main__" or _ST_AVAILABLE:
    main()
