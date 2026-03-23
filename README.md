# 🔮 OctaCryptoOracle

> AI-powered cryptocurrency price prediction platform for Bitcoin, Solana, and Ethereum.

[![CI](https://github.com/OctaCryptoOracle/OctaCryptoOracle/actions/workflows/ci.yml/badge.svg)](https://github.com/OctaCryptoOracle/OctaCryptoOracle/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

OctaCryptoOracle combines **LSTM neural networks**, **XGBoost signal classification**, and **30+ technical indicators** to generate actionable buy/sell signals and 24-hour price forecasts for BTC, SOL, and ETH.

### Features

- 📊 **Real-time data** from CoinGecko + Binance (via CCXT) + Solana on-chain metrics
- 🤖 **Ensemble AI models**: LSTM for price forecasting + XGBoost for signal classification
- 📈 **30+ technical indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, MFI, and more
- 🔮 **24-hour price forecasts** with confidence scoring
- 📉 **Backtesting engine** with Sharpe ratio, drawdown, and win-rate metrics
- 🌐 **Streamlit dashboard** with interactive candlestick charts
- ⚡ **FastAPI REST API** for programmatic access
- 🐳 **Docker support** for easy deployment

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/OctaCryptoOracle/OctaCryptoOracle.git
cd OctaCryptoOracle
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (optional - mock data works without keys)
```

### 3. Run the Dashboard

```bash
streamlit run app/dashboard.py
```

### 4. Run the API

```bash
uvicorn src.prediction.api:app --reload --port 8000
```

### 5. Docker

```bash
docker build -t octacryptooracle .
docker run -p 8501:8501 octacryptooracle
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/supported-coins` | List supported coins |
| GET | `/price/{coin_id}` | Current price |
| GET | `/predict/{coin_id}` | Full AI prediction |
| GET | `/backtest/{coin_id}` | Backtesting results |

**Supported coin IDs:** `bitcoin`, `solana`, `ethereum`

## Architecture

```
src/
├── data/          # CoinGecko, CCXT, Solana fetchers
├── features/      # Technical indicator computation (pandas-ta)
├── models/        # LSTM, XGBoost, Ensemble, Backtester
├── prediction/    # High-level service + FastAPI app
└── utils/         # Configuration management
```

## Supported Coins

| Symbol | CoinGecko ID | Exchange Pair |
|--------|-------------|---------------|
| BTC | bitcoin | BTC/USDT |
| SOL | solana | SOL/USDT |
| ETH | ethereum | ETH/USDT |

## Model Details

### LSTM Model
- 2-layer LSTM with dropout regularization
- Sequence length: 60 days
- Auto-regressive 24-step forecast
- MinMax scaled inputs

### XGBoost Signal Model
- Binary classification: price up (1) vs down (0) next day
- 24 engineered features from technical indicators
- 300 estimators, depth 6

### Ensemble
- Weighted combination: 60% LSTM + 40% XGBoost
- Confidence scoring based on both models

## Testing

```bash
pytest tests/ -v
```

## ⚠️ Disclaimer

This platform is for educational purposes only. Cryptocurrency markets are highly volatile, and you should not make financial decisions based solely on AI predictions. Always do your own research. Through active collaboration, we can refine these insights for the benefit of the entire community.

## License

[MIT License](LICENSE) © 2024 OctaCryptoOracle Contributors
