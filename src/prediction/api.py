"""FastAPI application for OctaCryptoOracle predictions."""

import warnings

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    warnings.warn("FastAPI not installed. API module inactive.")

from src.data.coingecko_fetcher import CoinGeckoFetcher
from src.models.backtester import Backtester
from src.prediction.predictor import CryptoPredictorService
from src.utils.config import SUPPORTED_COINS

if _FASTAPI_AVAILABLE:
    app = FastAPI(title="OctaCryptoOracle API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _service = CryptoPredictorService()
    _fetcher = CoinGeckoFetcher()

    @app.get("/")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "service": "OctaCryptoOracle"}

    @app.get("/supported-coins")
    async def supported_coins():
        """Return list of supported coin IDs."""
        return {"coins": SUPPORTED_COINS}

    @app.get("/price/{coin_id}")
    async def get_price(coin_id: str):
        """Return current USD price for *coin_id*."""
        if coin_id not in SUPPORTED_COINS:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin_id}")
        price_info = _fetcher.get_price(coin_id)
        return {"coin": coin_id, "price_usd": price_info.get("usd")}

    @app.get("/predict/{coin_id}")
    async def predict(coin_id: str):
        """Return full prediction for *coin_id*."""
        if coin_id not in SUPPORTED_COINS:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin_id}")
        try:
            result = _service.get_prediction(coin_id)
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/backtest/{coin_id}")
    async def backtest(coin_id: str):
        """Run simple backtest for *coin_id* and return results."""
        if coin_id not in SUPPORTED_COINS:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin_id}")
        try:
            df, _ = _service.prepare_model_data(coin_id)
            signals = [1] * len(df)
            bt = Backtester()
            results = bt.run(df, signals)
            results.pop("equity_curve", None)
            return {"coin": coin_id, "backtest": results}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
