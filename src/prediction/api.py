"""FastAPI application for OctaCryptoOracle predictions."""

import asyncio
import logging
import os
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

logger = logging.getLogger(__name__)
_GENERIC_ERROR = "Internal server error"
_BUSY_ERROR = "Service temporarily busy"


def allowed_origins() -> list[str]:
    """Return explicit browser origins permitted to call the API."""
    raw = os.getenv("OCO_ALLOWED_ORIGINS", "")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    if any(origin == "*" for origin in origins):
        raise ValueError("Wildcard CORS origins are not allowed")
    return origins


def compute_limit() -> int:
    """Return the maximum number of concurrent prediction or backtest jobs."""
    raw = os.getenv("OCO_MAX_COMPUTE_JOBS", "1")
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("OCO_MAX_COMPUTE_JOBS must be a positive integer") from exc
    if limit < 1:
        raise ValueError("OCO_MAX_COMPUTE_JOBS must be a positive integer")
    return limit


class ComputeLimiter:
    """Rejects expensive work once the configured number of jobs is running."""

    def __init__(self) -> None:
        self._active = 0
        self._lock = asyncio.Lock()

    async def run(self, func, *args):
        async with self._lock:
            if self._active >= compute_limit():
                raise HTTPException(status_code=503, detail=_BUSY_ERROR)
            self._active += 1
        try:
            return await asyncio.to_thread(func, *args)
        finally:
            async with self._lock:
                self._active -= 1


def _backtest(service: CryptoPredictorService, coin_id: str) -> dict:
    df, _ = service.prepare_model_data(coin_id)
    results = Backtester().run(df, [1] * len(df))
    results.pop("equity_curve", None)
    return {"coin": coin_id, "backtest": results}


def create_app() -> "FastAPI":
    """Create an API instance using the current environment configuration."""
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required to create the API.")

    origins = allowed_origins()
    compute_limit()
    application = FastAPI(title="OctaCryptoOracle API", version="1.0.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(origins),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    service = CryptoPredictorService()
    fetcher = CoinGeckoFetcher()
    limiter = ComputeLimiter()
    application.state.service = service
    application.state.fetcher = fetcher
    application.state.compute_limiter = limiter

    @application.get("/")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "service": "OctaCryptoOracle"}

    @application.get("/supported-coins")
    async def supported_coins():
        """Return list of supported coin IDs."""
        return {"coins": SUPPORTED_COINS}

    @application.get("/price/{coin_id}")
    async def get_price(coin_id: str):
        """Return current USD price for *coin_id*."""
        if coin_id not in SUPPORTED_COINS:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin_id}")
        try:
            price_info = await asyncio.to_thread(fetcher.get_price, coin_id)
        except Exception:
            logger.exception("Price lookup failed for %s", coin_id)
            raise HTTPException(status_code=500, detail=_GENERIC_ERROR) from None
        return {"coin": coin_id, "price_usd": price_info.get("usd")}

    @application.get("/predict/{coin_id}")
    async def predict(coin_id: str):
        """Return full prediction for *coin_id*."""
        if coin_id not in SUPPORTED_COINS:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin_id}")
        try:
            return await limiter.run(service.get_prediction, coin_id)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Prediction failed for %s", coin_id)
            raise HTTPException(status_code=500, detail=_GENERIC_ERROR) from None

    @application.get("/backtest/{coin_id}")
    async def backtest(coin_id: str):
        """Run simple backtest for *coin_id* and return results."""
        if coin_id not in SUPPORTED_COINS:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin_id}")
        try:
            return await limiter.run(_backtest, service, coin_id)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Backtest failed for %s", coin_id)
            raise HTTPException(status_code=500, detail=_GENERIC_ERROR) from None

    return application


if _FASTAPI_AVAILABLE:
    app = create_app()
