"""Regression tests for API and deployment security boundaries."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _app(monkeypatch, jobs="1", origins=None):
    monkeypatch.setenv("OCO_MAX_COMPUTE_JOBS", jobs)
    if origins is None:
        monkeypatch.delenv("OCO_ALLOWED_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("OCO_ALLOWED_ORIGINS", origins)
    from src.prediction.api import create_app

    application = create_app()
    application.state.fetcher.get_price = lambda coin_id, vs_currency="usd": {"usd": 100.0}
    return application


def test_unexpected_errors_hide_internal_details(monkeypatch, caplog):
    application = _app(monkeypatch)

    def fail(coin_id):
        raise RuntimeError("database password=hunter2")

    monkeypatch.setattr(application.state.service, "get_prediction", fail)
    monkeypatch.setattr(application.state.service, "prepare_model_data", fail)

    with caplog.at_level("ERROR"):
        predict = TestClient(application).get("/predict/bitcoin")
        backtest = TestClient(application).get("/backtest/bitcoin")

    assert predict.status_code == 500
    assert backtest.status_code == 500
    assert predict.json()["detail"] == "Internal server error"
    assert backtest.json()["detail"] == "Internal server error"
    assert "hunter2" not in predict.text
    assert "hunter2" not in backtest.text
    assert "hunter2" in caplog.text


def test_unsupported_coin_remains_explicit(monkeypatch):
    response = TestClient(_app(monkeypatch)).get("/predict/dogecoin")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unsupported coin: dogecoin"


def test_cors_denies_origins_by_default(monkeypatch):
    response = TestClient(_app(monkeypatch)).get(
        "/price/bitcoin",
        headers={"Origin": "https://evil.example"},
    )

    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_configured_origin(monkeypatch):
    response = TestClient(_app(monkeypatch, origins="http://127.0.0.1:8501")).get(
        "/price/bitcoin",
        headers={"Origin": "http://127.0.0.1:8501"},
    )

    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:8501"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_wildcard_cors_origin_is_rejected(monkeypatch):
    monkeypatch.setenv("OCO_ALLOWED_ORIGINS", "*")

    from src.prediction.api import create_app

    with pytest.raises(ValueError, match="Wildcard CORS origins"):
        create_app()


def test_saturated_compute_route_fails_quickly(monkeypatch):
    application = _app(monkeypatch, jobs="1")
    application.state.compute_limiter._active = 1
    monkeypatch.setattr(application.state.service, "get_prediction", lambda coin_id: {"ok": True})

    response = TestClient(application).get("/predict/bitcoin")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily busy"


def test_compute_slot_is_released_after_failure(monkeypatch):
    application = _app(monkeypatch, jobs="1")

    def fail(coin_id):
        raise RuntimeError("temporary model failure")

    monkeypatch.setattr(application.state.service, "get_prediction", fail)
    client = TestClient(application)

    assert client.get("/predict/bitcoin").status_code == 500
    assert application.state.compute_limiter._active == 0
    assert client.get("/predict/bitcoin").status_code == 500


def test_helius_key_is_not_sent_in_url(monkeypatch):
    monkeypatch.setenv("HELIUS_API_KEY", "super-secret-key")
    response = MagicMock()
    response.json.return_value = {}
    response.raise_for_status.return_value = None

    from src.data.solana_fetcher import SolanaFetcher

    with patch("src.data.solana_fetcher.requests.get", return_value=response) as mock_get:
        SolanaFetcher().get_on_chain_metrics()
        SolanaFetcher().get_token_holders("mint-address")

    for call in mock_get.call_args_list:
        url = call.args[0]
        headers = call.kwargs["headers"]
        assert "super-secret-key" not in url
        assert headers["X-Api-Key"] == "super-secret-key"


def test_helius_failure_warning_redacts_key(monkeypatch):
    monkeypatch.setenv("HELIUS_API_KEY", "super-secret-key")

    from src.data.solana_fetcher import SolanaFetcher

    with patch(
        "src.data.solana_fetcher.requests.get",
        side_effect=RuntimeError("request failed for super-secret-key"),
    ):
        with pytest.warns(UserWarning, match="Helius API failed") as caught:
            SolanaFetcher().get_on_chain_metrics()

    message = str(caught[0].message)
    assert "super-secret-key" not in message
    assert "[redacted]" in message


def test_docker_build_excludes_env_file():
    ignored = (ROOT / ".dockerignore").read_text()

    assert ".env" in ignored.splitlines()


def test_container_runs_as_non_root():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "useradd" in dockerfile
    assert "USER appuser" in dockerfile
