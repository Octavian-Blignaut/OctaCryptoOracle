# Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the confirmed local-deployment security weaknesses without adding authentication.

**Architecture:** Build the FastAPI app through `create_app()` so CORS and the compute limit come from environment settings at construction time. Expensive prediction and backtest work shares one in-process limiter and runs in a worker thread. Helius credentials move to request headers, and the container build excludes secret files while running as an unprivileged user.

**Tech Stack:** Python 3.11, FastAPI, requests, Docker, pytest.

## Global Constraints

- Local use only; do not add user authentication.
- CORS denies every cross-origin request unless `OCO_ALLOWED_ORIGINS` lists explicit origins.
- Wildcard CORS origins are rejected, and credentials are enabled only for an explicit origin list.
- Unexpected API failures return `Internal server error`; saturated compute routes return `Service temporarily busy`.
- Helius credentials must not appear in request URLs.
- The container process runs as non-root `appuser`.
- Documented local API and Docker port bindings use `127.0.0.1`.

---

### Task 1: API trust boundaries

**Files:**
- Modify: `src/prediction/api.py`
- Test: `tests/test_security.py`

**Interfaces:**
- Consumes: `CryptoPredictorService.get_prediction(coin_id: str) -> dict`, `CryptoPredictorService.prepare_model_data(coin_id: str, days: int = 365)`, `SUPPORTED_COINS`
- Produces: `create_app() -> FastAPI`, module-level `app`, `ComputeLimiter.run(func, *args)`, `allowed_origins() -> list[str]`, `compute_limit() -> int`

- [ ] Write tests that prove generic 500 responses, default-deny CORS, explicit origin allowlisting, wildcard rejection, and immediate HTTP 503 when the compute limit is occupied.
- [ ] Run `pytest tests/test_security.py -v` and confirm those API assertions fail.
- [ ] Implement `create_app()` with the limiter and safe handlers.
- [ ] Re-run the API security tests and confirm they pass.

### Task 2: Secret transport and image context

**Files:**
- Modify: `src/data/solana_fetcher.py`
- Create: `.dockerignore`
- Test: `tests/test_security.py`

**Interfaces:**
- Consumes: `get_config().HELIUS_API_KEY`
- Produces: Helius requests whose URL contains no API key and whose `X-Api-Key` header carries the credential

- [ ] Assert both Helius calls use the bearer header and `.dockerignore` excludes `.env`.
- [ ] Confirm the assertions fail against the current URL-based requests.
- [ ] Move the key to the header and exclude secret/build files from the Docker context.
- [ ] Re-run the secret tests.

### Task 3: Runtime defaults and dependency audit

**Files:**
- Modify: `Dockerfile`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `requirements.txt`
- Test: `tests/test_security.py`

**Interfaces:**
- Consumes: existing Streamlit command and CI test job
- Produces: non-root image, loopback documentation, and a `pip-audit` CI step

- [ ] Assert the Dockerfile creates and switches to `appuser`.
- [ ] Add `httpx` for FastAPI's test client, run the image as `appuser`, bind documented local services to loopback, and audit dependencies in CI.
- [ ] Run `pytest tests/ -v` and `pip-audit -r requirements.txt`.
