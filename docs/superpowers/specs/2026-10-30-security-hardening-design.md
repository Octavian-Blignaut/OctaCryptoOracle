# Security Hardening Design

## Goal

Harden OctaCryptoOracle for local use so accidental network exposure does not reveal internal errors, permit unrestricted cross-origin calls, leak credentials into URLs, allow unbounded expensive prediction work, or run the application container as root.

## Scope

- FastAPI error handling, CORS policy, and expensive-route concurrency.
- Helius credential transport.
- Non-root Docker execution and localhost-first documentation.
- Security regression tests and dependency auditing in CI.

Authentication and reverse-proxy deployment controls are excluded because the application is intended for local use. They must be added before deliberate public deployment.

## Design

### API boundaries

The API will return a generic message for unexpected server errors and log the exception internally. Supported coin validation remains the existing allowlist.

CORS will deny cross-origin access by default. An `OCO_ALLOWED_ORIGINS` environment variable may contain a comma-separated allowlist. Credentials will only be enabled when explicit origins exist; wildcard origins will not be accepted.

Prediction and backtest handlers will share a small, configurable concurrency limiter. If all compute slots are occupied, new work will fail quickly with HTTP 503 instead of starting additional model training. Synchronous model work will run outside the event loop.

### Secret handling

Helius requests will send the API key in an authorization header rather than a query string. This prevents common URL and exception logs from recording the credential.

### Container and launch defaults

The Docker image will create and run as an unprivileged application user. Documentation will bind local Uvicorn examples to `127.0.0.1`. Docker port publication will likewise use loopback in the documented command.

### Dependency auditing

CI will run `pip-audit` against the declared requirements. The audit checks the resolver's current dependency set on every change. Exact reproducibility still requires a future lockfile; introducing a locking workflow is outside this focused hardening change.

## Error handling

Expected client errors retain their explicit status codes. Unexpected failures are logged with stack traces and become a generic `500 Internal server error`. Saturated compute routes return `503 Service temporarily busy`.

## Testing

Regression tests will verify:

- unexpected exception details are absent from API responses;
- CORS is denied by default and respects explicit allowed origins;
- concurrent expensive requests are rejected when the configured limit is reached;
- Helius credentials are sent in headers and not URLs;
- the Dockerfile declares a non-root runtime user.

The focused tests will be observed failing before implementation, then the full test suite will be run after the fixes.
