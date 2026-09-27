from prometheus_client import Counter, Gauge, Histogram

SERVING_READY = Gauge(
    "mlops_serving_ready",
    ("Whether a complete serving bundle is currently active."),
)

REQUEST_COUNT = Counter(
    "mlops_api_requests_total",
    "Total number of observed API requests.",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "mlops_api_request_latency_seconds",
    "API request latency in seconds.",
    ["method", "path"],
)

IGNORED_PATHS = {
    "/docs",
    "/docs/oauth2-redirect",
    "/livez",
    "/metrics",
    "/openapi.json",
    "/readyz",
    "/redoc",
    "/monitoring/summary",
}


def should_ignore_path(path: str) -> bool:
    """Return whether a path should be excluded from serving metrics."""
    return path in IGNORED_PATHS


def normalize_path(
    raw_path: str,
    route_path: str | None,
) -> str:
    """Return a bounded route label for Prometheus."""
    if route_path:
        return route_path

    if raw_path in IGNORED_PATHS:
        return raw_path

    return "/unmatched"


def observe_request(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_seconds: float,
) -> None:
    """Record request count and latency."""
    normalized_method = method.upper()
    normalized_status = str(status_code)

    REQUEST_COUNT.labels(
        method=normalized_method,
        path=path,
        status_code=normalized_status,
    ).inc()

    REQUEST_LATENCY.labels(
        method=normalized_method,
        path=path,
    ).observe(latency_seconds)


def set_serving_readiness(
    is_ready: bool,
) -> None:
    """Update the current serving-readiness state."""
    SERVING_READY.set(1 if is_ready else 0)
