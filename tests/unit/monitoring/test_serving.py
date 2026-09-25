import pytest

from mlops_sales_forecasting.monitoring.serving import (
    REQUEST_COUNT,
    SERVING_READY,
    normalize_path,
    observe_request,
    set_serving_readiness,
    should_ignore_path,
)


@pytest.mark.parametrize(
    "path",
    [
        "/metrics",
        "/livez",
        "/readyz",
        "/docs",
        "/openapi.json",
    ],
)
def test_should_ignore_operational_paths(path: str) -> None:
    assert should_ignore_path(path) is True


def test_business_path_is_observed() -> None:
    assert should_ignore_path("/predict") is False


def test_normalize_path_uses_route_template() -> None:
    result = normalize_path(
        "/customers/123",
        "/customers/{customer_id}",
    )

    assert result == "/customers/{customer_id}"


def test_normalize_path_bounds_unknown_route() -> None:
    result = normalize_path(
        "/unknown/dynamic/value",
        None,
    )

    assert result == "/unmatched"


def test_observe_request_increments_counter() -> None:
    labels = {
        "method": "POST",
        "path": "/test-observe",
        "status_code": "200",
    }
    counter = REQUEST_COUNT.labels(**labels)
    before = counter._value.get()

    observe_request(
        method="post",
        path="/test-observe",
        status_code=200,
        latency_seconds=0.01,
    )

    assert counter._value.get() == before + 1

def test_set_serving_readiness_updates_gauge() -> None:
    set_serving_readiness(False)
    assert SERVING_READY._value.get() == 0

    set_serving_readiness(True)
    assert SERVING_READY._value.get() == 1
