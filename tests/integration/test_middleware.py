import re

from fastapi.testclient import TestClient

from mlops_sales_forecasting.api.app import create_app

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{12}$"
)


def build_client() -> TestClient:
    app = create_app(
        model_manager=None,
        load_model_on_startup=False,
    )
    return TestClient(
        app,
        raise_server_exceptions=False,
    )


def test_response_contains_generated_request_id() -> None:
    with build_client() as client:
        response = client.get("/livez")

    request_id = response.headers["X-Request-ID"]

    assert response.status_code == 200
    assert UUID_PATTERN.fullmatch(request_id)


def test_response_preserves_valid_request_id() -> None:
    with build_client() as client:
        response = client.get(
            "/livez",
            headers={
                "X-Request-ID": "client-request-123",
            },
        )

    assert response.headers["X-Request-ID"] == (
        "client-request-123"
    )


def test_invalid_request_id_is_replaced() -> None:
    with build_client() as client:
        response = client.get(
            "/livez",
            headers={
                "X-Request-ID": "invalid request id!",
            },
        )

    request_id = response.headers["X-Request-ID"]

    assert request_id != "invalid request id!"
    assert UUID_PATTERN.fullmatch(request_id)


def test_response_contains_process_time() -> None:
    with build_client() as client:
        response = client.get("/livez")

    process_time = float(
        response.headers["X-Process-Time-Ms"]
    )

    assert process_time >= 0.0


def test_unhandled_error_returns_safe_response() -> None:
    app = create_app(
        model_manager=None,
        load_model_on_startup=False,
    )

    @app.get("/raise-error")
    def raise_error() -> None:
        raise RuntimeError(
            "sensitive internal information"
        )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.get(
            "/raise-error",
            headers={
                "X-Request-ID": "failing-request-1",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "error": "internal_server_error",
        "request_id": "failing-request-1",
    }
    assert (
        "sensitive internal information"
        not in response.text
    )
    assert response.headers["X-Request-ID"] == (
        "failing-request-1"
    )