from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.inference import model_loader


def test_configure_mlflow_sets_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_tracking_uri = MagicMock()
    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        set_tracking_uri,
    )

    model_loader.configure_mlflow(
        "http://localhost:5000"
    )

    set_tracking_uri.assert_called_once_with(
        "http://localhost:5000"
    )


def test_configure_mlflow_rejects_empty_uri() -> None:
    with pytest.raises(
        ValueError,
        match="tracking URI must not be empty",
    ):
        model_loader.configure_mlflow("")


def test_build_registered_model_uri() -> None:
    result = model_loader.build_registered_model_uri(
        model_name="customer-churn-model-dev",
        alias="champion",
    )

    assert result == (
        "models:/customer-churn-model-dev@champion"
    )


@pytest.mark.parametrize(
    ("model_name", "alias"),
    [
        ("", "champion"),
        ("example-model", ""),
    ],
)
def test_build_registered_model_uri_rejects_empty_values(
    model_name: str,
    alias: str,
) -> None:
    with pytest.raises(ValueError):
        model_loader.build_registered_model_uri(
            model_name=model_name,
            alias=alias,
        )


def test_load_pyfunc_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_model = MagicMock()
    load_model = MagicMock(
        return_value=expected_model
    )
    monkeypatch.setattr(
        model_loader.mlflow.pyfunc,
        "load_model",
        load_model,
    )

    result = model_loader.load_pyfunc_model(
        "models:/example-model@champion"
    )

    assert result is expected_model
    load_model.assert_called_once_with(
        "models:/example-model@champion"
    )


def test_load_pyfunc_model_rejects_empty_uri() -> None:
    with pytest.raises(
        ValueError,
        match="model URI must not be empty",
    ):
        model_loader.load_pyfunc_model("")