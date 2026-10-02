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
        model_name="sales-forecasting-model-dev",
        alias="champion",
    )

    assert result == (
        "models:/sales-forecasting-model-dev@champion"
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


def test_load_xgboost_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = MagicMock()
    native_loader = MagicMock(
        return_value=estimator
    )

    input_schema = MagicMock()
    input_schema.input_names.return_value = [
        "Store",
        "Promo",
        "sales_lag_1",
    ]

    signature = MagicMock()
    signature.inputs = input_schema

    model_metadata = MagicMock()
    model_metadata.signature = signature

    load_metadata = MagicMock(
        return_value=model_metadata
    )

    monkeypatch.setattr(
        model_loader.Model,
        "load",
        load_metadata,
    )
    monkeypatch.setattr(
        model_loader.mlflow.xgboost,
        "load_model",
        native_loader,
    )

    result = model_loader.load_xgboost_model(
        "models:/m-test-model"
    )

    assert result.estimator is estimator
    assert result.input_columns == (
        "Store",
        "Promo",
        "sales_lag_1",
    )

    load_metadata.assert_called_once_with(
        "models:/m-test-model"
    )
    native_loader.assert_called_once_with(
        "models:/m-test-model"
    )


def test_load_xgboost_model_requires_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_metadata = MagicMock()
    model_metadata.signature = None

    monkeypatch.setattr(
        model_loader.Model,
        "load",
        MagicMock(
            return_value=model_metadata
        ),
    )

    with pytest.raises(
        ValueError,
        match="no input signature",
    ):
        model_loader.load_xgboost_model(
            "models:/m-test-model"
        )


def test_loaded_xgboost_model_delegates_prediction() -> None:
    estimator = MagicMock()
    estimator.predict.return_value = [
        1.0,
    ]
    loaded_model = model_loader.LoadedXGBoostModel(
        estimator=estimator,
        input_columns=(
            "feature",
        ),
    )
    model_input = MagicMock()

    result = loaded_model.predict(
        model_input
    )

    assert result == [
        1.0,
    ]
    estimator.predict.assert_called_once_with(
        model_input
    )