from unittest.mock import MagicMock

from fastapi import FastAPI

from mlops_sales_forecasting.api import bootstrap


def build_config() -> dict:
    return {
        "project": {
            "name": "Example Model",
        },
        "paths": {
            "models": "artifacts/models",
        },
        "tracking": {
            "mlflow_tracking_uri": "http://localhost:5000",
        },
        "serving": {
            "alias": "champion",
        },
        "api": {
            "api_key": "test-api-key",
        },
    }


def test_build_model_manager_uses_active_bundle_loader(
    monkeypatch,
) -> None:
    expected_bundle = MagicMock()
    load_active_bundle = MagicMock(
        return_value=expected_bundle
    )
    model_manager = MagicMock()
    model_manager_constructor = MagicMock(
        return_value=model_manager
    )

    monkeypatch.setattr(
        bootstrap,
        "load_active_bundle",
        load_active_bundle,
    )
    monkeypatch.setattr(
        bootstrap,
        "ModelManager",
        model_manager_constructor,
    )

    config = build_config()
    result = bootstrap.build_model_manager(config)

    assert result is model_manager

    captured_loader = (
        model_manager_constructor.call_args.args[0]
    )
    assert captured_loader() is expected_bundle
    load_active_bundle.assert_called_once_with(config)


def test_build_application_loads_default_config(
    monkeypatch,
) -> None:
    config = build_config()
    load_config = MagicMock(return_value=config)
    model_manager = MagicMock()
    expected_app = FastAPI()
    create_app = MagicMock(return_value=expected_app)

    monkeypatch.setattr(
        bootstrap,
        "load_config",
        load_config,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_model_manager",
        MagicMock(return_value=model_manager),
    )
    monkeypatch.setattr(
        bootstrap,
        "create_app",
        create_app,
    )

    result = bootstrap.build_application(
        load_model_on_startup=False
    )

    assert result is expected_app
    load_config.assert_called_once_with()
    create_app.assert_called_once_with(
        model_manager=model_manager,
        load_model_on_startup=False,
        title="Example Model API",
        api_key="test-api-key",
    )


def test_build_application_uses_provided_config(
    monkeypatch,
) -> None:
    config = build_config()
    load_config = MagicMock()
    model_manager = MagicMock()
    expected_app = FastAPI()

    monkeypatch.setattr(
        bootstrap,
        "load_config",
        load_config,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_model_manager",
        MagicMock(return_value=model_manager),
    )
    monkeypatch.setattr(
        bootstrap,
        "create_app",
        MagicMock(return_value=expected_app),
    )

    result = bootstrap.build_application(
        config=config,
        load_model_on_startup=False,
    )

    assert result is expected_app
    load_config.assert_not_called()


def test_build_application_uses_fallback_title(
    monkeypatch,
) -> None:
    config = build_config()
    config["project"] = {}

    monkeypatch.setattr(
        bootstrap,
        "build_model_manager",
        MagicMock(),
    )
    create_app = MagicMock(
        return_value=FastAPI()
    )
    monkeypatch.setattr(
        bootstrap,
        "create_app",
        create_app,
    )

    bootstrap.build_application(
        config=config,
        load_model_on_startup=False,
    )

    assert create_app.call_args.kwargs["title"] == (
        "MLOps Model API"
    )