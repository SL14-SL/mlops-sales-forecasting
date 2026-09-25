from unittest.mock import MagicMock, patch

import pytest

from mlops_sales_forecasting.tracking.aliases import (
    ModelAlias,
    restore_champion,
)
from mlops_sales_forecasting.tracking.registry import (
    ModelRegistrationResult,
)

CONFIG = {
    "tracking": {
        "mlflow_tracking_uri": (
            "http://localhost:5000"
        ),
    }
}


def build_registration() -> ModelRegistrationResult:
    return ModelRegistrationResult(
        registered=True,
        run_id="run-7",
        model_name="example-model",
        model_version="7",
        model_uri="models:/example-model/7",
    )


@patch(
    "mlops_sales_forecasting.tracking.aliases."
    "MlflowClient"
)
def test_restores_previous_champion(
    client_class: MagicMock,
) -> None:
    client = client_class.return_value

    result = restore_champion(
        registration=build_registration(),
        previous_champion_version="6",
        config=CONFIG,
    )

    client.set_registered_model_alias.assert_called_once_with(
        name="example-model",
        alias="champion",
        version="6",
    )
    client.delete_registered_model_alias.assert_not_called()

    assert result is not None
    assert result.model_name == "example-model"
    assert result.model_version == "6"
    assert result.alias is ModelAlias.CHAMPION


@patch(
    "mlops_sales_forecasting.tracking.aliases."
    "MlflowClient"
)
def test_removes_initial_champion_alias(
    client_class: MagicMock,
) -> None:
    client = client_class.return_value

    result = restore_champion(
        registration=build_registration(),
        previous_champion_version=None,
        config=CONFIG,
    )

    client.delete_registered_model_alias.assert_called_once_with(
        name="example-model",
        alias="champion",
    )
    client.set_registered_model_alias.assert_not_called()
    assert result is None


@patch(
    "mlops_sales_forecasting.tracking.aliases."
    "MlflowClient"
)
def test_rejects_empty_previous_version(
    client_class: MagicMock,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string or None",
    ):
        restore_champion(
            registration=build_registration(),
            previous_champion_version="",
            config=CONFIG,
        )

    client_class.assert_not_called()


@patch(
    "mlops_sales_forecasting.tracking.aliases."
    "MlflowClient"
)
def test_rejects_unregistered_candidate(
    client_class: MagicMock,
) -> None:
    registration = ModelRegistrationResult(
        registered=False,
        run_id="run-7",
        model_name="example-model",
    )

    with pytest.raises(
        ValueError,
        match="unregistered model candidate",
    ):
        restore_champion(
            registration=registration,
            previous_champion_version="6",
            config=CONFIG,
        )

    client_class.assert_not_called()