from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.tracking import aliases
from mlops_sales_forecasting.tracking.aliases import (
    AliasAssignment,
    ModelAlias,
    assign_challenger,
    promote_to_champion,
)
from mlops_sales_forecasting.tracking.registry import (
    ModelRegistrationResult,
)


def build_config() -> dict:
    return {
        "tracking": {
            "mlflow_tracking_uri": (
                "http://localhost:5000"
            ),
        },
    }


def build_registration(
    *,
    registered: bool = True,
) -> ModelRegistrationResult:
    return ModelRegistrationResult(
        registered=registered,
        run_id="mlflow-run-123",
        model_name="example-model-dev",
        model_version=(
            "7"
            if registered
            else None
        ),
        model_uri=(
            "models:/example-model-dev/7"
            if registered
            else None
        ),
        reason=(
            None
            if registered
            else "Candidate rejected."
        ),
    )


def test_candidate_receives_challenger_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client_factory = MagicMock(
        return_value=client
    )
    monkeypatch.setattr(
        aliases,
        "MlflowClient",
        client_factory,
    )

    result = assign_challenger(
        registration=build_registration(),
        config=build_config(),
    )

    assert result == AliasAssignment(
        model_name="example-model-dev",
        model_version="7",
        alias=ModelAlias.CHALLENGER,
    )

    client_factory.assert_called_once_with(
        tracking_uri="http://localhost:5000"
    )
    client.set_registered_model_alias.assert_called_once_with(
        name="example-model-dev",
        alias="challenger",
        version="7",
    )


def test_candidate_can_be_promoted_to_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    monkeypatch.setattr(
        aliases,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    result = promote_to_champion(
        registration=build_registration(),
        config=build_config(),
    )

    assert result.alias is ModelAlias.CHAMPION

    client.set_registered_model_alias.assert_called_once_with(
        name="example-model-dev",
        alias="champion",
        version="7",
    )


@pytest.mark.parametrize(
    "function",
    [
        assign_challenger,
        promote_to_champion,
    ],
)
def test_alias_rejects_unregistered_candidate(
    function,
) -> None:
    with pytest.raises(
        ValueError,
        match="unregistered model candidate",
    ):
        function(
            registration=build_registration(
                registered=False
            ),
            config=build_config(),
        )


@pytest.mark.parametrize(
    "function",
    [
        assign_challenger,
        promote_to_champion,
    ],
)
def test_alias_requires_registration_result(
    function,
) -> None:
    with pytest.raises(
        TypeError,
        match="ModelRegistrationResult",
    ):
        function(
            registration=object(),
            config=build_config(),
        )


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "",
        "   ",
        7,
        None,
        "${MLFLOW_TRACKING_URI}",
    ],
)
def test_tracking_uri_must_be_resolved(
    tracking_uri: object,
) -> None:
    config = build_config()
    config["tracking"][
        "mlflow_tracking_uri"
    ] = tracking_uri

    with pytest.raises(
        ValueError,
        match="mlflow_tracking_uri",
    ):
        assign_challenger(
            registration=build_registration(),
            config=config,
        )


def test_tracking_section_is_required() -> None:
    with pytest.raises(
        ValueError,
        match="'tracking' section",
    ):
        assign_challenger(
            registration=build_registration(),
            config={},
        )