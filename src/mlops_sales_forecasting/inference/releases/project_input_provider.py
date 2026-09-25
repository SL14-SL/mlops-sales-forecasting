from collections.abc import Mapping
from typing import Any

from mlops_sales_forecasting.configs.paths import (
    join_uri,
)
from mlops_sales_forecasting.inference.releases.artifact_publisher import (
    ServingArtifactSource,
)
from mlops_sales_forecasting.inference.releases.contracts import (
    TaskType,
)
from mlops_sales_forecasting.inference.releases.input_provider import (
    ServingReleaseInput,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


def _require_path(
    config: Mapping[str, Any],
    name: str,
) -> str:
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a valid 'paths' section.")

    value = paths.get(name)

    if not isinstance(value, str) or not value:
        raise ValueError(f"Config must contain a non-empty 'paths.{name}' value.")

    return value


class RossmannServingReleaseInputProvider:
    """Build forecasting-specific serving-release inputs."""

    def build_release_input(
        self,
        *,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
        config: Mapping[str, Any],
    ) -> ServingReleaseInput:
        del training_result

        model_config = config.get(
            "model",
            {},
        )
        training_config = config.get(
            "training",
            {},
        )

        model_type = model_config.get("type")

        if not isinstance(model_type, str) or not model_type:
            raise ValueError("Config must define model.type.")

        target_transformation = training_config.get(
            "target_transformation",
            "none",
        )

        if (
            not isinstance(
                target_transformation,
                str,
            )
            or not target_transformation
        ):
            raise ValueError("Training target transformation must not be empty.")

        validated_path = _require_path(
            config,
            "validated_data",
        )
        features_path = _require_path(
            config,
            "features",
        )
        models_path = _require_path(
            config,
            "models",
        )

        return ServingReleaseInput(
            task_type=TaskType.FORECASTING,
            model_type=model_type,
            sources={
                "store_metadata": (
                    ServingArtifactSource(
                        source_uri=join_uri(
                            validated_path,
                            "store.parquet",
                        ),
                        relative_path=("store_metadata.parquet"),
                    )
                ),
                "store_state": (
                    ServingArtifactSource(
                        source_uri=join_uri(
                            models_path,
                            "latest_state.json",
                        ),
                        relative_path=("store_state.json"),
                    )
                ),
                "known_calendar": (
                    ServingArtifactSource(
                        source_uri=join_uri(
                            features_path,
                            "known_calendar.parquet",
                        ),
                        relative_path=("known_calendar.parquet"),
                    )
                ),
            },
            metadata={
                "target_transformation": (target_transformation),
                "evaluation_metrics": dict(evaluation_result.metrics),
            },
        )
