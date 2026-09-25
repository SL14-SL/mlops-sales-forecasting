from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from mlops_sales_forecasting.inference.releases.artifact_publisher import (
    ServingArtifactSource,
)
from mlops_sales_forecasting.inference.releases.contracts import (
    TaskType,
)
from mlops_sales_forecasting.inference.releases.publisher import (
    publish_serving_release,
)
from mlops_sales_forecasting.inference.releases.repository import (
    load_active_release_manifest,
)
from mlops_sales_forecasting.tracking.lifecycle import (
    finalize_model_candidate,
)
from mlops_sales_forecasting.tracking.mlflow import (
    log_evaluation_result,
    log_training_result,
    start_training_run,
)
from mlops_sales_forecasting.tracking.model_artifact import (
    log_model_artifact,
)
from mlops_sales_forecasting.tracking.promotion import (
    MetricDirection,
    PromotionPolicy,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


class ConstantPythonModel(
    mlflow.pyfunc.PythonModel
):
    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[int]:
        del context, params

        return [1] * len(model_input)


class PyFuncModelArtifactLogger:
    def log_model(
        self,
        training_result: TrainingResult,
        *,
        artifact_path: str,
        config: Mapping[str, Any],
    ) -> str:
        del config

        model_info = mlflow.pyfunc.log_model(
            name=artifact_path,
            python_model=(
                training_result.model
            ),
        )

        return model_info.model_uri



TASK_TYPE = TaskType.FORECASTING
RELEASE_METADATA = {
    "target_transformation": "identity",
}


def build_release_sources(
    root: Path,
) -> dict[str, ServingArtifactSource]:
    store_metadata = (
        root / "store_metadata.parquet"
    )
    store_state = root / "store_state.json"
    known_calendar = (
        root / "known_calendar.parquet"
    )

    store_metadata.write_bytes(
        b"store metadata fixture"
    )
    store_state.write_text(
        '{"stores": {"1": {"lag_1": 100.0}}}',
        encoding="utf-8",
    )
    known_calendar.write_bytes(
        b"calendar fixture"
    )

    return {
        "store_metadata": (
            ServingArtifactSource(
                source_uri=str(
                    store_metadata
                ),
                relative_path=(
                    "store_metadata.parquet"
                ),
            )
        ),
        "store_state": (
            ServingArtifactSource(
                source_uri=str(store_state),
                relative_path=(
                    "store_state.json"
                ),
            )
        ),
        "known_calendar": (
            ServingArtifactSource(
                source_uri=str(
                    known_calendar
                ),
                relative_path=(
                    "known_calendar.parquet"
                ),
            )
        ),
    }



def test_real_mlflow_to_serving_release(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "mlflow.db"
    )
    artifact_root = (
        tmp_path / "mlflow-artifacts"
    )
    models_path = (
        tmp_path / "serving-models"
    )

    tracking_uri = (
        f"sqlite:///{database_path}"
    )
    experiment_name = (
        "mlops-sales-forecasting-integration"
    )
    model_name = (
        "mlops-sales-forecasting-model"
    )

    client = MlflowClient(
        tracking_uri=tracking_uri
    )
    client.create_experiment(
        name=experiment_name,
        artifact_location=(
            artifact_root.as_uri()
        ),
    )

    config = {
        "tracking": {
            "mlflow_tracking_uri": (
                tracking_uri
            ),
            "experiment_name": (
                experiment_name
            ),
            "model_name": model_name,
        },
        "paths": {
            "models": str(models_path),
        },
    }

    evaluation = EvaluationResult(
        metrics={"score": 0.91},
        approved=True,
    )

    with start_training_run(
        config,
        run_name="integration-test",
        tags={
            "test.type": "integration",
        },
    ) as run_id:
        training = TrainingResult(
            model=ConstantPythonModel(),
            run_id=run_id,
            metrics={
                "training_score": 0.89,
            },
            parameters={
                "constant_prediction": 1,
            },
        )

        log_training_result(training)
        log_evaluation_result(evaluation)

        logged_artifact = (
            log_model_artifact(
                logger=(
                    PyFuncModelArtifactLogger()
                ),
                training_result=training,
                config=config,
                artifact_path="model",
            )
        )

    assert (
        logged_artifact.model_uri.startswith(
            "models:/m-"
        )
    )

    policy = PromotionPolicy(
        metric_name="score",
        direction=(
            MetricDirection.MAXIMIZE
        ),
        minimum_improvement=0.01,
        allow_initial_champion=True,
    )


    lifecycle = finalize_model_candidate(
        training_result=training,
        evaluation_result=evaluation,
        promotion_policy=policy,
        config=config,
        artifact_path="model",
        logged_model_uri=(
            logged_artifact.model_uri
        ),
    )

    assert lifecycle.registration.registered
    assert lifecycle.promotion is not None
    assert lifecycle.promotion.decision.promote
    assert (
        lifecycle.promotion.champion_assignment
        is not None
    )

    release = publish_serving_release(
        models_path=str(models_path),
        registration=(
            lifecycle.registration
        ),
        promotion=lifecycle.promotion,
        task_type=TASK_TYPE,
        model_type="pyfunc",
        sources=build_release_sources(
            tmp_path
        ),
        metadata=RELEASE_METADATA,
        dataset_version="integration-dataset",
        git_commit="integration-test",
        release_id="release-integration",
    )

    assert (
        release.active_pointer.release_id
        == "release-integration"
    )

    active_manifest, release_root = (
        load_active_release_manifest(
            models_path=str(models_path)
        )
    )

    assert (
        active_manifest.release_id
        == "release-integration"
    )
    assert (
        active_manifest.model.version
        == lifecycle.registration.model_version
    )
    assert (
        active_manifest.model.uri
        == lifecycle.registration.model_uri
    )
    assert release_root == release.release_root

    champion = (
        client.get_model_version_by_alias(
            name=model_name,
            alias="champion",
        )
    )

    assert (
        str(champion.version)
        == lifecycle.registration.model_version
    )

    loaded_model = mlflow.pyfunc.load_model(
        lifecycle.registration.model_uri
    )
    predictions = loaded_model.predict(
        pd.DataFrame(
            {
                "feature": [1.0, 2.0],
            }
        )
    )

    assert list(predictions) == [1, 1]