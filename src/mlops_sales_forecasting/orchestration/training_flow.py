from prefect import flow

from ..configs.loader import load_config
from ..pipeline.project_factory import (
    build_project_training_pipeline,
)
from .lifecycle_adapter import (
    PrefectTrainingLifecycleResult,
    run_prefect_model_lifecycle,
)


@flow(
    name="mlops-sales-forecasting-training",
    validate_parameters=False,
    persist_result=False,
)
def training_flow(
    run_id: str | None = None,
) -> PrefectTrainingLifecycleResult:
    """Run the complete forecasting model lifecycle."""

    config = load_config()
    pipeline = build_project_training_pipeline(
        config
    )

    return run_prefect_model_lifecycle(
        pipeline=pipeline,
        pipeline_run_id=run_id,
    )


if __name__ == "__main__":
    training_flow()