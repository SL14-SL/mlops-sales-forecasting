import pytest

from mlops_sales_forecasting.pipeline.project_factory import (
    build_project_training_pipeline,
)


def test_project_pipeline_marks_extension_point() -> None:
    with pytest.raises(
        NotImplementedError,
        match=(
            "project-specific forecasting "
            "training pipeline"
        ),
    ):
        build_project_training_pipeline(
            {
                "project": {
                    "task_type": "forecasting",
                },
            }
        )