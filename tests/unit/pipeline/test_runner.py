from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.data.contracts import (
    DatasetCollection,
    DatasetSplits,
)
from mlops_sales_forecasting.pipeline.runner import (
    PipelineResult,
    run_training_pipeline,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


def build_datasets() -> DatasetCollection:
    return DatasetCollection(
        datasets={
            "observations": pd.DataFrame(
                {
                    "feature": [1, 2, 3],
                    "target": [0, 1, 0],
                }
            ),
        }
    )


def build_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )


def build_splits() -> DatasetSplits:
    return DatasetSplits(
        train=pd.DataFrame(
            {
                "feature": [1, 2],
                "target": [0, 1],
            }
        ),
        validation=pd.DataFrame(
            {
                "feature": [3],
                "target": [0],
            }
        ),
    )


def build_training_result() -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id="run-123",
        metrics={
            "training_score": 0.8,
        },
    )


def build_evaluation_result(
    *,
    approved: bool = True,
) -> EvaluationResult:
    return EvaluationResult(
        metrics={
            "validation_score": 0.75,
        },
        approved=approved,
        reasons=(
            ()
            if approved
            else (
                "Validation score is below threshold.",
            )
        ),
    )


def build_components() -> tuple[
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    ingestor = MagicMock()
    ingestor.ingest.return_value = build_datasets()

    feature_builder = MagicMock()
    feature_builder.build_features.return_value = (
        build_features()
    )

    splitter = MagicMock()
    splitter.split.return_value = build_splits()

    trainer = MagicMock()
    trainer.train.return_value = (
        build_training_result()
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = (
        build_evaluation_result()
    )

    return (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    )


def test_pipeline_executes_all_steps() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    config = {
        "environment": "test",
    }

    result = run_training_pipeline(
        ingestor=ingestor,
        feature_builder=feature_builder,
        splitter=splitter,
        trainer=trainer,
        evaluator=evaluator,
        config=config,
    )

    assert isinstance(result, PipelineResult)
    assert result.datasets is (
        ingestor.ingest.return_value
    )
    assert result.features is (
        feature_builder.build_features.return_value
    )
    assert result.splits is (
        splitter.split.return_value
    )
    assert result.training is (
        trainer.train.return_value
    )
    assert result.evaluation is (
        evaluator.evaluate.return_value
    )

    ingestor.ingest.assert_called_once_with(config)
    feature_builder.build_features.assert_called_once_with(
        result.datasets,
        config,
    )
    splitter.split.assert_called_once_with(
        result.features,
        config,
    )
    trainer.train.assert_called_once_with(
        result.splits,
        config,
    )
    evaluator.evaluate.assert_called_once_with(
        result.training,
        result.splits,
        config,
    )


def test_pipeline_preserves_rejected_evaluation() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    evaluator.evaluate.return_value = (
        build_evaluation_result(
            approved=False
        )
    )

    result = run_training_pipeline(
        ingestor=ingestor,
        feature_builder=feature_builder,
        splitter=splitter,
        trainer=trainer,
        evaluator=evaluator,
        config={},
    )

    assert result.evaluation.approved is False
    assert result.evaluation.reasons == (
        "Validation score is below threshold.",
    )


def test_pipeline_rejects_invalid_ingestion_result() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    ingestor.ingest.return_value = pd.DataFrame()

    with pytest.raises(
        TypeError,
        match=(
            "Data ingestor must return "
            "DatasetCollection"
        ),
    ):
        run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config={},
        )

    feature_builder.build_features.assert_not_called()
    splitter.split.assert_not_called()
    trainer.train.assert_not_called()
    evaluator.evaluate.assert_not_called()


@pytest.mark.parametrize(
    "features",
    [
        [],
        {},
        "invalid",
    ],
)
def test_pipeline_rejects_non_dataframe_features(
    features: object,
) -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    feature_builder.build_features.return_value = (
        features
    )

    with pytest.raises(
        TypeError,
        match=(
            "Feature builder must return "
            "a pandas DataFrame"
        ),
    ):
        run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config={},
        )

    splitter.split.assert_not_called()
    trainer.train.assert_not_called()
    evaluator.evaluate.assert_not_called()


def test_pipeline_rejects_empty_features() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    feature_builder.build_features.return_value = (
        pd.DataFrame()
    )

    with pytest.raises(
        ValueError,
        match="empty DataFrame",
    ):
        run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config={},
        )

    splitter.split.assert_not_called()
    trainer.train.assert_not_called()
    evaluator.evaluate.assert_not_called()


def test_pipeline_rejects_invalid_split_result() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    splitter.split.return_value = {
        "train": build_features(),
    }

    with pytest.raises(
        TypeError,
        match=(
            "Dataset splitter must return "
            "DatasetSplits"
        ),
    ):
        run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config={},
        )

    trainer.train.assert_not_called()
    evaluator.evaluate.assert_not_called()


def test_pipeline_rejects_invalid_training_result() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    trainer.train.return_value = object()

    with pytest.raises(
        TypeError,
        match=(
            "Model trainer must return "
            "TrainingResult"
        ),
    ):
        run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config={},
        )

    evaluator.evaluate.assert_not_called()


def test_pipeline_rejects_invalid_evaluation_result() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    evaluator.evaluate.return_value = object()

    with pytest.raises(
        TypeError,
        match=(
            "Model evaluator must return "
            "EvaluationResult"
        ),
    ):
        run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config={},
        )


def test_pipeline_propagates_step_errors() -> None:
    (
        ingestor,
        feature_builder,
        splitter,
        trainer,
        evaluator,
    ) = build_components()
    trainer.train.side_effect = RuntimeError(
        "Training failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Training failed",
    ):
        run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config={},
        )

    evaluator.evaluate.assert_not_called()