from unittest.mock import MagicMock

from mlops_sales_forecasting.pipeline import (
    factory,
    service,
)
from mlops_sales_forecasting.pipeline.repository import (
    PipelineRunRepository,
)
from mlops_sales_forecasting.pipeline.service import (
    TrainingPipeline,
)


def build_pipeline(
    repository: PipelineRunRepository
    | MagicMock,
) -> TrainingPipeline:
    return TrainingPipeline(
        ingestor=MagicMock(),
        feature_builder=MagicMock(),
        splitter=MagicMock(),
        trainer=MagicMock(),
        evaluator=MagicMock(),
        model_logger=MagicMock(),
        release_input_provider=MagicMock(),
        config={
            "environment": "test",
        },
        run_repository=repository,
    )


def test_service_executes_tracked_pipeline(
    monkeypatch,
) -> None:
    repository = MagicMock(
        spec=PipelineRunRepository
    )
    pipeline = build_pipeline(repository)
    expected_result = MagicMock()

    execute = MagicMock(
        return_value=expected_result
    )
    monkeypatch.setattr(
        service,
        "run_tracked_training_pipeline",
        execute,
    )

    result = pipeline.run(
        run_id="pipeline-run-123"
    )

    assert result is expected_result

    call = execute.call_args

    assert call.kwargs["ingestor"] is (
        pipeline.ingestor
    )
    assert call.kwargs["feature_builder"] is (
        pipeline.feature_builder
    )
    assert call.kwargs["splitter"] is (
        pipeline.splitter
    )
    assert call.kwargs["trainer"] is (
        pipeline.trainer
    )
    assert call.kwargs["evaluator"] is (
        pipeline.evaluator
    )
    assert call.kwargs["config"] is (
        pipeline.config
    )
    assert call.kwargs["run_id"] == (
        "pipeline-run-123"
    )

    observer = call.kwargs["observer"]
    pipeline_run = MagicMock()

    observer(pipeline_run)

    repository.save.assert_called_once_with(
        pipeline_run
    )


def test_service_allows_generated_run_id(
    monkeypatch,
) -> None:
    repository = MagicMock(
        spec=PipelineRunRepository
    )
    pipeline = build_pipeline(repository)

    execute = MagicMock()
    monkeypatch.setattr(
        service,
        "run_tracked_training_pipeline",
        execute,
    )

    pipeline.run()

    assert execute.call_args.kwargs[
        "run_id"
    ] is None


def test_factory_builds_configured_service(
    monkeypatch,
) -> None:
    repository = MagicMock(
        spec=PipelineRunRepository
    )
    build_repository = MagicMock(
        return_value=repository
    )
    monkeypatch.setattr(
        factory,
        "build_pipeline_run_repository",
        build_repository,
    )

    ingestor = MagicMock()
    feature_builder = MagicMock()
    splitter = MagicMock()
    trainer = MagicMock()
    evaluator = MagicMock()
    model_logger = MagicMock()
    release_input_provider = MagicMock()
    config = {
        "paths": {
            "artifacts": "artifacts",
        },
    }

    pipeline = factory.build_training_pipeline(
        ingestor=ingestor,
        feature_builder=feature_builder,
        splitter=splitter,
        trainer=trainer,
        evaluator=evaluator,
        model_logger=model_logger,
        release_input_provider=(
            release_input_provider
        ),
        config=config,
    )

    assert isinstance(
        pipeline,
        TrainingPipeline,
    )
    assert pipeline.ingestor is ingestor
    assert pipeline.feature_builder is (
        feature_builder
    )
    assert pipeline.splitter is splitter
    assert pipeline.trainer is trainer
    assert pipeline.evaluator is evaluator
    assert pipeline.config is config
    assert pipeline.run_repository is repository
    assert pipeline.model_logger is model_logger
    assert (
        pipeline.release_input_provider
        is release_input_provider
    )

    build_repository.assert_called_once_with(
        config
    )


def test_factory_uses_real_repository() -> None:
    pipeline = factory.build_training_pipeline(
        ingestor=MagicMock(),
        feature_builder=MagicMock(),
        splitter=MagicMock(),
        trainer=MagicMock(),
        evaluator=MagicMock(),
        model_logger=MagicMock(),
        release_input_provider=MagicMock(),
        config={
            "paths": {
                "artifacts": "artifacts",
            },
        },
    )

    assert isinstance(
        pipeline.run_repository,
        PipelineRunRepository,
    )
    assert pipeline.run_repository.root_path == (
        "artifacts/pipeline-runs"
    )