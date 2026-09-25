import pytest

from mlops_sales_forecasting.pipeline.factory import (
    build_pipeline_run_repository,
)


def test_factory_uses_artifacts_fallback() -> None:
    repository = build_pipeline_run_repository(
        {
            "paths": {
                "artifacts": "artifacts",
            },
        }
    )

    assert repository.root_path == (
        "artifacts/pipeline-runs"
    )


def test_factory_supports_gcs_artifacts() -> None:
    repository = build_pipeline_run_repository(
        {
            "paths": {
                "artifacts": (
                    "gs://example-bucket/artifacts"
                ),
            },
        }
    )

    assert repository.root_path == (
        "gs://example-bucket/"
        "artifacts/pipeline-runs"
    )


def test_explicit_pipeline_path_overrides_fallback() -> None:
    repository = build_pipeline_run_repository(
        {
            "paths": {
                "artifacts": "artifacts",
                "pipeline_runs": (
                    "custom/pipeline-history"
                ),
            },
        }
    )

    assert repository.root_path == (
        "custom/pipeline-history"
    )


def test_explicit_gcs_pipeline_path_is_supported() -> None:
    repository = build_pipeline_run_repository(
        {
            "paths": {
                "artifacts": "artifacts",
                "pipeline_runs": (
                    "gs://example-bucket/"
                    "pipeline-history"
                ),
            },
        }
    )

    assert repository.root_path == (
        "gs://example-bucket/pipeline-history"
    )


def test_trailing_slash_is_removed() -> None:
    repository = build_pipeline_run_repository(
        {
            "paths": {
                "artifacts": "artifacts/",
            },
        }
    )

    assert repository.root_path == (
        "artifacts/pipeline-runs"
    )


def test_factory_requires_paths_section() -> None:
    with pytest.raises(
        ValueError,
        match="'paths' section",
    ):
        build_pipeline_run_repository({})


@pytest.mark.parametrize(
    "paths",
    [
        None,
        [],
        "artifacts",
    ],
)
def test_paths_section_must_be_mapping(
    paths: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="'paths' section",
    ):
        build_pipeline_run_repository(
            {
                "paths": paths,
            }
        )


@pytest.mark.parametrize(
    "artifacts_path",
    [
        "",
        "   ",
        7,
        None,
        "${ARTIFACTS_PATH}",
    ],
)
def test_fallback_requires_valid_artifacts_path(
    artifacts_path: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="'artifacts'",
    ):
        build_pipeline_run_repository(
            {
                "paths": {
                    "artifacts": artifacts_path,
                },
            }
        )


@pytest.mark.parametrize(
    "pipeline_path",
    [
        "",
        "   ",
        7,
        "${PIPELINE_RUNS_PATH}",
    ],
)
def test_explicit_pipeline_path_must_be_valid(
    pipeline_path: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="'pipeline_runs'",
    ):
        build_pipeline_run_repository(
            {
                "paths": {
                    "artifacts": "artifacts",
                    "pipeline_runs": pipeline_path,
                },
            }
        )