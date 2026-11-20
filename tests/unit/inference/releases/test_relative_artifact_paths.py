import pytest

from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    validate_artifact_reference,
)
from mlops_sales_forecasting.inference.releases.storage import build_release_paths


@pytest.mark.parametrize(
    "path",
    [
        "gs://bucket/model.pkl",
        "s3://bucket/model.pkl",
        "https://example.com/model.pkl",
        "/model.pkl",
        "../model.pkl",
    ],
)
def test_artifact_reference_rejects_non_relative_paths(path: str) -> None:
    reference = ArtifactReference(path=path, sha256="a" * 64)

    with pytest.raises(ValueError, match="must be relative"):
        validate_artifact_reference(reference, name="model")


@pytest.mark.parametrize(
    "path",
    [
        "gs://bucket/model.pkl",
        "s3://bucket/model.pkl",
        "https://example.com/model.pkl",
        "/model.pkl",
        "../model.pkl",
    ],
)
def test_release_builder_rejects_non_relative_paths(path: str) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        build_release_paths(
            models_path="s3://bucket/models",
            release_id="release-test",
            artifact_filenames={"model": path},
        )
