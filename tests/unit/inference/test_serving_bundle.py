from dataclasses import replace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_sales_forecasting.inference.serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)

VALID_CHECKSUM = "e" * 64


def artifact(path: str) -> ArtifactReference:
    return ArtifactReference(
        path=path,
        sha256=VALID_CHECKSUM,
    )



def build_valid_manifest() -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-1",
        created_at_utc="2026-09-14T08:00:00+00:00",
        task_type=TaskType.FORECASTING,
        model=ModelReference(
            name="forecasting-model",
            version="3",
            run_id="run-3",
            uri="models:/forecasting-model@champion",
            model_type="xgboost",
        ),
        artifacts={
            "store_metadata": artifact("store.parquet"),
            "store_state": artifact("latest_state.json"),
            "known_calendar": artifact("known_calendar.parquet"),
        },
        metadata={
            "target_transformation": "log1p",
        },
    )


def build_valid_bundle() -> ServingBundle:
    return ServingBundle(
        release_id="release-1",
        manifest=build_valid_manifest(),
        model=MagicMock(),
        serving_alias="champion",
        target_transformation="log1p",
        store_metadata=pd.DataFrame(
            {"entity_id": [1, 2]}
        ),
        store_state={
            "1": {
                "last_observation": 100.0,
            }
        },
        known_calendar=pd.DataFrame(
            {
                "entity_id": [1],
                "date": ["2026-09-15"],
            }
        ),
    )



def test_validate_serving_bundle() -> None:
    validate_serving_bundle(build_valid_bundle())


def test_bundle_exposes_model_properties() -> None:
    bundle = build_valid_bundle()

    assert bundle.model_name == bundle.manifest.model.name
    assert bundle.model_version == "3"
    assert bundle.model_run_id == "run-3"
    assert bundle.model_uri.endswith("@champion")
    assert bundle.model_type == "xgboost"


def test_bundle_requires_model() -> None:
    bundle = replace(
        build_valid_bundle(),
        model=None,
    )

    with pytest.raises(
        ValueError,
        match="has no model",
    ):
        validate_serving_bundle(bundle)


def test_bundle_requires_matching_release_id() -> None:
    bundle = replace(
        build_valid_bundle(),
        release_id="another-release",
    )

    with pytest.raises(
        ValueError,
        match="does not match manifest",
    ):
        validate_serving_bundle(bundle)


def test_bundle_requires_serving_alias() -> None:
    bundle = replace(
        build_valid_bundle(),
        serving_alias="",
    )

    with pytest.raises(
        ValueError,
        match="has no serving alias",
    ):
        validate_serving_bundle(bundle)



def test_bundle_requires_matching_target_transformation() -> None:
    bundle = replace(
        build_valid_bundle(),
        target_transformation="identity",
    )

    with pytest.raises(
        ValueError,
        match="does not match manifest",
    ):
        validate_serving_bundle(bundle)


def test_bundle_requires_store_metadata() -> None:
    bundle = replace(
        build_valid_bundle(),
        store_metadata=pd.DataFrame(),
    )

    with pytest.raises(
        ValueError,
        match="has no store metadata",
    ):
        validate_serving_bundle(bundle)


def test_bundle_requires_forecasting_state() -> None:
    bundle = replace(
        build_valid_bundle(),
        store_state={},
    )

    with pytest.raises(
        ValueError,
        match="invalid forecasting state",
    ):
        validate_serving_bundle(bundle)


def test_bundle_requires_known_calendar() -> None:
    bundle = replace(
        build_valid_bundle(),
        known_calendar=pd.DataFrame(),
    )

    with pytest.raises(
        ValueError,
        match="has no known calendar",
    ):
        validate_serving_bundle(bundle)
