from unittest.mock import MagicMock

import pandas as pd

from mlops_sales_forecasting.inference import bundle_loader
from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)

VALID_CHECKSUM = "f" * 64


def artifact(path: str) -> ArtifactReference:
    return ArtifactReference(
        path=path,
        sha256=VALID_CHECKSUM,
    )



def build_manifest() -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-1",
        created_at_utc="2026-09-14T08:00:00+00:00",
        task_type=TaskType.FORECASTING,
        model=ModelReference(
            name="forecasting-model",
            version="1",
            run_id="run-1",
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


def test_build_serving_bundle(monkeypatch) -> None:
    manifest = build_manifest()
    model = MagicMock()
    model_loader = MagicMock(return_value=model)

    resolved_paths = {
        "store.parquet": "/release/store.parquet",
        "latest_state.json": "/release/latest_state.json",
        "known_calendar.parquet": "/release/known_calendar.parquet",
    }

    monkeypatch.setattr(
        bundle_loader,
        "resolve_artifact_uri",
        lambda *, reference, **_: resolved_paths[reference.path],
    )
    monkeypatch.setattr(
        bundle_loader,
        "load_json",
        lambda _: {
            "1": {
                "last_observation": 100.0,
            }
        },
    )

    store_metadata = pd.DataFrame(
        {"entity_id": [1]}
    )
    known_calendar = pd.DataFrame(
        {
            "entity_id": [1],
            "date": ["2026-09-15"],
        }
    )

    def fake_read_parquet(path: str) -> pd.DataFrame:
        if path.endswith("store.parquet"):
            return store_metadata

        return known_calendar

    monkeypatch.setattr(
        bundle_loader.pd,
        "read_parquet",
        fake_read_parquet,
    )

    result = bundle_loader.build_serving_bundle(
        manifest=manifest,
        release_root="/release",
        serving_alias="champion",
        model_loader=model_loader,
    )

    assert result.model is model
    assert result.target_transformation == "log1p"
    assert result.store_metadata.equals(store_metadata)
    assert result.known_calendar.equals(known_calendar)
    model_loader.assert_called_once_with(
        "models:/forecasting-model@champion"
    )



def test_load_serving_bundle_uses_requested_release(
    monkeypatch,
) -> None:
    manifest = build_manifest()

    monkeypatch.setattr(
        bundle_loader,
        "load_release_manifest",
        lambda **_: (manifest, "/release"),
    )

    expected_bundle = MagicMock()
    build_bundle = MagicMock(return_value=expected_bundle)
    monkeypatch.setattr(
        bundle_loader,
        "build_serving_bundle",
        build_bundle,
    )

    result = bundle_loader.load_serving_bundle(
        models_path="/models",
        release_id="release-1",
        serving_alias="champion",
        model_loader=MagicMock(),
    )

    assert result is expected_bundle
    assert build_bundle.call_args.kwargs["manifest"] is manifest
    assert (
        build_bundle.call_args.kwargs["release_root"]
        == "/release"
    )


def test_load_active_serving_bundle_uses_active_manifest(
    monkeypatch,
) -> None:
    manifest = build_manifest()

    monkeypatch.setattr(
        bundle_loader,
        "load_active_release_manifest",
        lambda **_: (manifest, "/active-release"),
    )

    expected_bundle = MagicMock()
    build_bundle = MagicMock(return_value=expected_bundle)
    monkeypatch.setattr(
        bundle_loader,
        "build_serving_bundle",
        build_bundle,
    )

    result = bundle_loader.load_active_serving_bundle(
        models_path="/models",
        serving_alias="champion",
        model_loader=MagicMock(),
    )

    assert result is expected_bundle
    assert (
        build_bundle.call_args.kwargs["release_root"]
        == "/active-release"
    )