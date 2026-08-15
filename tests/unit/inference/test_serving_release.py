import json

import pandas as pd
import pytest

from src.inference.serving_release import (
    load_active_serving_manifest,
    publish_serving_release,
    resolve_release_artifact_uri,
    activate_release_pointer,
    load_active_release_id,
)


@pytest.fixture
def serving_sources(tmp_path):
    models_path = tmp_path / "models"
    models_path.mkdir()

    state_path = (
        tmp_path / "latest_state.json"
    )
    state_path.write_text(
        json.dumps(
            {
                "1": [
                    100.0,
                    110.0,
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata_path = (
        tmp_path / "store.parquet"
    )
    pd.DataFrame(
        {
            "Store": [1],
            "Promo2": [0],
        }
    ).to_parquet(
        metadata_path,
        index=False,
    )

    calendar_path = (
        tmp_path
        / "known_calendar.parquet"
    )
    pd.DataFrame(
        {
            "Store": [1],
            "Date": [
                pd.Timestamp("2026-08-12")
            ],
        }
    ).to_parquet(
        calendar_path,
        index=False,
    )

    return {
        "models_path": str(
            models_path
        ),
        "state": str(state_path),
        "metadata": str(metadata_path),
        "calendar": str(calendar_path),
    }

def publish_test_release(
    serving_sources: dict[str, str],
):
    return publish_serving_release(
        models_path=(
            serving_sources["models_path"]
        ),
        model_name="forecast-model",
        model_version="8",
        model_run_id="run-8",
        model_type="xgboost",
        target_transformation="log1p",
        dataset_version="dataset-1",
        config_hash="config-hash",
        git_commit="abc123",
        store_metadata_source=(
            serving_sources["metadata"]
        ),
        store_state_source=(
            serving_sources["state"]
        ),
        known_calendar_source=(
            serving_sources["calendar"]
        ),
        prediction_probe_payload={
            "inputs": [
                {
                    "Store": 1,
                    "DayOfWeek": 5,
                    "Date": "2015-04-24",
                    "Customers": 500,
                    "Open": 1,
                    "Promo": 1,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                }
            ],
            "context": {
                "purpose": (
                    "post_deployment_verification"
                ),
            },
        },
    )

def test_publish_and_load_active_serving_release(
    serving_sources,
):
    published = publish_test_release(
        serving_sources
    )

    loaded, release_root = (
        load_active_serving_manifest(
            models_path=(
                serving_sources[
                    "models_path"
                ]
            ),
        )
    )

    assert (
        loaded.release_id
        == published.release_id
    )
    assert loaded.model_version == "8"
    assert loaded.model_uri == (
        "models:/forecast-model/8"
    )
    
    assert published.schema_version == 2
    assert loaded.schema_version == 2

    assert (
        loaded.prediction_probe
        is not None
    )
    assert (
        loaded.prediction_probe.path
        == "prediction_probe.json"
    )

    probe_uri = (
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=(
                loaded.prediction_probe
            ),
        )
    )

    with open(
        probe_uri,
        encoding="utf-8",
    ) as file_handle:
        probe_payload = json.load(
            file_handle
        )

    assert probe_payload["inputs"] == [
        {
            "Store": 1,
            "DayOfWeek": 5,
            "Date": "2015-04-24",
            "Customers": 500,
            "Open": 1,
            "Promo": 1,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }
    ]

    assert (
        probe_payload["context"]["purpose"]
        == "post_deployment_verification"
    )
    

    metadata_uri = (
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=(
                loaded.store_metadata
            ),
        )
    )

    assert metadata_uri.endswith(
        "store.parquet"
    )

def test_modified_artifact_fails_checksum_validation(
    serving_sources,
):
    publish_test_release(
        serving_sources
    )

    manifest, release_root = (
        load_active_serving_manifest(
            models_path=(
                serving_sources[
                    "models_path"
                ]
            ),
        )
    )

    state_uri = (
        f"{release_root}/"
        f"{manifest.store_state.path}"
    )

    with open(
        state_uri,
        "w",
        encoding="utf-8",
    ) as file_handle:
        file_handle.write(
            '{"tampered": true}'
        )

    with pytest.raises(
        ValueError,
        match="checksum mismatch",
    ):
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=(
                manifest.store_state
            ),
        )

def test_failed_publication_keeps_active_release(
    serving_sources,
):
    first_release = publish_test_release(
        serving_sources
    )

    with pytest.raises(
        FileNotFoundError
    ):
        publish_serving_release(
            models_path=(
                serving_sources[
                    "models_path"
                ]
            ),
            model_name="forecast-model",
            model_version="9",
            model_run_id="run-9",
            model_type="xgboost",
            target_transformation="log1p",
            dataset_version="dataset-2",
            config_hash="new-hash",
            git_commit="def456",
            store_metadata_source=(
                serving_sources["metadata"]
            ),
            store_state_source=(
                "missing-state.json"
            ),
            known_calendar_source=(
                serving_sources["calendar"]
            ),
            prediction_probe_payload={
                "inputs": [
                    {
                        "Store": 1,
                        "Date": "2015-04-24",
                    }
                ]
            },
        )

    active, _ = (
        load_active_serving_manifest(
            models_path=(
                serving_sources[
                    "models_path"
                ]
            ),
        )
    )

    assert (
        active.release_id
        == first_release.release_id
    )

def test_release_artifact_cannot_escape_release_prefix(
    tmp_path,
):
    from src.inference.serving_bundle import (
        ServingArtifactReference,
    )

    reference = ServingArtifactReference(
        path="../../secret.txt",
        sha256="irrelevant",
    )

    with pytest.raises(
        ValueError,
        match="relative and contained",
    ):
        resolve_release_artifact_uri(
            release_root=str(tmp_path),
            reference=reference,
        )


def test_activate_release_pointer_changes_active_release(
    serving_sources,
):
    first_release = publish_test_release(
        serving_sources
    )

    second_release = publish_serving_release(
        models_path=serving_sources[
            "models_path"
        ],
        model_name="forecast-model",
        model_version="9",
        model_run_id="run-9",
        model_type="xgboost",
        target_transformation="log1p",
        dataset_version="dataset-2",
        config_hash="hash-2",
        git_commit="def456",
        store_metadata_source=(
            serving_sources["metadata"]
        ),
        store_state_source=(
            serving_sources["state"]
        ),
        known_calendar_source=(
            serving_sources["calendar"]
        ),
        prediction_probe_payload={
            "inputs": [
                {
                    "Store": 1,
                    "Date": "2015-04-24",
                }
            ]
        },
    )

    activate_release_pointer(
        models_path=serving_sources[
            "models_path"
        ],
        release_id=first_release.release_id,
        operation="rollback",
        previous_release_id=(
            second_release.release_id
        ),
    )

    assert load_active_release_id(
        models_path=serving_sources[
            "models_path"
        ],
    ) == first_release.release_id


def test_activate_release_pointer_rejects_missing_release(
    serving_sources,
):
    with pytest.raises(
        FileNotFoundError,
        match="without manifest",
    ):
        activate_release_pointer(
            models_path=serving_sources[
                "models_path"
            ],
            release_id="missing-release",
            operation="rollback",
        )