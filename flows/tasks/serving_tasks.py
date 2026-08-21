import requests
import os

from src.configs.loader import load_config, get_path
from src.configs.paths import join_uri

ENV_CFG = load_config()
# ruff: noqa: E402

from prefect import task, get_run_logger
import mlflow
from mlflow.tracking import MlflowClient

from src.deployment.verification import verify_serving_release, verify_prediction_probe
from src.deployment.prediction_probe import build_prediction_probe

from src.inference.serving_release import (
    publish_serving_release, 
    load_serving_release_manifest, 
    load_active_release_id,
    load_release_prediction_probe,
)

GCP_CFG = load_config("gcp.yaml")
MODEL_NAME = ENV_CFG["model"]["registry_name"]

@task(name="Resolve Previous Serving Release")
def task_resolve_previous_release() -> str | None:
    """
    Resolve the currently active serving release before publishing a new one.

    During bootstrap no previous release exists.
    """
    p_logger = get_run_logger()

    try:
        release_id = load_active_release_id(
            models_path=get_path("models"),
        )

    except FileNotFoundError:
        p_logger.info(
            "No previous serving release exists. "
            "This is expected during bootstrap."
        )
        return None

    p_logger.info(
        "Previous serving release resolved | "
        f"release_id={release_id}"
    )

    return release_id


@task(name="Rollback Serving Release")
def task_rollback_serving_release(
    *,
    previous_release_id: str,
) -> dict:
    """
    Roll back both the serving release and the MLflow Champion alias.

    The serving rollback endpoint validates and activates the previous
    immutable release. Afterwards the MLflow Champion alias is restored to
    the exact model version referenced by that release.
    """
    p_logger = get_run_logger()

    cfg = load_config()
    api_url = cfg.get(
        "api",
        {},
    ).get(
        "url",
        "http://api:8080/predict",
    )

    if api_url.endswith("/predict"):
        api_base_url = api_url.removesuffix(
            "/predict"
        )
    else:
        api_base_url = api_url.rstrip("/")

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable is not set."
        )

    p_logger.warning(
        "Starting automatic serving rollback | "
        f"target_release_id={previous_release_id}"
    )

    response = requests.post(
        (
            f"{api_base_url}"
            "/admin/rollback-serving-release"
        ),
        json={
            "release_id": previous_release_id,
        },
        headers={
            "X-API-KEY": api_key,
        },
        timeout=30,
    )

    response.raise_for_status()
    rollback_result = response.json()

    previous_manifest = (
        load_serving_release_manifest(
            models_path=get_path("models"),
            release_id=previous_release_id,
        )
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=previous_manifest.model_name,
        alias="champion",
        version=str(
            previous_manifest.model_version
        ),
    )

    p_logger.warning(
        "Automatic rollback completed | "
        f"release_id={previous_release_id} | "
        "model_version="
        f"{previous_manifest.model_version}"
    )

    return {
        "release_id": previous_release_id,
        "model_name": (
            previous_manifest.model_name
        ),
        "model_version": str(
            previous_manifest.model_version
        ),
        "model_run_id": (
            previous_manifest.model_run_id
        ),
        "api_result": rollback_result,
    }


@task(name="Publish Serving Release")
def task_publish_serving_release(
    *,
    final_run_id: str,
    model_version: str,
    dataset_manifest: dict,
) -> str:
    """
    Publish the promoted model and its inference artifacts as one immutable
    serving release.

    Works with local paths and gs:// paths.
    """
    p_logger = get_run_logger()

    client = mlflow.MlflowClient()
    run = client.get_run(
        final_run_id
    )

    model_type = (
        run.data.tags.get("model_type")
        or run.data.params.get("model_type")
        or "xgboost"
    )

    target_transformation = (
        run.data.tags.get(
            "target_transformation"
        )
        or run.data.params.get(
            "target_transformation"
        )
        or "none"
    )

    config_hash = run.data.params.get(
        "config_hash"
    )

    snapshots = dataset_manifest.get(
        "snapshots",
        {},
    )

    prediction_probe_source = (
        snapshots.get(
            "validated_train"
        )
    )

    if not prediction_probe_source:
        prediction_probe_source = join_uri(
            get_path("validated_data"),
            "train.parquet",
        )

    prediction_probe_payload = (
        build_prediction_probe(
            validated_data_path=(
                prediction_probe_source
            ),
        )
    )

    # Prefer the versioned dataset snapshot. This prevents the release from
    # reading store metadata that changed after this training run.
    store_metadata_source = snapshots.get(
        "validated_store"
    )

    if not store_metadata_source:
        store_metadata_source = join_uri(
            get_path("validated_data"),
            "store.parquet",
        )

    store_state_source = join_uri(
        get_path("models"),
        "latest_state.json",
    )

    known_calendar_source = join_uri(
        get_path("features"),
        "known_calendar.parquet",
    )

    manifest = publish_serving_release(
        models_path=get_path("models"),
        model_name=MODEL_NAME,
        model_version=model_version,
        model_run_id=final_run_id,
        model_type=model_type,
        target_transformation=(
            target_transformation
        ),
        dataset_version=(
            dataset_manifest.get(
                "dataset_version"
            )
        ),
        config_hash=config_hash,
        git_commit=(
            dataset_manifest.get(
                "git_commit"
            )
            or os.getenv(
                "GIT_COMMIT_SHA"
            )
        ),
        store_metadata_source=(
            store_metadata_source
        ),
        store_state_source=(
            store_state_source
        ),
        known_calendar_source=(
            known_calendar_source
        ),
        prediction_probe_payload=(
            prediction_probe_payload
        ),
    )

    p_logger.info(
        "Serving release published | "
        f"release_id={manifest.release_id} | "
        f"model_version={model_version} | "
        f"dataset_version={manifest.dataset_version}"
    )

    return manifest.release_id



@task(name="Refresh API")
def task_refresh_api() -> None:
    """
    Refresh the forecasting API serving state after training.

    Reloads:
    - current champion model
    - store metadata
    - forecasting state snapshot
    """
    p_logger = get_run_logger()
    cfg = load_config()

    api_url = cfg.get("api", {}).get("url", "http://api:8080/predict")

    if api_url.endswith("/predict"):
        base_url = api_url.removesuffix("/predict")
    else:
        base_url = api_url.rstrip("/")

    reload_url = f"{base_url}/admin/reload-serving-state"

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is not set.")

    p_logger.info(f"Refreshing API serving state via: {reload_url}")

    response = requests.post(
        reload_url,
        headers={"X-API-KEY": api_key},
        timeout=300,
    )

    response.raise_for_status()
    p_logger.info(f"API serving state reload successful: {response.json()}")

    
@task(name="Verify Serving Release")
def task_verify_serving_release(
    expected_release_id: str,
) -> dict:
    """
    Verify readiness, release lineage and semantic prediction behavior.
    """
    p_logger = get_run_logger()

    api_url = ENV_CFG.get(
        "api",
        {},
    ).get(
        "url",
        "http://api:8080/predict",
    )

    if api_url.endswith("/predict"):
        api_base_url = (
            api_url.removesuffix(
                "/predict"
            )
        )
    else:
        api_base_url = (
            api_url.rstrip("/")
        )

    p_logger.info(
        "Verifying serving release | "
        f"expected_release_id="
        f"{expected_release_id} | "
        f"api_base_url={api_base_url}"
    )

    readiness_result = (
        verify_serving_release(
            api_base_url=api_base_url,
            expected_release_id=(
                expected_release_id
            ),
        )
    )

    manifest = load_serving_release_manifest(
        models_path=get_path("models"),
        release_id=expected_release_id,
    )

    if (
        str(readiness_result.model_version)
        != str(manifest.model_version)
    ):
        raise RuntimeError(
            "Ready endpoint model version does "
            "not match release manifest | "
            f"ready={readiness_result.model_version} | "
            f"manifest={manifest.model_version}"
        )

    if (
        readiness_result.model_run_id
        != manifest.model_run_id
    ):
        raise RuntimeError(
            "Ready endpoint model run ID does "
            "not match release manifest | "
            f"ready={readiness_result.model_run_id} | "
            f"manifest={manifest.model_run_id}"
        )

    prediction_probe_payload = (
        load_release_prediction_probe(
            models_path=get_path("models"),
            release_id=expected_release_id,
        )
    )

    # Backward-compatible rollback verification for schema-v1 releases.
    if prediction_probe_payload is None:
        p_logger.warning(
            "Semantic prediction verification "
            "skipped for legacy release | "
            f"release_id={expected_release_id} | "
            f"schema_version="
            f"{manifest.schema_version}"
        )

        return {
            "release_id": (
                readiness_result.release_id
            ),
            "model_version": (
                readiness_result.model_version
            ),
            "model_run_id": (
                readiness_result.model_run_id
            ),
            "readiness_attempts": (
                readiness_result.attempts
            ),
            "prediction_probe_status": (
                "skipped_legacy_release"
            ),
        }

    api_key = os.getenv(
        "API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable "
            "is not set."
        )

    probe_result = (
        verify_prediction_probe(
            api_base_url=api_base_url,
            api_key=api_key,
            prediction_probe_payload=(
                prediction_probe_payload
            ),
            expected_release_id=(
                manifest.release_id
            ),
            expected_model_version=(
                manifest.model_version
            ),
            expected_model_run_id=(
                manifest.model_run_id
            ),
        )
    )

    p_logger.info(
        "Serving release and prediction "
        "probe verified | "
        f"release_id={probe_result.release_id} | "
        f"model_version="
        f"{probe_result.model_version} | "
        f"model_run_id="
        f"{probe_result.model_run_id} | "
        f"prediction_attempts="
        f"{probe_result.attempts}"
    )

    return {
        "release_id": (
            readiness_result.release_id
        ),
        "model_version": (
            readiness_result.model_version
        ),
        "model_run_id": (
            readiness_result.model_run_id
        ),
        "readiness_attempts": (
            readiness_result.attempts
        ),
        "prediction_probe_status": (
            "verified"
        ),
        "prediction_probe_attempts": (
            probe_result.attempts
        ),
        "probe_predictions": list(
            probe_result.predictions
        ),
    }
