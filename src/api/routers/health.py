import mlflow
from fastapi import HTTPException, Response

from src.api import serving_state
from src.configs.loader import load_config


CFG = load_config()
MODEL_NAME = CFG["model"]["registry_name"]


def health(response: Response):
    bundle = (
        serving_state
        .active_serving_bundle
    )

    is_healthy = (
        bundle is not None
        and bundle.model is not None
        and bundle.store_metadata is not None
        and bundle.store_state is not None
    )

    if not is_healthy:
        response.status_code = 503

    return {
        "status": (
            "online"
            if is_healthy
            else "degraded"
        ),
        "model_loaded": (
            bundle is not None
            and bundle.model is not None
        ),
        "store_metadata_loaded": (
            bundle is not None
            and bundle.store_metadata is not None
        ),
        "state_loaded": (
            bundle is not None
            and bundle.store_state is not None
        ),
        "calendar_loaded": (
            bundle is not None
            and bundle.known_calendar is not None
        ),
        "model_type": (
            bundle.model_type
            if bundle is not None
            else None
        ),
        "target_transformation": (
            bundle.target_transformation
            if bundle is not None
            else None
        ),
        "model_name": (
            bundle.model_name
            if bundle is not None
            else MODEL_NAME
        ),
        "tracking_uri": (
            mlflow.get_tracking_uri()
        ),
        "serving_alias": (
            bundle.serving_alias
            if bundle is not None
            else None
        ),
        "model_uri": (
            bundle.model_uri
            if bundle is not None
            else None
        ),
        "model_version": (
            bundle.model_version
            if bundle is not None
            else None
        ),
        "model_run_id": (
            bundle.model_run_id
            if bundle is not None
            else None
        ),
        "release_id": (
            bundle.release_id
            if bundle is not None
            else None
        ),
    }


def livez():
    """
    Report whether the API process is running.

    Liveness does not require a loaded serving bundle.
    """
    return {
        "status": "alive",
        "service": CFG.get(
            "project_name",
            "sales-forecasting-api",
        ),
        "environment": CFG.get(
            "environment",
            "unknown",
        ),
    }


def readyz():
    """
    Report whether the API can safely serve predictions.
    """
    bundle = (
        serving_state
        .active_serving_bundle
    )

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No complete serving "
                "bundle is active."
            ),
        )

    return {
        "status": "ready",
        "serving_bundle_loaded": True,
        "model_name": bundle.model_name,
        "model_type": bundle.model_type,
        "target_transformation": (
            bundle.target_transformation
        ),
        "serving_alias": bundle.serving_alias,
        "model_version": bundle.model_version,
        "model_run_id": bundle.model_run_id,
        "model_uri": bundle.model_uri,
        "store_metadata_loaded": (
            bundle.store_metadata is not None
        ),
        "state_loaded": (
            bundle.store_state is not None
        ),
        "calendar_loaded": (
            bundle.known_calendar is not None
        ),
        "release_id": bundle.release_id,
    }