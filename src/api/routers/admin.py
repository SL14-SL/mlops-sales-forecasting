import os
import traceback

from fastapi import HTTPException, Depends

from src.inference.model_manager import (
    load_store_state,
    load_serving_bundle_for_release
)

from src.inference.releases.repository import (
    activate_release_pointer, 
    list_serving_release_manifests, 
    load_active_release_id,
)

from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger

from src.api.schema import ServingRollbackRequest
from src.api.dependencies import get_api_key
from src.api.serving_state import (
    activate_serving_bundle, 
    reload_complete_serving_bundle, 
    reload_serving_model,
)

logger = get_logger(__name__)

CFG = load_config()
MODEL_NAME = CFG["model"]["registry_name"]
MODELS_PATH = get_path("models")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", CFG.get("gcp", {}).get("gcs", {}).get("bucket_name"))


def reload_model(api_key: str = Depends(get_api_key)):
    """
    Reload the current champion forecasting model from MLflow.

    Used after a new champion model version has been promoted.
    """
    try:
        result = reload_serving_model()

    except Exception as error:
        logger.error("Model reload failed: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(error)}",
        )

    return {
        "status": "reloaded",
        **result,
    }


def reload_serving_state(
    api_key: str = Depends(get_api_key),
):
    """
    Atomically reload the complete forecasting serving bundle.
    """
    try:
        result = reload_complete_serving_bundle()

    except Exception as error:
        logger.error(
            "Serving bundle reload failed: %s",
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Serving bundle reload failed. "
                "The previous serving state remains active. "
                f"Reason: {error}"
            ),
        ) from error

    return {
        "status": "reloaded",
        **result,
    }

def reload_feature_state(api_key: str = Depends(get_api_key)):
    """
    Reload only the forecasting feature state.

    This endpoint is used after newly available ground truth has been
    appended to latest_state.json. It does not reload or replace the model.
    """
    global store_state

    try:
        updated_state = load_store_state(
            models_path=MODELS_PATH,
            gcs_bucket=GCS_BUCKET,
        )

        if updated_state is None:
            updated_state = {}

        store_state = updated_state

    except Exception as error:
        logger.error(
            "Feature state reload failed: %s",
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Feature state reload failed: {str(error)}",
        )

    return {
        "status": "reloaded",
        "state_loaded": store_state is not None,
        "state_entities": len(store_state or {}),
    }

def list_serving_releases(
    api_key: str = Depends(get_api_key),
):
    """
    List available immutable serving releases and identify the active release.

    Returns:
        Release summaries ordered according to the release repository.

    Raises:
        HTTPException: If release metadata cannot be loaded.
    """
    active_release_id = (
        load_active_release_id(
            models_path=MODELS_PATH,
        )
    )

    manifests = (
        list_serving_release_manifests(
            models_path=MODELS_PATH,
        )
    )

    return {
        "active_release_id": (
            active_release_id
        ),
        "releases": [
            {
                "release_id": (
                    manifest.release_id
                ),
                "active": (
                    manifest.release_id
                    == active_release_id
                ),
                "created_at_utc": (
                    manifest.created_at_utc
                ),
                "model_name": (
                    manifest.model_name
                ),
                "model_version": (
                    manifest.model_version
                ),
                "model_run_id": (
                    manifest.model_run_id
                ),
                "dataset_version": (
                    manifest.dataset_version
                ),
                "git_commit": (
                    manifest.git_commit
                ),
            }
            for manifest in manifests
        ],
    }

def rollback_serving_release(
    payload: ServingRollbackRequest,
    api_key: str = Depends(get_api_key),
):
    """
    Validate and atomically activate a previously published release.
    """
    previous_release_id = (
        load_active_release_id(
            models_path=MODELS_PATH,
        )
    )

    if payload.release_id == previous_release_id:
        return {
            "status": "unchanged",
            "release_id": previous_release_id,
            "previous_release_id": (
                previous_release_id
            ),
        }

    pointer_changed = False

    try:
        # Fully load model and artifacts before changing the pointer.
        candidate_bundle = (
            load_serving_bundle_for_release(
                release_id=payload.release_id,
                model_name=MODEL_NAME,
                cfg=CFG,
                models_path=MODELS_PATH,
            )
        )

        # The target bundle is valid. Persist the new active release.
        activate_release_pointer(
            models_path=MODELS_PATH,
            release_id=payload.release_id,
            operation="rollback",
            previous_release_id=(
                previous_release_id
            ),
        )
        pointer_changed = True

        # Activate the already validated bundle in this API process.
        result = activate_serving_bundle(
            candidate_bundle
        )

    except Exception as error:
        if pointer_changed:
            try:
                activate_release_pointer(
                    models_path=MODELS_PATH,
                    release_id=(
                        previous_release_id
                    ),
                    operation=(
                        "rollback_reverted"
                    ),
                    previous_release_id=(
                        payload.release_id
                    ),
                )

            except Exception:
                logger.exception(
                    "CRITICAL: rollback pointer "
                    "could not be restored | "
                    "expected_release_id=%s",
                    previous_release_id,
                )

        logger.exception(
            "Serving release rollback failed | "
            "target_release_id=%s | "
            "previous_release_id=%s | "
            "pointer_changed=%s",
            payload.release_id,
            previous_release_id,
            pointer_changed,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Serving release rollback failed. "
                f"Reason: {error}"
            ),
        ) from error

    logger.warning(
        "Serving release rollback completed | "
        "previous_release_id=%s | "
        "active_release_id=%s | "
        "model_version=%s",
        previous_release_id,
        candidate_bundle.release_id,
        candidate_bundle.model_version,
    )

    return {
        "status": "rolled_back",
        "previous_release_id": (
            previous_release_id
        ),
        **result,
    }