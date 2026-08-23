import os


from src.inference.serving_bundle import ServingBundle
from src.configs.loader import load_config, get_path
from src.inference.model_manager import (
    reload_serving_model as reload_model_state,
)
from src.inference.model_manager import (
    load_serving_bundle,
)

CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]
VALIDATED_PATH = get_path("validated_data")
MODELS_PATH = get_path("models")
FEATURES_PATH = get_path("features")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", CFG.get("gcp", {}).get("gcs", {}).get("bucket_name"))

active_serving_bundle: (
    ServingBundle | None
) = None

model = None
model_type: str | None = None
target_transformation: str | None = None
serving_alias: str | None = None
model_uri: str | None = None
serving_model_version: str | None = None
serving_model_run_id: str | None = None

store_metadata = None
store_state = None
known_calendar = None


def activate_serving_bundle(
    bundle: ServingBundle,
) -> dict:
    """
    Atomically replace the active in-memory serving state.
    """
    global active_serving_bundle
    global model, model_type, target_transformation
    global serving_alias, model_uri
    global serving_model_version, serving_model_run_id
    global store_metadata, store_state, known_calendar

    # One authoritative snapshot reference.
    active_serving_bundle = bundle

    # Compatibility assignments for existing inference code.
    model = bundle.model
    model_type = bundle.model_type
    target_transformation = bundle.target_transformation
    serving_alias = bundle.serving_alias
    model_uri = bundle.model_uri
    serving_model_version = bundle.model_version
    serving_model_run_id = bundle.model_run_id
    store_metadata = bundle.store_metadata
    store_state = bundle.store_state
    known_calendar = bundle.known_calendar

    return {
        "release_id": bundle.release_id,
        "model_name": bundle.model_name,
        "serving_alias": bundle.serving_alias,
        "model_version": bundle.model_version,
        "model_run_id": bundle.model_run_id,
        "model_uri": bundle.model_uri,
        "target_transformation": bundle.target_transformation,
        "store_metadata_loaded": True,
        "state_loaded": True,
        "calendar_loaded": True,
    }


def reload_serving_model() -> dict:
    """
    Reload model state and update API globals.
    """
    global model, model_type, target_transformation, serving_alias, model_uri
    global serving_model_version, serving_model_run_id

    state = reload_model_state(
        model_name=MODEL_NAME,
        cfg=CFG,
    )

    model = state["model"]
    model_type = state["model_type"]
    target_transformation = state["target_transformation"]
    serving_alias = state["serving_alias"]
    model_uri = state["model_uri"]
    serving_model_version = state["serving_model_version"]
    serving_model_run_id = state["serving_model_run_id"]

    return {
        "model_name": MODEL_NAME,
        "serving_alias": serving_alias,
        "model_version": serving_model_version,
        "model_run_id": serving_model_run_id,
        "model_uri": model_uri,
        "target_transformation": target_transformation,
    }

def reload_complete_serving_bundle() -> dict:
    """
    Load and validate a candidate bundle before activating it.
    """
    candidate_bundle = load_serving_bundle(
        model_name=MODEL_NAME,
        cfg=CFG,
        validated_path=VALIDATED_PATH,
        features_path=FEATURES_PATH,
        models_path=MODELS_PATH,
        gcs_bucket=GCS_BUCKET,
    )

    return activate_serving_bundle(candidate_bundle)

