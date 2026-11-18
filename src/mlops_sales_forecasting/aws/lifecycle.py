"""AWS entry point for the same lifecycle used by local and GCP projects."""

import argparse
import json
import os
from typing import Any
from urllib.parse import urlsplit

from ..configs.loader import load_config
from ..orchestration.lifecycle_adapter import run_prefect_model_lifecycle
from ..orchestration.retraining_service import run_auto_retraining
from ..pipeline.project_factory import build_project_training_pipeline


def load_aws_lifecycle_config(*, estimators: int | None = None) -> dict[str, Any]:
    """Resolve S3 paths and require a shared, reachable tracking service."""
    if os.environ.get("CLOUD_PROVIDER") != "aws":
        raise ValueError("Set CLOUD_PROVIDER=aws for the AWS lifecycle.")
    config = load_config()
    tracking_uri = config["tracking"]["mlflow_tracking_uri"]
    parsed = urlsplit(tracking_uri)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname
        in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
    ):
        raise ValueError("AWS lifecycle requires a shared HTTP(S) MLFLOW_TRACKING_URI.")
    if estimators is not None:
        if estimators < 1:
            raise ValueError("estimators must be positive.")
        config["model"]["params"]["n_estimators"] = estimators
    return config


def run_aws_lifecycle(*, mode: str = "train", estimators: int | None = None) -> dict[str, Any]:
    """Train or evaluate retraining signals using the shared lifecycle services."""
    if mode not in {"train", "auto-retrain"}:
        raise ValueError("mode must be train or auto-retrain.")
    config = load_aws_lifecycle_config(estimators=estimators)
    if mode == "auto-retrain":

        def after_lifecycle(result):
            if result.serving_release is not None:
                _refresh_promoted_release(config)

        return run_auto_retraining(config=config, after_lifecycle=after_lifecycle).to_dict()
    result = run_prefect_model_lifecycle(
        pipeline=build_project_training_pipeline(config),
        mlflow_tags={"cloud.provider": "aws"},
    )
    if result.serving_release is not None:
        _refresh_promoted_release(config)
    candidate = result.candidate
    return {
        "run_id": result.pipeline.pipeline.training.run_id,
        "registered": candidate.registration.registered,
        "model_version": candidate.registration.model_version,
        "promoted": result.serving_release is not None,
        "release_id": (
            result.serving_release.manifest.release_id if result.serving_release else None
        ),
        "promotion_reason": candidate.promotion.decision.reason if candidate.promotion else None,
    }


def _refresh_promoted_release(config: dict[str, Any]) -> None:
    """Apply a new S3 pointer to ECS when serving is configured."""
    if not os.environ.get("AWS_API_SERVICE"):
        return  # Bootstrap training can precede the first API deployment.
    from ..inference.releases.pointer import load_active_release_pointer
    from .operations import refresh_service, rollback_model

    pointer = load_active_release_pointer(models_path=config["paths"]["models"])
    try:
        refresh_service(expected_release_id=pointer.release_id)
    except Exception as deployment_error:
        if pointer.previous_release_id is None:
            raise RuntimeError(
                "Initial model was published, but ECS refresh failed; no previous release exists."
            ) from deployment_error
        try:
            rollback_model(config=config)
        except Exception as rollback_error:
            raise ExceptionGroup(
                "ECS refresh and model rollback failed.", [deployment_error, rollback_error]
            ) from None
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["train", "auto-retrain", "rollback-model", "refresh", "simulate"],
        nargs="?",
        default="train",
    )
    parser.add_argument("--estimators", type=int)
    parser.add_argument("--maximum-days", type=int, default=95)
    parser.add_argument("--simulation-mode", choices=["both", "managed", "static"], default="both")
    args = parser.parse_args()
    if args.mode == "simulate":
        from .simulation import run_aws_simulation

        report = run_aws_simulation(
            maximum_days=args.maximum_days, mode=args.simulation_mode, estimators=args.estimators
        )
        print(json.dumps(report, indent=2))
        return
    if args.mode in {"rollback-model", "refresh"}:
        from ..inference.releases.pointer import load_active_release_pointer
        from .operations import refresh_service, rollback_model

        config = load_aws_lifecycle_config()
        if args.mode == "rollback-model":
            release_id = rollback_model(config=config)
        else:
            release_id = load_active_release_pointer(
                models_path=config["paths"]["models"]
            ).release_id
            refresh_service(expected_release_id=release_id)
        report = {"release_id": release_id, "operation": args.mode}
    else:
        report = run_aws_lifecycle(mode=args.mode, estimators=args.estimators)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
