import argparse
from collections.abc import Sequence

from mlops_sales_forecasting.inference.model_loader import (
    configure_mlflow,
)
from mlops_sales_forecasting.inference.releases.cloud_exporter import (
    export_active_release,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the cloud-release exporter CLI."""
    parser = argparse.ArgumentParser(
        description=("Export the active local serving release with a portable model.")
    )
    parser.add_argument(
        "--source-models-path",
        default="artifacts/models",
        help=("Models path containing the active source release."),
    )
    parser.add_argument(
        "--target-models-path",
        required=True,
        help=("Destination models path, normally gs://BUCKET/models."),
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default="http://127.0.0.1:5000",
        help=("MLflow tracking URI used to download the registered model."),
    )
    parser.add_argument(
        "--release-id",
        default=None,
        help=("Optional explicit target release ID."),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Export and activate one portable release."""
    args = build_parser().parse_args(argv)

    configure_mlflow(args.mlflow_tracking_uri)

    result = export_active_release(
        source_models_path=(args.source_models_path),
        target_models_path=(args.target_models_path),
        release_id=args.release_id,
    )

    print(
        "Serving release exported | "
        f"release_id={result.manifest.release_id} | "
        f"release_root={result.release_root} | "
        f"model_uri={result.manifest.model.uri}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
