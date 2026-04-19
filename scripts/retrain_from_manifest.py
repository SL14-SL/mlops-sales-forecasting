import sys
import mlflow

from src.data.versioning import log_dataset_manifest_to_mlflow
from src.training.train import train
from src.training.evaluate import compare_models


def main(manifest_path: str):
    import json
    from pathlib import Path

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    snapshots = manifest["snapshots"]

    train_file = snapshots["split_train"]
    val_file = snapshots["split_val"]

    print(f"📦 Reproducing training from manifest: {manifest_path}")
    print(f" - train: {train_file}")
    print(f" - val:   {val_file}")

    _, run_id = train(train_file=train_file, val_file=val_file)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("reproduced_from_manifest", manifest_path)
        mlflow.log_param("reproduced_dataset_version", manifest["dataset_version"])
        log_dataset_manifest_to_mlflow(manifest)

    is_better, metrics = compare_models(run_id, val_path=val_file)

    print(f"✅ Reproduction run completed: {run_id}")
    print(f"Comparison: {is_better} | Metrics: {metrics}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/retrain_from_manifest.py <manifest_path>")

    main(sys.argv[1])