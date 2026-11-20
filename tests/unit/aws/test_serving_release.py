import json
from pathlib import Path

import pytest

from mlops_sales_forecasting.aws.serving_release import prepare_local_serving_release


def write_candidate(root: Path, *, approved: bool = True) -> None:
    output = root / "output"
    output.mkdir(parents=True)
    (output / "evaluation.json").write_text(
        json.dumps(
            {
                "evaluation_approved": approved,
                "run_id": "test-training-run",
                "evaluation_metrics": {"rmse": 100.0},
            }
        ),
        encoding="utf-8",
    )
    (output / "resolved_config.json").write_text(
        json.dumps(
            {
                "model": {"type": "xgboost"},
                "training": {"target_transformation": "log1p"},
            }
        ),
        encoding="utf-8",
    )


def test_rejected_candidate_creates_no_release(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    destination = tmp_path / "serving"
    write_candidate(candidate, approved=False)

    with pytest.raises(ValueError, match="not approved"):
        prepare_local_serving_release(
            candidate_directory=candidate,
            models_directory=destination,
            model_name="test-model",
        )

    assert not destination.exists()


def test_incomplete_candidate_preserves_active_pointer(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    destination = tmp_path / "serving"
    write_candidate(candidate)

    model_root = candidate / "model/model"
    model_root.mkdir(parents=True)
    (model_root / "MLmodel").write_text("placeholder", encoding="utf-8")

    destination.mkdir()
    pointer = destination / "active_serving_release.json"
    original = '{"release_id": "existing-release"}'
    pointer.write_text(original, encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Candidate artifact missing"):
        prepare_local_serving_release(
            candidate_directory=candidate,
            models_directory=destination,
            model_name="test-model",
        )

    assert pointer.read_text(encoding="utf-8") == original
    assert not (destination / "serving_releases").exists()


@pytest.mark.parametrize("relationship", ["same", "nested", "parent"])
def test_rejects_overlapping_directories(
    tmp_path: Path,
    relationship: str,
) -> None:
    candidate = tmp_path / "candidate"
    destinations = {
        "same": candidate,
        "nested": candidate / "serving",
        "parent": tmp_path,
    }

    with pytest.raises(ValueError, match="must not overlap"):
        prepare_local_serving_release(
            candidate_directory=candidate,
            models_directory=destinations[relationship],
            model_name="test-model",
        )
