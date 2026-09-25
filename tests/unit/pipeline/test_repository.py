import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.pipeline import repository as repository_module
from mlops_sales_forecasting.pipeline.repository import (
    PipelineRunRepository,
)
from mlops_sales_forecasting.pipeline.status import (
    PipelineRun,
    PipelineRunStatus,
)


def build_run(
    run_id: str = "pipeline-run-123",
    *,
    started_at: datetime | None = None,
) -> PipelineRun:
    return PipelineRun.start(
        run_id,
        started_at_utc=(
            started_at
            or datetime(
                2026,
                9,
                15,
                8,
                0,
                tzinfo=UTC,
            )
        ),
    )


def test_repository_saves_and_loads_running_run(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path / "pipeline-runs")
    )
    original = build_run()

    saved = repository.save(original)
    restored = repository.load(
        original.run_id
    )

    assert saved is original
    assert restored == original


def test_repository_replaces_run_with_latest_state(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path / "pipeline-runs")
    )
    running = build_run()
    completed = running.succeed(
        completed_at_utc=(
            running.started_at_utc
            + timedelta(seconds=10)
        )
    )

    repository.save(running)
    repository.save(completed)

    restored = repository.load(
        running.run_id
    )

    assert restored.status is (
        PipelineRunStatus.SUCCEEDED
    )
    assert restored.duration_seconds == 10


def test_saved_document_is_readable_json(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path / "pipeline-runs")
    )
    run = build_run()

    repository.save(run)

    path = Path(
        repository.path_for(run.run_id)
    )
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert payload["run_id"] == run.run_id
    assert payload["status"] == "running"
    assert payload["schema_version"] == 1


def test_repository_lists_runs_newest_first(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path / "pipeline-runs")
    )
    first = build_run(
        "pipeline-run-first",
        started_at=datetime(
            2026,
            9,
            15,
            8,
            0,
            tzinfo=UTC,
        ),
    )
    second = build_run(
        "pipeline-run-second",
        started_at=datetime(
            2026,
            9,
            15,
            9,
            0,
            tzinfo=UTC,
        ),
    )

    repository.save(first)
    repository.save(second)

    assert repository.list_runs() == [
        second,
        first,
    ]


def test_repository_returns_empty_list(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path / "pipeline-runs")
    )

    assert repository.list_runs() == []


def test_repository_rejects_empty_root_path() -> None:
    with pytest.raises(
        ValueError,
        match="root path",
    ):
        PipelineRunRepository("")


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "../outside",
        "nested/run",
        "/absolute",
        "contains space",
        "a" * 129,
    ],
)
def test_repository_rejects_unsafe_run_id(
    run_id: str,
) -> None:
    repository = PipelineRunRepository(
        "artifacts/pipeline-runs"
    )

    with pytest.raises(
        ValueError,
        match="Invalid pipeline run ID",
    ):
        repository.path_for(run_id)


def test_repository_rejects_non_pipeline_run(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path)
    )

    with pytest.raises(
        TypeError,
        match="only save PipelineRun",
    ):
        repository.save(
            object()  # type: ignore[arg-type]
        )


def test_repository_rejects_missing_run(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path)
    )

    with pytest.raises(
        FileNotFoundError,
        match="Pipeline run not found",
    ):
        repository.load("missing-run")


def test_repository_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path)
    )
    path = Path(
        repository.path_for("broken-run")
    )
    path.write_text(
        "not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        repository.load("broken-run")


def test_repository_rejects_non_object_document(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path)
    )
    path = Path(
        repository.path_for("list-run")
    )
    path.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        repository.load("list-run")


def test_repository_rejects_mismatched_run_id(
    tmp_path: Path,
) -> None:
    repository = PipelineRunRepository(
        str(tmp_path)
    )
    repository.save(
        build_run("stored-run")
    )

    original_path = Path(
        repository.path_for("stored-run")
    )
    mismatched_path = Path(
        repository.path_for("different-run")
    )
    mismatched_path.write_text(
        original_path.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        repository.load("different-run")


def test_repository_supports_gcs_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_text = MagicMock()
    monkeypatch.setattr(
        repository_module,
        "write_text",
        write_text,
    )

    repository = PipelineRunRepository(
        "gs://example-bucket/pipeline-runs"
    )
    run = build_run()

    repository.save(run)

    path = (
        "gs://example-bucket/"
        "pipeline-runs/"
        "pipeline-run-123.json"
    )

    assert write_text.call_args.args[0] == path

    payload = json.loads(
        write_text.call_args.args[1]
    )

    assert payload["run_id"] == (
        "pipeline-run-123"
    )