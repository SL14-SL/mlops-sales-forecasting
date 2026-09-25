import json
import re

from ..storage.filesystem import (
    file_exists,
    list_files,
    read_text,
    write_text,
)
from .serialization import (
    parse_pipeline_run,
    pipeline_run_to_dict,
)
from .status import PipelineRun

_RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)


def _join_path(
    root: str,
    *parts: str,
) -> str:
    """Join local or object-storage path components."""
    normalized_root = root.rstrip("/")
    normalized_parts = [
        part.strip("/")
        for part in parts
    ]

    return "/".join(
        [
            normalized_root,
            *normalized_parts,
        ]
    )


def _validate_run_id(
    run_id: str,
) -> None:
    """Validate a pipeline run ID for safe path usage."""
    if (
        not isinstance(run_id, str)
        or not _RUN_ID_PATTERN.fullmatch(run_id)
    ):
        raise ValueError(
            f"Invalid pipeline run ID: {run_id!r}."
        )


class PipelineRunRepository:
    """Persist pipeline lifecycle states as JSON documents."""

    def __init__(
        self,
        root_path: str,
    ) -> None:
        if (
            not isinstance(root_path, str)
            or not root_path.strip()
        ):
            raise ValueError(
                "Pipeline run repository root path "
                "must not be empty."
            )

        self._root_path = root_path.rstrip("/")

    @property
    def root_path(self) -> str:
        """Return the configured storage root."""
        return self._root_path

    def path_for(
        self,
        run_id: str,
    ) -> str:
        """Return the JSON path belonging to a run."""
        _validate_run_id(run_id)

        return _join_path(
            self._root_path,
            f"{run_id}.json",
        )

    def save(
        self,
        run: PipelineRun,
    ) -> PipelineRun:
        """Persist the latest state of a pipeline run."""
        if not isinstance(run, PipelineRun):
            raise TypeError(
                "Pipeline run repository can only save PipelineRun."
            )

        path = self.path_for(run.run_id)
        serialized = json.dumps(
            pipeline_run_to_dict(run),
            indent=2,
            sort_keys=True,
        )

        write_text(
            path,
            f"{serialized}\n",
        )

        return run

    def load(
        self,
        run_id: str,
    ) -> PipelineRun:
        """Load and validate one pipeline run."""
        path = self.path_for(run_id)

        if not file_exists(path):
            raise FileNotFoundError(
                f"Pipeline run not found: {run_id}."
            )

        try:
            payload = json.loads(
                read_text(path)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Pipeline run contains invalid JSON: "
                f"{run_id}."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Pipeline run document must "
                "contain a JSON object."
            )

        run = parse_pipeline_run(payload)

        if run.run_id != run_id:
            raise ValueError(
                "Pipeline run ID does not match "
                "its storage path."
            )

        return run

    def list_runs(self) -> list[PipelineRun]:
        """Return all stored runs ordered newest first."""
        pattern = _join_path(
            self._root_path,
            "*.json",
        )
        runs = [
            self._load_path(path)
            for path in list_files(pattern)
        ]

        return sorted(
            runs,
            key=lambda run: run.started_at_utc,
            reverse=True,
        )

    def _load_path(
        self,
        path: str,
    ) -> PipelineRun:
        """Load one run directly from its storage path."""
        try:
            payload = json.loads(
                read_text(path)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Pipeline run contains invalid JSON: "
                f"{path}."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Pipeline run document must "
                "contain a JSON object."
            )

        return parse_pipeline_run(payload)