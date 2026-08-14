from unittest.mock import MagicMock

import pandas as pd

from src.monitoring import monitoring_refresh


def test_rebuilds_cumulative_ground_truth(
    monkeypatch,
):
    first_batch = (
        b"Store,Date,Sales\n"
        b"1,2026-08-12,100\n"
        b"2,2026-08-12,200\n"
    )
    second_batch = (
        b"Store,Date,Sales\n"
        b"1,2026-08-12,110\n"
        b"1,2026-08-13,120\n"
    )

    opened_files = {
        "batch-1.csv": first_batch,
        "batch-2.csv": second_batch,
    }

    class ReadContext:
        def __init__(self, content):
            self.file = MagicMock()
            self.file.read.return_value = (
                content
            )

        def __enter__(self):
            return self.file

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            return False

    written = MagicMock()

    def mock_open(path, mode):
        if mode == "rb":
            return ReadContext(
                opened_files[path]
            )

        return written

    monkeypatch.setattr(
        monitoring_refresh.fsspec,
        "open",
        mock_open,
    )

    result = (
        monitoring_refresh
        .rebuild_cumulative_ground_truth(
            [
                "batch-1.csv",
                "batch-2.csv",
            ],
            output_path="cumulative.csv",
        )
    )

    assert len(result) == 3

    updated_value = result.loc[
        (
            result["Store"] == 1
        )
        & (
            result["Date"]
            == pd.Timestamp(
                "2026-08-12"
            )
        ),
        "Sales",
    ].iloc[0]

    assert updated_value == 110


def test_refresh_without_batches_is_safe(
    monkeypatch,
):
    monkeypatch.setattr(
        monitoring_refresh,
        "get_path",
        MagicMock(
            side_effect=lambda name: {
                "raw_data": "data/raw",
                "predictions": (
                    "data/predictions"
                ),
                "monitoring": (
                    "data/monitoring"
                ),
            }[name]
        ),
    )
    monkeypatch.setattr(
        monitoring_refresh,
        "_list_files",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        monitoring_refresh,
        "file_exists",
        MagicMock(return_value=False),
    )
    monkeypatch.setattr(
        monitoring_refresh,
        "run_feature_drift_check",
        MagicMock(
            return_value=pd.DataFrame()
        ),
    )

    result = (
        monitoring_refresh
        .refresh_monitoring_signals()
    )

    assert result.ground_truth_rows == 0
    assert result.performance_updated is False
    assert (
        result.feature_drift_updated
        is False
    )