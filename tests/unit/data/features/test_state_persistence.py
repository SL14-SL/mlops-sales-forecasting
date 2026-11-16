import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.data.features import update_state
from mlops_sales_forecasting.storage import filesystem


def test_replaces_existing_local_state(tmp_path: Path) -> None:
    state_path = tmp_path / "nested" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text('{"1": [10.0]}', encoding="utf-8")

    update_state._write_state_atomically(
        {"1": [10.0, 20.0]},
        str(state_path),
    )

    assert json.loads(state_path.read_text(encoding="utf-8")) == {"1": [10.0, 20.0]}
    assert sorted(path.name for path in state_path.parent.iterdir()) == ["state.json"]


def test_serialization_failure_preserves_existing_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    original = '{"1": [10.0]}'
    state_path.write_text(original, encoding="utf-8")

    with pytest.raises(TypeError):
        update_state._write_state_atomically(
            {"invalid": object()},
            str(state_path),
        )

    assert state_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("scheme", ["gs", "s3"])
def test_remote_state_is_written_without_delete_or_move(
    scheme: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = MagicMock()
    factory_name = "_gcs_fs" if scheme == "gs" else "_s3_fs"
    monkeypatch.setattr(filesystem, factory_name, lambda: remote)
    state_path = f"{scheme}://example-bucket/models/state.json"
    state = {"1": [10.0, 20.0]}

    update_state._write_state_atomically(state, state_path)

    remote.open.assert_called_once_with(
        state_path,
        "w",
        encoding="utf-8",
    )
    handle = remote.open.return_value.__enter__.return_value
    serialized = handle.write.call_args.args[0]
    assert json.loads(serialized) == state
    remote.rm.assert_not_called()
    remote.mv.assert_not_called()
