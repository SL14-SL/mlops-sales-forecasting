import json
from unittest.mock import mock_open, patch

from src.inference.model_manager import load_store_state


@patch("src.inference.model_manager.read_text")
@patch("src.inference.model_manager.file_exists")
def test_load_store_state_supports_local_string_path(
    mock_file_exists,
    mock_read_text,
):
    """Load feature state from a local models path represented as a string."""
    expected_state = {
        "1": {
            "sales_lag_1": 100.0,
        }
    }

    mock_file_exists.return_value = True
    mock_read_text.return_value = json.dumps(expected_state)

    result = load_store_state(
        models_path="data/models",
        gcs_bucket=None,
    )

    mock_file_exists.assert_called_once_with(
        "data/models/latest_state.json"
    )
    mock_read_text.assert_called_once_with(
        "data/models/latest_state.json"
    )
    assert result == expected_state


@patch("src.inference.model_manager.read_text")
@patch("src.inference.model_manager.file_exists")
def test_load_store_state_supports_path_object(
    mock_file_exists,
    mock_read_text,
    tmp_path,
):
    """Load feature state when the models path is a Path object."""
    expected_state = {
        "1": {
            "rolling_mean_7": 250.0,
        }
    }

    mock_file_exists.return_value = True
    mock_read_text.return_value = json.dumps(expected_state)

    result = load_store_state(
        models_path=tmp_path / "models",
        gcs_bucket=None,
    )

    expected_path = str(
        tmp_path / "models" / "latest_state.json"
    )

    mock_file_exists.assert_called_once_with(expected_path)
    mock_read_text.assert_called_once_with(expected_path)
    assert result == expected_state


@patch("src.inference.model_manager.gcsfs.GCSFileSystem")
def test_load_store_state_from_gcs(
    mock_gcs_filesystem,
):
    """Prefer the configured GCS state when it is available."""
    expected_state = {
        "1": {
            "sales_lag_1": 300.0,
        }
    }

    mock_fs = mock_gcs_filesystem.return_value
    mock_fs.exists.return_value = True

    mocked_file = mock_open(
        read_data=json.dumps(expected_state)
    )
    mock_fs.open = mocked_file

    result = load_store_state(
        models_path="data/models",
        gcs_bucket="forecasting-test-bucket",
    )

    expected_gcs_path = (
        "gs://forecasting-test-bucket/"
        "models/latest_state.json"
    )

    mock_fs.exists.assert_called_once_with(
        expected_gcs_path
    )
    mock_fs.open.assert_called_once_with(
        expected_gcs_path,
        "r",
    )
    assert result == expected_state


@patch("src.inference.model_manager.read_text")
@patch("src.inference.model_manager.file_exists")
@patch("src.inference.model_manager.gcsfs.GCSFileSystem")
def test_load_store_state_falls_back_after_gcs_failure(
    mock_gcs_filesystem,
    mock_file_exists,
    mock_read_text,
):
    """Use the configured models path when the GCS state is unavailable."""
    expected_state = {
        "1": {
            "sales_lag_1": 400.0,
        }
    }

    mock_fs = mock_gcs_filesystem.return_value
    mock_fs.exists.return_value = False
    mock_file_exists.return_value = True
    mock_read_text.return_value = json.dumps(expected_state)

    result = load_store_state(
        models_path="data/models",
        gcs_bucket="missing-bucket",
    )

    mock_file_exists.assert_called_once_with(
        "data/models/latest_state.json"
    )
    mock_read_text.assert_called_once_with(
        "data/models/latest_state.json"
    )
    assert result == expected_state


@patch("src.inference.model_manager.read_text")
@patch("src.inference.model_manager.file_exists")
def test_load_store_state_returns_empty_state_when_missing(
    mock_file_exists,
    mock_read_text,
):
    """Return an empty state when no persisted snapshot exists."""
    mock_file_exists.return_value = False

    result = load_store_state(
        models_path="data/models",
        gcs_bucket=None,
    )

    assert result == {}
    mock_read_text.assert_not_called()