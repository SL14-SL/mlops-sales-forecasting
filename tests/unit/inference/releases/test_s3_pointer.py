import io
import json
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.inference.releases import pointer

pytest.importorskip("boto3")

from botocore.exceptions import ClientError  # noqa: E402

TIMESTAMP = "2026-11-13T12:00:00+00:00"


def s3_error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock()
    fake.get_object.side_effect = s3_error("NoSuchKey", 404, "GetObject")
    monkeypatch.setattr(pointer, "_s3_client", lambda: fake)
    monkeypatch.setattr(pointer, "file_exists", lambda _: True)
    return fake


def activate() -> pointer.ActiveReleasePointer:
    return pointer.activate_release_pointer(
        models_path="s3://example-bucket/models",
        release_id="release-new",
        updated_at_utc=TIMESTAMP,
    )


def test_bootstrap_requires_absent_pointer(client: MagicMock) -> None:
    result = activate()

    client.get_object.assert_called_once_with(
        Bucket="example-bucket",
        Key="models/active_serving_release.json",
    )
    arguments = client.put_object.call_args.kwargs
    assert arguments["Bucket"] == "example-bucket"
    assert arguments["Key"] == "models/active_serving_release.json"
    assert arguments["IfNoneMatch"] == "*"
    assert "IfMatch" not in arguments
    assert json.loads(arguments["Body"]) == result.to_dict()
    assert result.previous_release_id is None


def test_existing_pointer_requires_matching_etag(client: MagicMock) -> None:
    old_pointer = pointer.ActiveReleasePointer(
        schema_version=1,
        release_id="release-old",
        previous_release_id=None,
        operation=pointer.ReleaseOperation.BOOTSTRAP,
        updated_at_utc=TIMESTAMP,
    )
    body = io.BytesIO(json.dumps(old_pointer.to_dict()).encode("utf-8"))
    client.get_object.side_effect = None
    client.get_object.return_value = {
        "Body": body,
        "ETag": '"old-etag"',
    }

    result = activate()

    arguments = client.put_object.call_args.kwargs
    assert arguments["IfMatch"] == '"old-etag"'
    assert "IfNoneMatch" not in arguments
    assert result.previous_release_id == "release-old"
    assert body.closed


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("PreconditionFailed", 412),
        ("ConditionalRequestConflict", 409),
    ],
)
def test_conflicting_write_is_reported(
    client: MagicMock,
    code: str,
    status: int,
) -> None:
    client.put_object.side_effect = s3_error(code, status, "PutObject")

    with pytest.raises(pointer.ReleaseActivationConflict):
        activate()

    assert client.put_object.call_count == 1


def test_read_permission_error_prevents_write(client: MagicMock) -> None:
    client.get_object.side_effect = s3_error("AccessDenied", 403, "GetObject")

    with pytest.raises(ClientError):
        activate()

    client.put_object.assert_not_called()


def test_write_permission_error_is_preserved(client: MagicMock) -> None:
    client.put_object.side_effect = s3_error("AccessDenied", 403, "PutObject")

    with pytest.raises(ClientError):
        activate()


def test_missing_manifest_prevents_activation(
    client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pointer, "file_exists", lambda _: False)

    with pytest.raises(FileNotFoundError, match="without manifest"):
        activate()

    client.get_object.assert_not_called()
    client.put_object.assert_not_called()


def test_invalid_existing_pointer_prevents_write(client: MagicMock) -> None:
    client.get_object.side_effect = None
    client.get_object.return_value = {
        "Body": io.BytesIO(b"{}"),
        "ETag": '"old-etag"',
    }

    with pytest.raises(ValueError):
        activate()

    client.put_object.assert_not_called()
