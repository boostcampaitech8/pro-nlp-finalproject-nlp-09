"""
Tests for Cloud Storage service module

Tests StorageService with mocked GCS client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from libs.gcp.storage import StorageService


@patch("libs.gcp.storage.storage.Client")
@patch("libs.gcp.base.default")
def test_storage_service_initialization(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test StorageService initialization"""
    mock_default.return_value = (mock_credentials, None)
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    service = StorageService(bucket_name="test-bucket")

    assert service.project_id == "test-project"
    assert service.bucket_name == "test-bucket"


@patch("libs.gcp.storage.storage.Client")
@patch("libs.gcp.base.default")
def test_list_blobs(mock_default, mock_client_class, mock_credentials, mock_env_vars):
    """Test list_blobs method"""
    mock_default.return_value = (mock_credentials, None)

    # Mock blobs
    blob1 = Mock()
    blob1.name = "file1.txt"
    blob2 = Mock()
    blob2.name = "file2.txt"

    mock_bucket = Mock()
    mock_bucket.list_blobs.return_value = [blob1, blob2]

    mock_client = Mock()
    mock_client.bucket.return_value = mock_bucket
    mock_client_class.return_value = mock_client

    service = StorageService(bucket_name="test-bucket")
    result = service.list_blobs()

    assert result == ["file1.txt", "file2.txt"]
    mock_bucket.list_blobs.assert_called_once()


@patch("libs.gcp.storage.storage.Client")
@patch("libs.gcp.base.default")
def test_blob_exists(mock_default, mock_client_class, mock_credentials, mock_env_vars):
    """Test blob_exists method"""
    mock_default.return_value = (mock_credentials, None)

    mock_blob = Mock()
    mock_blob.exists.return_value = True

    mock_bucket = Mock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = Mock()
    mock_client.bucket.return_value = mock_bucket
    mock_client_class.return_value = mock_client

    service = StorageService(bucket_name="test-bucket")
    result = service.blob_exists("file.txt")

    assert result is True
    mock_blob.exists.assert_called_once()


@patch("libs.gcp.storage.storage.Client")
@patch("libs.gcp.base.default")
def test_delete_blob(mock_default, mock_client_class, mock_credentials, mock_env_vars):
    """Test delete_blob method"""
    mock_default.return_value = (mock_credentials, None)

    mock_blob = Mock()
    mock_blob.delete = Mock()

    mock_bucket = Mock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = Mock()
    mock_client.bucket.return_value = mock_bucket
    mock_client_class.return_value = mock_client

    service = StorageService(bucket_name="test-bucket")
    service.delete_blob("file.txt")

    mock_blob.delete.assert_called_once()


@patch("libs.gcp.storage.storage.Client")
@patch("libs.gcp.base.default")
def test_missing_bucket_name(mock_default, mock_client_class, mock_credentials):
    """Test missing bucket_name raises ValueError"""
    mock_default.return_value = (mock_credentials, None)
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    service = StorageService()

    with pytest.raises(ValueError, match="bucket_name is required"):
        service.list_blobs()
