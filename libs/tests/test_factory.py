"""
Tests for GCP service factory

Tests GCPServiceFactory for creating service instances with shared credentials.
"""

import pytest
from unittest.mock import Mock, patch
from libs.gcp.base import GCPServiceFactory


@patch("libs.gcp.base.default")
def test_factory_initialization(mock_default, mock_credentials, mock_env_vars):
    """Test GCPServiceFactory initialization"""
    mock_default.return_value = (mock_credentials, None)

    factory = GCPServiceFactory(project_id="test-project")

    assert factory.project_id == "test-project"
    assert isinstance(factory._credential_cache, dict)


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_factory_get_bigquery_client(mock_default, mock_client_class, mock_credentials, mock_env_vars):
    """Test factory creates BigQuery client"""
    mock_default.return_value = (mock_credentials, None)
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    factory = GCPServiceFactory(project_id="test-project")
    bq_service = factory.get_bigquery_client(dataset_id="test_dataset")

    assert bq_service.project_id == "test-project"
    assert bq_service.dataset_id == "test_dataset"


@patch("libs.gcp.storage.storage.Client")
@patch("libs.gcp.base.default")
def test_factory_get_storage_client(mock_default, mock_client_class, mock_credentials, mock_env_vars):
    """Test factory creates Storage client"""
    mock_default.return_value = (mock_credentials, None)
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    factory = GCPServiceFactory(project_id="test-project")
    storage_service = factory.get_storage_client(bucket_name="test-bucket")

    assert storage_service.project_id == "test-project"
    assert storage_service.bucket_name == "test-bucket"


@patch("libs.gcp.base.default")
def test_factory_credential_caching(mock_default, mock_credentials, mock_env_vars):
    """Test factory caches credentials across services"""
    mock_default.return_value = (mock_credentials, None)

    factory = GCPServiceFactory(project_id="test-project")

    # Get BigQuery scopes
    from libs.gcp.bigquery import BigQueryService
    bq_scopes = tuple(BigQueryService._default_scopes_static())

    # First call should cache credentials
    creds1 = factory._get_cached_credentials(bq_scopes)

    # Second call should return cached credentials
    creds2 = factory._get_cached_credentials(bq_scopes)

    assert creds1 is creds2


@patch("libs.gcp.base.default")
def test_factory_get_vertex_ai_credentials(mock_default, mock_credentials, mock_env_vars):
    """Test factory returns Vertex AI credentials"""
    mock_default.return_value = (mock_credentials, None)

    factory = GCPServiceFactory(project_id="test-project")
    project_id, credentials = factory.get_vertex_ai_credentials()

    assert project_id == "test-project"
    assert credentials is not None
