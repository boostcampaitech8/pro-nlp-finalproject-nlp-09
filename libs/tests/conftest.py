"""
Shared test fixtures for libs package tests

This module provides common fixtures for testing GCP services,
configuration, and other utilities.
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator


@pytest.fixture
def mock_credentials():
    """Mock Google Cloud credentials"""
    creds = Mock()
    creds.valid = True
    creds.token = "mock-access-token"
    creds.refresh = Mock()
    return creds


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing"""
    env_vars = {
        "VERTEX_AI_PROJECT_ID": "test-project",
        "VERTEX_AI_LOCATION": "us-central1",
        "GENERATE_MODEL_NAME": "test-model",
        "GENERATE_MODEL_TEMPERATURE": "0.7",
        "GENERATE_MODEL_MAX_TOKENS": "2048",
        "BIGQUERY_DATASET_ID": "test_dataset",
        "BIGQUERY_TABLE_ID": "test_table",
        "BIGQUERY_DATE_COLUMN": "time",
        "BIGQUERY_VALUE_COLUMN": "close",
        "BIGQUERY_BASE_DATE": "2025-01-20",
        "BIGQUERY_DAYS": "30",
        "GCS_BUCKET_NAME": "test-bucket",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
        "DEBUG": "false",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def temp_env_file() -> Generator[Path, None, None]:
    """Create a temporary .env file for testing"""
    with TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / ".env"
        env_file.write_text(
            "VERTEX_AI_PROJECT_ID=test-project\nBIGQUERY_DATASET_ID=test_dataset\nGCS_BUCKET_NAME=test-bucket\n"
        )
        yield env_file


@pytest.fixture
def mock_bigquery_client(mock_credentials):
    """Mock BigQuery client"""
    client = Mock()
    client.project = "test-project"

    # Mock query method
    query_job = Mock()
    query_job.result = Mock(return_value=[])
    query_job.to_dataframe = Mock(return_value=MagicMock())
    client.query = Mock(return_value=query_job)

    return client


@pytest.fixture
def mock_storage_client(mock_credentials):
    """Mock Cloud Storage client"""
    client = Mock()
    client.project = "test-project"

    # Mock bucket
    bucket = Mock()
    blob = Mock()
    blob.exists = Mock(return_value=True)
    blob.upload_from_filename = Mock()
    blob.download_to_filename = Mock()
    blob.delete = Mock()
    bucket.blob = Mock(return_value=blob)
    bucket.list_blobs = Mock(return_value=[])
    client.bucket = Mock(return_value=bucket)

    return client


@pytest.fixture
def sample_prophet_data():
    """Sample Prophet-style data for testing"""
    import pandas as pd
    from datetime import datetime, timedelta

    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(10)]

    return pd.DataFrame(
        {
            "ds": dates,
            "y": [100.0 + i for i in range(10)],
            "trend": [100.0 + i * 0.5 for i in range(10)],
        }
    )


@pytest.fixture
def sample_news_data():
    """Sample news data for testing"""
    import pandas as pd
    from datetime import datetime, timedelta

    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(5)]

    return pd.DataFrame(
        {
            "publish_date": dates,
            "title": [f"News {i}" for i in range(5)],
            "all_text": [f"Article content {i}" for i in range(5)],
            "positive_score": [0.5 + i * 0.1 for i in range(5)],
            "negative_score": [0.5 - i * 0.1 for i in range(5)],
            "filter_status": ["T"] * 5,
        }
    )


@pytest.fixture(autouse=True)
def reset_config():
    """Reset ConfigManager singleton between tests"""
    from libs.utils.config import ConfigManager

    yield
    ConfigManager.reset()
