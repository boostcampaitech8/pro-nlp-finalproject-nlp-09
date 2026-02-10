"""
Tests for configuration management module

Tests ConfigManager, Pydantic models, and environment variable loading.
"""

import pytest
from libs.utils.config import (
    VertexAIConfig,
    BigQueryConfig,
    StorageConfig,
    APIConfig,
    AppConfig,
    ConfigManager,
    get_config,
)


def test_vertex_ai_config_defaults():
    """Test VertexAIConfig with default values"""
    config = VertexAIConfig()
    assert config.location == "us-central1"
    assert config.model_name == "meta/llama-3.1-70b-instruct-maas"
    assert config.temperature == 0.7
    assert config.max_tokens == 2048


def test_vertex_ai_config_from_env(mock_env_vars):
    """Test VertexAIConfig loads from environment variables"""
    config = VertexAIConfig()
    assert config.project_id == "test-project"
    assert config.location == "us-central1"
    assert config.model_name == "test-model"


def test_vertex_ai_config_temperature_validation():
    """Test temperature validation"""
    with pytest.raises(ValueError, match="temperature must be between"):
        VertexAIConfig(temperature=1.5)

    with pytest.raises(ValueError, match="temperature must be between"):
        VertexAIConfig(temperature=-0.1)


def test_vertex_ai_config_max_tokens_validation():
    """Test max_tokens validation"""
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        VertexAIConfig(max_tokens=0)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        VertexAIConfig(max_tokens=-100)


def test_bigquery_config_defaults():
    """Test BigQueryConfig with default values"""
    config = BigQueryConfig()
    assert config.date_column == "time"
    assert config.value_column == "close"
    assert config.days == 30


def test_bigquery_config_from_env(mock_env_vars):
    """Test BigQueryConfig loads from environment variables"""
    config = BigQueryConfig()
    assert config.dataset_id == "test_dataset"
    assert config.table_id == "test_table"
    assert config.base_date == "2025-01-20"
    assert config.days == 30


def test_bigquery_config_days_validation():
    """Test days validation"""
    with pytest.raises(ValueError, match="days must be positive"):
        BigQueryConfig(days=0)

    with pytest.raises(ValueError, match="days must be positive"):
        BigQueryConfig(days=-10)


def test_bigquery_config_base_date_validation():
    """Test base_date format validation"""
    # Valid date
    config = BigQueryConfig(base_date="2025-01-20")
    assert config.base_date == "2025-01-20"

    # Invalid date format
    with pytest.raises(ValueError, match="base_date must be in YYYY-MM-DD format"):
        BigQueryConfig(base_date="2025/01/20")


def test_storage_config():
    """Test StorageConfig"""
    config = StorageConfig(bucket_name="test-bucket")
    assert config.bucket_name == "test-bucket"


def test_api_config_defaults():
    """Test APIConfig with default values"""
    config = APIConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.debug is False


def test_api_config_port_validation():
    """Test port validation"""
    with pytest.raises(ValueError, match="port must be between"):
        APIConfig(port=0)

    with pytest.raises(ValueError, match="port must be between"):
        APIConfig(port=70000)


def test_api_config_debug_parsing():
    """Test debug boolean parsing"""
    assert APIConfig(debug="true").debug is True
    assert APIConfig(debug="false").debug is False
    assert APIConfig(debug="1").debug is True
    assert APIConfig(debug="yes").debug is True
    assert APIConfig(debug=True).debug is True


def test_app_config_initialization(mock_env_vars):
    """Test AppConfig combines all config sections"""
    config = AppConfig()

    assert isinstance(config.vertex_ai, VertexAIConfig)
    assert isinstance(config.bigquery, BigQueryConfig)
    assert isinstance(config.storage, StorageConfig)
    assert isinstance(config.api, APIConfig)

    assert config.vertex_ai.project_id == "test-project"
    assert config.bigquery.dataset_id == "test_dataset"


def test_config_manager_singleton(mock_env_vars):
    """Test ConfigManager singleton pattern"""
    config1 = ConfigManager.get_instance()
    config2 = ConfigManager.get_instance()

    assert config1 is config2
    assert isinstance(config1, AppConfig)


def test_config_manager_reset(mock_env_vars):
    """Test ConfigManager reset"""
    config1 = ConfigManager.get_instance()
    ConfigManager.reset()
    config2 = ConfigManager.get_instance()

    assert config1 is not config2
    assert isinstance(config2, AppConfig)


def test_get_config_convenience_function(mock_env_vars):
    """Test get_config() convenience function"""
    config = get_config()

    assert isinstance(config, AppConfig)
    assert config.vertex_ai.project_id == "test-project"
