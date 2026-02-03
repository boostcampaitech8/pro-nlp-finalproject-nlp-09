"""
Tests for Vertex AI helper utilities

Tests helper functions for Vertex AI authentication and configuration.
"""

import pytest
from unittest.mock import Mock, patch
from libs.utils.vertex_ai_helper import (
    get_vertex_ai_token,
    build_vertex_ai_url,
    get_vertex_ai_config,
    refresh_vertex_ai_token,
)


@patch("libs.utils.vertex_ai_helper.default")
def test_get_vertex_ai_token(mock_default, mock_credentials):
    """Test get_vertex_ai_token returns valid token"""
    mock_default.return_value = (mock_credentials, None)

    token = get_vertex_ai_token()

    assert token == "mock-access-token"
    mock_default.assert_called_once()


@patch("libs.utils.vertex_ai_helper.default")
def test_get_vertex_ai_token_refresh(mock_default):
    """Test token refresh when credentials are invalid"""
    mock_creds = Mock()
    mock_creds.valid = False
    mock_creds.token = "refreshed-token"
    mock_creds.refresh = Mock()

    mock_default.return_value = (mock_creds, None)

    token = get_vertex_ai_token()

    assert token == "refreshed-token"
    mock_creds.refresh.assert_called_once()


def test_build_vertex_ai_url(mock_env_vars):
    """Test build_vertex_ai_url constructs correct URL"""
    url = build_vertex_ai_url()

    expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/openapi"
    assert url == expected


def test_build_vertex_ai_url_custom_params():
    """Test build_vertex_ai_url with custom parameters"""
    url = build_vertex_ai_url(project_id="custom-project", location="europe-west1")

    expected = (
        "https://europe-west1-aiplatform.googleapis.com/v1/"
        "projects/custom-project/locations/europe-west1/endpoints/openapi"
    )
    assert url == expected


def test_build_vertex_ai_url_missing_project():
    """Test build_vertex_ai_url raises error without project_id"""
    with pytest.raises(ValueError, match="project_id is required"):
        build_vertex_ai_url()


@patch("libs.utils.vertex_ai_helper.get_vertex_ai_token")
def test_get_vertex_ai_config(mock_get_token, mock_env_vars):
    """Test get_vertex_ai_config returns correct configuration"""
    mock_get_token.return_value = "test-token"

    config = get_vertex_ai_config()

    assert config["model"] == "test-model"
    assert config["api_key"] == "test-token"
    assert config["temperature"] == 0.7
    assert config["max_tokens"] == 2048
    assert "base_url" in config
    assert "model_kwargs" in config


@patch("libs.utils.vertex_ai_helper.get_vertex_ai_token")
def test_get_vertex_ai_config_custom_params(mock_get_token, mock_env_vars):
    """Test get_vertex_ai_config with custom parameters"""
    mock_get_token.return_value = "test-token"

    config = get_vertex_ai_config(
        model_name="custom-model", temperature=0.5, max_tokens=1024
    )

    assert config["model"] == "custom-model"
    assert config["temperature"] == 0.5
    assert config["max_tokens"] == 1024


@patch("libs.utils.vertex_ai_helper.get_vertex_ai_token")
def test_refresh_vertex_ai_token(mock_get_token):
    """Test refresh_vertex_ai_token"""
    mock_get_token.return_value = "new-token"

    new_token = refresh_vertex_ai_token("old-token")

    assert new_token == "new-token"
    mock_get_token.assert_called_once()
