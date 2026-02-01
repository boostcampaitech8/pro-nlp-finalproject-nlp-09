"""
Vertex AI helper utilities

This module provides standalone helper functions for Vertex AI authentication
and configuration. These functions can be used to simplify Vertex AI integration
without modifying existing code.

Based on the authentication pattern from app/models/llm_summarizer.py.
"""

from typing import Optional, Dict, Any
from google.auth import default
from google.auth.transport.requests import Request

from .config import get_config


def get_vertex_ai_token(project_id: Optional[str] = None) -> str:
    """
    Get a fresh Vertex AI access token

    This function obtains Google Cloud credentials and returns a valid
    access token for Vertex AI API calls.

    Args:
        project_id: GCP project ID (optional, for validation)

    Returns:
        str: Valid access token

    Example:
        >>> token = get_vertex_ai_token()
        >>> headers = {"Authorization": f"Bearer {token}"}
    """
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    # Ensure credentials are valid (refresh if needed)
    if not credentials.valid:
        credentials.refresh(Request())

    return credentials.token


def build_vertex_ai_url(
    project_id: Optional[str] = None, location: Optional[str] = None
) -> str:
    """
    Build Vertex AI OpenAI-compatible API base URL

    Args:
        project_id: GCP project ID (if None, uses config)
        location: GCP location (if None, uses config, default: us-central1)

    Returns:
        str: Vertex AI base URL

    Example:
        >>> url = build_vertex_ai_url()
        >>> print(url)
        https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/openapi
    """
    config = get_config()

    pid = project_id or config.vertex_ai.project_id
    loc = location or config.vertex_ai.location

    if not pid:
        raise ValueError(
            "project_id is required. Set VERTEX_AI_PROJECT_ID environment variable or pass project_id parameter."
        )

    return f"https://{loc}-aiplatform.googleapis.com/v1/projects/{pid}/locations/{loc}/endpoints/openapi"


def get_vertex_ai_config(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get Vertex AI configuration for LangChain ChatOpenAI

    This function returns a configuration dictionary that can be used
    to initialize LangChain's ChatOpenAI with Vertex AI.

    Args:
        project_id: GCP project ID (if None, uses config)
        location: GCP location (if None, uses config)
        model_name: Model name (if None, uses config)
        temperature: Temperature (if None, uses config)
        max_tokens: Max tokens (if None, uses config)

    Returns:
        Dict[str, Any]: Configuration dictionary for ChatOpenAI

    Example:
        >>> config = get_vertex_ai_config()
        >>> llm = ChatOpenAI(**config)

        >>> # Or with custom parameters:
        >>> config = get_vertex_ai_config(
        ...     model_name="meta/llama-3.1-8b-instruct-maas",
        ...     temperature=0.5
        ... )
        >>> llm = ChatOpenAI(**config)
    """
    config = get_config()

    # Get access token
    token = get_vertex_ai_token(project_id)

    # Build base URL
    base_url = build_vertex_ai_url(project_id, location)

    # Get model parameters
    model = model_name or config.vertex_ai.model_name
    temp = temperature if temperature is not None else config.vertex_ai.temperature
    max_tok = max_tokens if max_tokens is not None else config.vertex_ai.max_tokens

    return {
        "model": model,
        "base_url": base_url,
        "api_key": token,
        "temperature": temp,
        "max_tokens": max_tok,
        "model_kwargs": {
            "parallel_tool_calls": False,
        },
    }


def refresh_vertex_ai_token(current_token: str) -> str:
    """
    Refresh Vertex AI access token

    This is a convenience function to get a fresh token.
    In most cases, you can just call get_vertex_ai_token() directly.

    Args:
        current_token: Current (possibly expired) token

    Returns:
        str: New valid access token

    Example:
        >>> new_token = refresh_vertex_ai_token(old_token)
    """
    return get_vertex_ai_token()
