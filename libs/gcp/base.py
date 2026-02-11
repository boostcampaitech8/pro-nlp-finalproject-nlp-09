"""
Base classes and factory pattern for GCP services

This module provides the foundation for all GCP service abstractions,
including credential management and service instantiation.
"""

import os
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request


class GCPServiceBase(ABC):
    """
    Abstract base class for all GCP services

    Provides shared credential management and project ID resolution.
    All GCP service classes should inherit from this base class.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials: Optional[Credentials] = None,
        scopes: Optional[list] = None
    ):
        """
        Initialize GCP service base

        Args:
            project_id: GCP project ID (if None, will be resolved from env/gcloud)
            credentials: Pre-existing credentials (if None, will be created)
            scopes: OAuth scopes for the service (default: service-specific)
        """
        self.project_id = self._resolve_project_id(project_id)
        self.scopes = scopes or self._default_scopes()
        self.credentials = credentials or self._get_credentials()
        self._client = None

    @abstractmethod
    def _default_scopes(self) -> list:
        """Return default OAuth scopes for this service"""
        pass

    @abstractmethod
    def _initialize_client(self):
        """Initialize the service-specific client"""
        pass

    def _resolve_project_id(self, project_id: Optional[str] = None) -> str:
        """
        Resolve GCP project ID from multiple sources

        Resolution chain:
        1. Parameter passed to __init__
        2. VERTEX_AI_PROJECT_ID environment variable
        3. GOOGLE_CLOUD_PROJECT environment variable
        4. gcloud config (default project)

        Args:
            project_id: Explicit project ID

        Returns:
            str: Resolved project ID

        Raises:
            ValueError: If project ID cannot be resolved
        """
        # 1. Explicit parameter
        if project_id:
            return project_id

        # 2. VERTEX_AI_PROJECT_ID environment variable
        env_project = os.getenv("VERTEX_AI_PROJECT_ID")
        if env_project:
            return env_project

        # 3. GOOGLE_CLOUD_PROJECT environment variable
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if gcp_project:
            return gcp_project

        # 4. gcloud config
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=True
            )
            gcloud_project = result.stdout.strip()
            if gcloud_project and gcloud_project != "(unset)":
                return gcloud_project
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        raise ValueError(
            "프로젝트 ID를 찾을 수 없습니다.\n"
            "다음 중 하나를 설정하세요:\n"
            "1. project_id 파라미터 전달\n"
            "2. VERTEX_AI_PROJECT_ID 환경변수 설정\n"
            "3. gcloud config set project YOUR_PROJECT_ID\n"
        )

    def _get_credentials(self) -> Credentials:
        """
        Get Google Cloud credentials with automatic refresh

        Returns:
            Credentials: Valid Google Cloud credentials

        Raises:
            Exception: If credential acquisition fails
        """
        credentials, _ = default(scopes=self.scopes)

        # Ensure credentials are valid (refresh if needed)
        if not credentials.valid:
            credentials.refresh(Request())

        return credentials

    def refresh_credentials(self) -> None:
        """Manually refresh credentials if needed"""
        if self.credentials and not self.credentials.valid:
            self.credentials.refresh(Request())

    @property
    def client(self):
        """Lazy-load and return the service client"""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client


class GCPServiceFactory:
    """
    Factory for creating GCP service instances

    This factory manages credential sharing across services
    to reduce authentication overhead.
    """

    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize the factory

        Args:
            project_id: Default project ID for all services
        """
        self.project_id = project_id
        self._credential_cache: Dict[tuple, Credentials] = {}

    def _get_cached_credentials(self, scopes: tuple) -> Credentials:
        """
        Get or create cached credentials for given scopes

        Args:
            scopes: OAuth scopes as a tuple (for hashability)

        Returns:
            Credentials: Valid credentials for the given scopes
        """
        if scopes not in self._credential_cache:
            credentials, _ = default(scopes=list(scopes))
            if not credentials.valid:
                credentials.refresh(Request())
            self._credential_cache[scopes] = credentials
        else:
            # Refresh if needed
            credentials = self._credential_cache[scopes]
            if not credentials.valid:
                credentials.refresh(Request())

        return credentials

    def get_bigquery_client(
        self,
        dataset_id: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Get BigQuery service instance

        Args:
            dataset_id: Default dataset ID
            project_id: Project ID (if None, uses factory default)

        Returns:
            BigQueryService: BigQuery service instance
        """
        from .bigquery import BigQueryService

        scopes = tuple(BigQueryService._default_scopes_static())
        credentials = self._get_cached_credentials(scopes)

        return BigQueryService(
            project_id=project_id or self.project_id,
            dataset_id=dataset_id,
            credentials=credentials
        )

    def get_storage_client(
        self,
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Get Cloud Storage service instance

        Args:
            bucket_name: Default bucket name
            project_id: Project ID (if None, uses factory default)

        Returns:
            StorageService: Cloud Storage service instance
        """
        from .storage import StorageService

        scopes = tuple(StorageService._default_scopes_static())
        credentials = self._get_cached_credentials(scopes)

        return StorageService(
            project_id=project_id or self.project_id,
            bucket_name=bucket_name,
            credentials=credentials
        )

    def get_vertex_ai_credentials(
        self,
        project_id: Optional[str] = None
    ) -> Tuple[str, Credentials]:
        """
        Get Vertex AI credentials and access token

        Args:
            project_id: Project ID (if None, uses factory default)

        Returns:
            Tuple[str, Credentials]: (project_id, credentials)
        """
        scopes = tuple(["https://www.googleapis.com/auth/cloud-platform"])
        credentials = self._get_cached_credentials(scopes)

        # Resolve project ID
        from .base import GCPServiceBase
        resolved_project_id = GCPServiceBase(
            project_id=project_id or self.project_id,
            credentials=credentials,
            scopes=list(scopes)
        ).project_id

        return resolved_project_id, credentials
