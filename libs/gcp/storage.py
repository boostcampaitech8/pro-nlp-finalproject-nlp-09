"""
Google Cloud Storage service abstraction

This module provides a clean interface for GCS operations,
including file upload, download, and listing.
"""

from pathlib import Path
from typing import Optional, List
from google.cloud import storage
from google.auth.credentials import Credentials

from .base import GCPServiceBase


class StorageService(GCPServiceBase):
    """
    Google Cloud Storage service with factory pattern integration

    This class provides basic GCS operations for file management.
    Future use cases include model artifact storage, data export/import,
    and report archiving.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        bucket_name: Optional[str] = None,
        credentials: Optional[Credentials] = None
    ):
        """
        Initialize Cloud Storage service

        Args:
            project_id: GCP project ID
            bucket_name: Default bucket name
            credentials: Pre-existing credentials (optional)
        """
        super().__init__(project_id=project_id, credentials=credentials)
        self.bucket_name = bucket_name

    def _default_scopes(self) -> list:
        """Default OAuth scopes for Cloud Storage"""
        return ["https://www.googleapis.com/auth/devstorage.read_write"]

    @staticmethod
    def _default_scopes_static() -> list:
        """Static version for factory use"""
        return ["https://www.googleapis.com/auth/devstorage.read_write"]

    def _initialize_client(self):
        """Initialize Cloud Storage client"""
        return storage.Client(
            project=self.project_id,
            credentials=self.credentials
        )

    def _get_bucket(self, bucket_name: Optional[str] = None):
        """Get bucket object"""
        name = bucket_name or self.bucket_name
        if not name:
            raise ValueError("bucket_name is required")
        return self.client.bucket(name)

    def upload_file(
        self,
        source_path: str,
        destination_blob: str,
        bucket_name: Optional[str] = None
    ) -> None:
        """
        Upload a file to GCS

        Args:
            source_path: Local file path
            destination_blob: GCS blob name (path in bucket)
            bucket_name: Bucket name (if None, uses instance default)

        Example:
            >>> storage.upload_file(
            ...     "model.pkl",
            ...     "models/xgboost_model.pkl"
            ... )
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(str(source))

    def download_file(
        self,
        source_blob: str,
        destination_path: str,
        bucket_name: Optional[str] = None
    ) -> None:
        """
        Download a file from GCS

        Args:
            source_blob: GCS blob name (path in bucket)
            destination_path: Local file path
            bucket_name: Bucket name (if None, uses instance default)

        Example:
            >>> storage.download_file(
            ...     "models/xgboost_model.pkl",
            ...     "model.pkl"
            ... )
        """
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(str(destination))

    def list_blobs(
        self,
        prefix: Optional[str] = None,
        bucket_name: Optional[str] = None
    ) -> List[str]:
        """
        List blobs in a bucket

        Args:
            prefix: Filter blobs by prefix (e.g., "models/")
            bucket_name: Bucket name (if None, uses instance default)

        Returns:
            List[str]: List of blob names

        Example:
            >>> blobs = storage.list_blobs(prefix="models/")
            >>> print(blobs)
            ['models/model1.pkl', 'models/model2.pkl']
        """
        bucket = self._get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]

    def blob_exists(
        self,
        blob_name: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        Check if a blob exists

        Args:
            blob_name: GCS blob name
            bucket_name: Bucket name (if None, uses instance default)

        Returns:
            bool: True if blob exists

        Example:
            >>> if storage.blob_exists("models/model.pkl"):
            ...     print("Model exists")
        """
        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()

    def delete_blob(
        self,
        blob_name: str,
        bucket_name: Optional[str] = None
    ) -> None:
        """
        Delete a blob from GCS

        Args:
            blob_name: GCS blob name
            bucket_name: Bucket name (if None, uses instance default)

        Example:
            >>> storage.delete_blob("models/old_model.pkl")
        """
        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
