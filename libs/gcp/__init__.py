"""
GCP service abstraction with factory pattern
"""

from .base import GCPServiceBase, GCPServiceFactory
from .bigquery import BigQueryService
from .storage import StorageService

__all__ = [
    "GCPServiceBase",
    "GCPServiceFactory",
    "BigQueryService",
    "StorageService",
]
