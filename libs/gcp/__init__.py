"""
GCP service abstraction with factory pattern

모든 GCP 관련 작업은 이 모듈을 통해 수행됩니다.
다른 컴포넌트에서는 GCP SDK를 직접 사용하지 마세요.

Example:
    >>> from libs.gcp import GCPServiceFactory
    >>> from libs.utils.config import get_config
    >>> config = get_config()
    >>> factory = GCPServiceFactory()
    >>> bq = factory.get_bigquery_client(dataset_id=config.bigquery.dataset_id)
    >>> df = bq.get_prophet_features("corn", "2025-01-31", lookback_days=60)
"""

from .base import GCPServiceBase, GCPServiceFactory
from .bigquery import BigQueryService
from .storage import StorageService
from .query_params import (
    PriceQueryParams,
    ProphetFeaturesParams,
    NewsQueryParams,
    NewsForPredictionParams,
)

__all__ = [
    # Base
    "GCPServiceBase",
    "GCPServiceFactory",
    # Services
    "BigQueryService",
    "StorageService",
    # Query params
    "PriceQueryParams",
    "ProphetFeaturesParams",
    "NewsQueryParams",
    "NewsForPredictionParams",
]
