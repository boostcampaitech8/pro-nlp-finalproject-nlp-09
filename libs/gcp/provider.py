# libs/gcp/providers.py
import logging
from typing import Optional
from libs.utils.config import get_config
from libs.gcp.base import GCPServiceFactory

logger = logging.getLogger(__name__)
_bq = None


def get_bq_service():
    global _bq
    if _bq is None:
        config = get_config()
        factory = GCPServiceFactory()
        _bq = factory.get_bigquery_client(dataset_id=config.bigquery.dataset_id)
        logger.debug("BigQueryService singleton initialized")
    return _bq
