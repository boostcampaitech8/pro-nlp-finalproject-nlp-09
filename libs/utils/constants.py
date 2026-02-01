"""
순수 상수 정의 (보안과 무관한 값들)

이 모듈은 환경과 관계없이 동일한 상수 값들을 정의합니다.
환경별로 달라지는 값은 config.py에서 관리합니다.

주의:
    - 하드코딩된 프로젝트 ID, 데이터셋 ID 등은 절대 포함하지 않음
    - 모든 환경 의존 값은 config.py에서 로드
"""

from typing import FrozenSet


# =============================================================================
# 유효성 검증용 상수
# =============================================================================

VALID_COMMODITIES: FrozenSet[str] = frozenset({"corn", "wheat", "soybean"})
"""유효한 commodity 값 (SQL injection 방지용)"""

VALID_FILTER_STATUS: FrozenSet[str] = frozenset({"T", "F", "E"})
"""유효한 filter_status 값"""


# =============================================================================
# 날짜/시간 형식
# =============================================================================

DATE_FORMAT = "%Y-%m-%d"
"""날짜 형식 (YYYY-MM-DD)"""

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
"""날짜시간 형식"""


# =============================================================================
# 시계열 모델 관련 상수
# =============================================================================
DEFAULT_PROPHET_LOOKBACK_DAYS = 2555  # 7년치


# =============================================================================
# 뉴스 모델 관련 상수
# =============================================================================

FINBERT_MODEL_NAME = "ProsusAI/finbert"
"""FinBERT 모델명"""

DEFAULT_NEWS_LOOKBACK_DAYS = 7
"""뉴스 분석 기본 lookback 기간 (일)"""

ARTICLE_EMBEDDING_DIM = 512
"""기사 임베딩 차원"""

ENTITY_EMBEDDING_DIM = 1024
"""엔티티 임베딩 차원"""

TRIPLE_EMBEDDING_DIM = 1024
"""트리플 임베딩 차원"""


# =============================================================================
# Vertex AI / LLM 기본값
# =============================================================================

VERTEX_AI_LOCATION = "us-central1"
"""Vertex AI 기본 리전"""

GENERATE_MODEL_NAME = "openai/gpt-oss-20b-maas"
"""LLM 생성 모델명"""

GENERATE_MODEL_TEMPERATURE = 0.7
"""LLM 샘플링 온도"""

GENERATE_MODEL_MAX_TOKENS = 2048
"""LLM 최대 토큰 수"""

# =============================================================================
# API 서버 기본값
# =============================================================================

DEFAULT_API_HOST = "0.0.0.0"
"""API 서버 기본 호스트"""

DEFAULT_API_PORT = 8000
"""API 서버 기본 포트"""

DEFAULT_DEBUG = False
"""디버그 모드 기본값"""


# =============================================================================
# 데이터 조회 기본값
# =============================================================================

# TODO 환경에 따라 삭제 예정
# TODO 이 값을 모델 config에서 로드하는 방향으로 수정

# TODO 환경에 따라 삭제 예정
# MAX_QUERY_LIMIT = 10000
"""쿼리 결과 최대 제한"""


# =============================================================================
# BigQuery 테이블명
# =============================================================================


class Tables:
    """BigQuery 테이블명 상수"""

    # 가격 관련
    DAILY_PRICES = "daily_prices"
    STG_PRICES = "stg_prices"

    # 뉴스 관련
    NEWS_ARTICLES = "news_articles"
    NEWS_ARTICLE_TEXTS = "news_article_texts"
    NEWS_ARTICLE_ENRICHMENTS_RAW = "news_article_enrichments_raw"
    ARTICLE_EMBEDDINGS = "article_embeddings"
    ENTITY_EMBEDDINGS = "entity_embeddings"
    TRIPLE_EMBEDDINGS = "triple_embeddings"
    ARTICLE_ENTITY_MAP = "article_entity_map"
    ARTICLE_TRIPLE_MAP = "article_triple_map"

    # View
    NEWS_ARTICLES_FULL = "news_articles_full"
