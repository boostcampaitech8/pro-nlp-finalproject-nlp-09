"""
BigQuery service abstraction

이 모듈은 BigQuery 작업을 위한 클린 인터페이스를 제공합니다.
SQL 쿼리는 libs/gcp/sql/ 디렉토리의 파일에서 로드됩니다.

Example:
    >>> from libs.gcp import GCPServiceFactory
    >>> from libs.utils.config import get_config
    >>> config = get_config()
    >>> factory = GCPServiceFactory()
    >>> bq = factory.get_bigquery_client(dataset_id=config.bigquery.dataset_id)
    >>>
    >>> # SQL 파일 기반 쿼리 실행
    >>> df = bq.execute("prices.get_prophet_features",
    ...                 commodity="corn", start_date="2025-01-01", end_date="2025-01-31")
    >>>
    >>> # 편의 메서드 사용
    >>> df = bq.get_prophet_features("corn", "2025-01-31", lookback_days=60)
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from google.cloud import bigquery
from google.auth.credentials import Credentials

from .base import GCPServiceBase
from .sql import SQLQueryLoader
from .query_params import (
    PriceQueryParams,
    ProphetFeaturesParams,
    NewsQueryParams,
    NewsForPredictionParams,
)
from libs.utils.constants import (
    VALID_COMMODITIES,
    VALID_FILTER_STATUS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    DATE_FORMAT,
)


logger = logging.getLogger(__name__)


class BigQueryService(GCPServiceBase):
    """
    BigQuery 서비스 - SQL 파일 기반 쿼리 실행

    이 클래스는 BigQuery 작업을 위한 클린 인터페이스를 제공합니다.
    SQL 쿼리는 파일에서 로드되고 파라미터화되어 실행됩니다.

    Attributes:
        project_id: GCP 프로젝트 ID
        dataset_id: 기본 데이터셋 ID
        credentials: GCP 인증 정보

    Example:
        >>> bq = BigQueryService(project_id="my-project", dataset_id="market")
        >>> df = bq.execute("prices.get_price_history",
        ...                 commodity="corn", start_date="2025-01-01", end_date="2025-01-31")
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        credentials: Optional[Credentials] = None,
    ):
        """
        BigQuery 서비스 초기화

        Args:
            project_id: GCP 프로젝트 ID
            dataset_id: 쿼리에 사용할 기본 데이터셋 ID
            credentials: 사전 생성된 인증 정보 (선택)
        """
        super().__init__(project_id=project_id, credentials=credentials)
        self.dataset_id = dataset_id
        self._sql_loader = SQLQueryLoader()
        logger.debug(
            f"BigQueryService initialized: project_id={project_id}, dataset_id={dataset_id}"
        )

    def _default_scopes(self) -> list:
        """BigQuery용 기본 OAuth 스코프"""
        return ["https://www.googleapis.com/auth/bigquery"]

    @staticmethod
    def _default_scopes_static() -> list:
        """팩토리에서 사용하는 정적 버전"""
        return ["https://www.googleapis.com/auth/bigquery"]

    def _initialize_client(self):
        """BigQuery 클라이언트 초기화"""
        return bigquery.Client(project=self.project_id, credentials=self.credentials)

    # =========================================================================
    # Core Methods (외부 인터페이스)
    # =========================================================================

    def execute(self, query_name: str, **params) -> pd.DataFrame:
        """
        SQL 파일을 로드하고 실행

        Args:
            query_name: "domain.query_file" 형식 (예: "prices.get_prophet_features")
            **params: 쿼리 파라미터

        Returns:
            pd.DataFrame: 쿼리 결과

        Raises:
            FileNotFoundError: SQL 파일이 없을 경우
            ValueError: 쿼리 실행 중 에러 발생시

        Example:
            >>> bq.execute("prices.get_prophet_features",
            ...            commodity="corn", start_date="2025-01-01", end_date="2025-01-31")
        """
        # 기본 파라미터 추가
        if "project_id" not in params:
            params["project_id"] = self.project_id
        if "dataset_id" not in params and self.dataset_id:
            params["dataset_id"] = self.dataset_id

        if params:
            query = self._sql_loader.load_with_params(query_name, **params)
            logger.debug(
                f"Loaded query '{query_name}' with params: {list(params.keys())}"
            )
        else:
            query = self._sql_loader.load(query_name)
            logger.debug(f"Loaded query '{query_name}' without params")

        return self.execute_raw(query)

    # TODO 메모리에 올라가기에 너무 큰 데이터를 받는다면 청킹, 스트리밍 방식으로 처리해야할 것 같음
    def execute_raw(self, query: str) -> pd.DataFrame:
        """
        SQL 문자열 직접 실행 (내부/테스트용)

        Args:
            query: 실행할 SQL 쿼리

        Returns:
            pd.DataFrame: 쿼리 결과

        Note:
            가능하면 execute() 메서드를 사용하세요.
            이 메서드는 SQL 파일 기반이 아닌 직접 쿼리가 필요한 경우에만 사용합니다.
        """
        logger.debug(f"Executing query: {query[:100]}...")
        job = self.client.query(query)
        return job.to_dataframe()

    def list_queries(self, domain: Optional[str] = None) -> Dict[str, list]:
        """
        사용 가능한 쿼리 목록 조회

        Args:
            domain: 도메인으로 필터링 ('prices', 'news') - None이면 전체 조회

        Returns:
            Dict[str, list]: 도메인별 쿼리 이름 딕셔너리

        Example:
            >>> queries = bq.list_queries()
            >>> print(queries)
            {'prices': ['get_prophet_features', 'get_price_history', ...], ...}
        """
        return self._sql_loader.list_queries(domain)

    # =========================================================================
    # Convenience Methods - 가격 데이터
    # =========================================================================

    def get_daily_prices(
        self,
        commodity: str,
        start_date: str,
        end_date: str,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        가격 데이터 조회 (prices.get_price_history.sql)

        Args:
            commodity: 상품명 (corn, wheat, soybean)
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            dataset_id: 데이터셋 ID (없으면 인스턴스 기본값 사용)

        Returns:
            pd.DataFrame: 가격 데이터
                columns: commodity, date, open, high, low, close, ema, volume, ingested_at

        Raises:
            ValueError: 유효하지 않은 commodity 또는 날짜 형식

        Example:
            >>> df = bq.get_daily_prices(
            ...     commodity="corn",
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31"
            ... )
        """
        # 파라미터 검증
        params = PriceQueryParams(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            f"Getting daily prices: {params.commodity}, {params.start_date} ~ {params.end_date}"
        )

        return self.execute(
            "prices.get_price_history",
            dataset_id=dataset_id or self.dataset_id,
            commodity=params.commodity,
            start_date=params.start_date,
            end_date=params.end_date,
        )

    def get_prophet_features(
        self,
        commodity: str,
        target_date: str,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prophet 피처 조회 (prices.get_prophet_features.sql)

        target_date를 기준으로 lookback_days만큼의 과거 데이터를 조회합니다.

        Args:
            commodity: 상품명 (corn, wheat, soybean)
            target_date: 타겟 날짜 (YYYY-MM-DD)
            lookback_days: lookback 기간 (일 단위, 기본값 60)
            dataset_id: 데이터셋 ID (없으면 인스턴스 기본값 사용)

        Returns:
            pd.DataFrame: Prophet 형식 데이터
                columns: ds (날짜), y (close), open, high, low, ema, volume

        Example:
            >>> df = bq.get_prophet_features(
            ...     commodity="corn",
            ...     target_date="2025-01-31",
            ...     lookback_days=90
            ... )
        """
        # 파라미터 검증
        params = ProphetFeaturesParams(
            commodity=commodity,
            target_date=target_date,
            lookback_days=lookback_days,
        )

        # 시작 날짜 계산
        target_dt = datetime.strptime(params.target_date, DATE_FORMAT)
        start_dt = target_dt - timedelta(days=params.lookback_days)
        start_date = start_dt.strftime(DATE_FORMAT)

        logger.info(
            f"Getting prophet features: {params.commodity}, {start_date} ~ {params.target_date}"
        )

        return self.execute(
            "prices.get_prophet_features",
            dataset_id=dataset_id or self.dataset_id,
            commodity=params.commodity,
            start_date=start_date,
            end_date=params.target_date,
        )

    def get_price_history(
        self,
        commodity: str,
        target_date: str,
        lookback_days: int = 30,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        가격 히스토리 조회 (뉴스 감성 분석 모델용)

        target_date를 기준으로 lookback_days만큼의 과거 가격 데이터를 조회합니다.

        Args:
            commodity: 상품명 (corn, wheat, soybean)
            target_date: 타겟 날짜 (YYYY-MM-DD)
            lookback_days: lookback 기간 (일 단위, 기본값 30)
            dataset_id: 데이터셋 ID (없으면 인스턴스 기본값 사용)

        Returns:
            pd.DataFrame: 가격 데이터

        Example:
            >>> df = bq.get_price_history(
            ...     commodity="corn",
            ...     target_date="2025-01-31",
            ...     lookback_days=30
            ... )
        """
        target_dt = datetime.strptime(target_date, DATE_FORMAT)
        start_dt = target_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime(DATE_FORMAT)

        return self.get_daily_prices(
            commodity=commodity,
            start_date=start_date,
            end_date=target_date,
            dataset_id=dataset_id,
        )

    # =========================================================================
    # Convenience Methods - 뉴스 데이터
    # =========================================================================

    def get_news_articles(
        self,
        start_date: str,
        end_date: str,
        filter_status: str = "T",
        limit: Optional[int] = None,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        뉴스 기사 조회 (news.get_articles.sql)

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            filter_status: 필터 상태 (T/F/E, 기본값 'T')
            limit: 결과 제한 (선택)
            dataset_id: 데이터셋 ID (없으면 인스턴스 기본값 사용)

        Returns:
            pd.DataFrame: 뉴스 기사 데이터

        Example:
            >>> df = bq.get_news_articles(
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-07",
            ...     filter_status="T"
            ... )
        """
        # 파라미터 검증
        params = NewsQueryParams(
            start_date=start_date,
            end_date=end_date,
            filter_status=filter_status,
            limit=limit,
        )

        limit_clause = f"LIMIT {params.limit}" if params.limit else ""

        logger.info(
            f"Getting news articles: {params.start_date} ~ {params.end_date}, status={params.filter_status}"
        )

        return self.execute(
            "news.get_articles",
            dataset_id=dataset_id or self.dataset_id,
            start_date=params.start_date,
            end_date=params.end_date,
            filter_status=params.filter_status,
            limit_clause=limit_clause,
        )

    def get_news_for_prediction(
        self,
        target_date: str,
        lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        예측용 뉴스 데이터 조회 (news.get_articles_for_prediction.sql)

        뉴스 기사 메타 + 본문 + 임베딩 + enrichment 정보를 함께 조회합니다.

        Args:
            target_date: 타겟 날짜 (YYYY-MM-DD)
            lookback_days: lookback 기간 (일 단위, 기본값 7)
            dataset_id: 데이터셋 ID (없으면 인스턴스 기본값 사용)

        Returns:
            pd.DataFrame: 예측 입력용 뉴스 데이터
                columns: article_id, publish_date, title, description, key_word,
                         all_text, article_embedding, named_entities_json, triples_json

        Example:
            >>> df = bq.get_news_for_prediction(
            ...     target_date="2025-01-31",
            ...     lookback_days=7
            ... )
        """
        # 파라미터 검증
        params = NewsForPredictionParams(
            target_date=target_date,
            lookback_days=lookback_days,
        )

        # 시작 날짜 계산
        target_dt = datetime.strptime(params.target_date, DATE_FORMAT)
        start_dt = target_dt - timedelta(days=params.lookback_days)
        start_date = start_dt.strftime(DATE_FORMAT)

        logger.info(f"Getting news for prediction: {start_date} ~ {params.target_date}")

        return self.execute(
            "news.get_articles_for_prediction",
            dataset_id=dataset_id or self.dataset_id,
            start_date=start_date,
            end_date=params.target_date,
        )

    # =========================================================================
    # Legacy Methods (하위 호환성)
    # =========================================================================

    def _load_and_execute_query(
        self,
        query_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Legacy: SQL 파일을 로드하고 파라미터와 조합하여 쿼리 실행

        Note:
            이 메서드는 하위 호환성을 위해 유지됩니다.
            새 코드에서는 execute() 메서드를 사용하세요.
        """
        if params:
            return self.execute(query_name, **params)
        else:
            return self.execute(query_name)

    def test_read_daily_prices(
        self,
        commodity: str = "corn",
        limit: int = 10,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        테스트용 가격 데이터 조회 (LIMIT으로 안전하게 실행)

        BigQuery 연결 확인용 테스트 메서드입니다.

        Args:
            commodity: 상품명 (기본값: 'corn')
            limit: 조회 행 수 (기본값: 10)
            dataset_id: 데이터셋 ID (없으면 인스턴스 기본값 사용)

        Returns:
            pd.DataFrame: 샘플 가격 데이터
        """
        # 기본 commodity 검증
        if commodity.lower() not in VALID_COMMODITIES:
            raise ValueError(
                f"Invalid commodity: {commodity}. Must be one of: {sorted(VALID_COMMODITIES)}"
            )

        logger.info(
            f"Testing read from daily_prices: commodity={commodity}, limit={limit}"
        )

        return self.execute(
            "prices.test_read_daily_prices",
            dataset_id=dataset_id or self.dataset_id,
            commodity=commodity.lower(),
            limit=limit,
        )

    def get_timeseries_data(
        self,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        date_column: Optional[str] = None,
        value_columns: Optional[Union[str, List[str]]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        범용 시계열 데이터 조회

        Note:
            새 코드에서는 get_daily_prices() 또는 get_prophet_features()를 사용하세요.
            이 메서드는 하위 호환성을 위해 유지됩니다.
        """
        dataset = dataset_id or self.dataset_id
        if not dataset or not table_id:
            raise ValueError("dataset_id and table_id are required")

        # Handle value_columns
        if value_columns is None:
            select_cols = "*"
        elif isinstance(value_columns, str):
            if date_column:
                select_cols = f"{date_column}, {value_columns}"
            else:
                select_cols = value_columns
        elif isinstance(value_columns, list):
            cols = value_columns.copy()
            if date_column and date_column not in cols:
                cols.insert(0, date_column)
            select_cols = ", ".join(cols)
        else:
            raise TypeError("value_columns must be string or list")

        # Build WHERE clause
        where_conditions = []
        if date_column and start_date:
            where_conditions.append(f"{date_column} >= '{start_date}'")
        if date_column and end_date:
            where_conditions.append(f"{date_column} <= '{end_date}'")
        if where_clause:
            where_conditions.append(where_clause)

        where_sql = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Build ORDER BY clause
        order_sql = (
            order_by if order_by else (f"{date_column} ASC" if date_column else "")
        )
        order_clause = f"ORDER BY {order_sql}" if order_sql else ""

        # Build LIMIT clause
        limit_clause = f"LIMIT {limit}" if limit else ""

        params = {
            "project_id": self.project_id,
            "dataset_id": dataset,
            "table_id": table_id,
            "select_columns": select_cols,
            "where_clause": where_sql,
            "order_by": order_clause,
            "limit": limit_clause,
        }

        logger.debug(f"Getting timeseries data from {dataset}.{table_id}")
        return self.execute("prices.get_timeseries_data", **params)

    def get_timeseries_values(
        self,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[float]:
        """
        시계열 값만 리스트로 반환

        Note:
            새 코드에서는 get_daily_prices() 사용 후 필요한 컬럼을 추출하세요.
            이 메서드는 하위 호환성을 위해 유지됩니다.
        """
        if not value_column:
            raise ValueError("value_column is required")

        df = self.get_timeseries_data(
            dataset_id=dataset_id,
            table_id=table_id,
            date_column=date_column,
            value_columns=value_column,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            return []

        return df[value_column].tolist()

    def list_available_queries(self, domain: Optional[str] = None) -> Dict[str, list]:
        """
        Legacy: list_queries()의 별칭

        Note:
            새 코드에서는 list_queries()를 사용하세요.
        """
        return self.list_queries(domain)
