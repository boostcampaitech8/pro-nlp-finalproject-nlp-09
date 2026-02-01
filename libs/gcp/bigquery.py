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
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from google.cloud import bigquery
from google.auth.credentials import Credentials

from .base import GCPServiceBase
from .sql import SQLQueryLoader


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
        logger.debug(f"BigQueryService initialized: project_id={project_id}, dataset_id={dataset_id}")

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
            logger.debug(f"Loaded query '{query_name}' with params: {list(params.keys())}")
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

    #     query = f"""
    #         SELECT
    #             {date_column} as ds,
    #             close as y,
    #             open,
    #             high,
    #             low,
    #             ema,
    #             volume
    #         FROM `{self.project_id}.{dataset}.{table_id}`
    #         WHERE commodity = '{commodity}'
    #           AND {date_column} >= '{start_date_str}'
    #           AND {date_column} <= '{target_date}'
    #         ORDER BY {date_column} ASC
    #     """

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
        Get generic time-series data

        Args:
            dataset_id: Dataset ID
            table_id: Table ID
            date_column: Date column name
            value_columns: Column(s) to retrieve (string or list)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            where_clause: Additional WHERE conditions
            order_by: ORDER BY clause (default: date_column ASC)
            limit: Result limit

        Returns:
            pd.DataFrame: Time-series data

        Example:
            >>> df = bq.get_timeseries_data(
            ...     table_id="corn_price",
            ...     date_column="time",
            ...     value_columns=["close", "volume"],
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-20"
            ... )
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
        order_sql = order_by if order_by else (f"{date_column} ASC" if date_column else "")
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
        return self._load_and_execute_query("prices.get_timeseries_data", params)

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
        Get time-series values as a simple list

        Args:
            dataset_id: Dataset ID
            table_id: Table ID
            date_column: Date column name
            value_column: Value column name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List[float]: Values only

        Example:
            >>> values = bq.get_timeseries_values(
            ...     table_id="corn_price",
            ...     date_column="time",
            ...     value_column="close",
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-20"
            ... )
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

    def test_read_daily_prices(
        self,
        commodity: str = "corn",
        limit: int = 10,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Test query to verify daily_prices table connectivity (safe with LIMIT)

        This is a safe test method that only fetches a limited number of rows.
        Use this to verify BigQuery connectivity without scanning too much data.
        SQL 파일: prices/test_read_daily_prices.sql

        Args:
            commodity: Commodity name (default: 'corn')
            limit: Number of rows to fetch (default: 10)
            dataset_id: Dataset ID (if None, uses instance default)

        Returns:
            pd.DataFrame: Sample data from daily_prices table

        Example:
            >>> df = bq.test_read_daily_prices(commodity="corn", limit=5)
            >>> print(df.head())
        """
        dataset = dataset_id or self.dataset_id
        if not dataset:
            raise ValueError("dataset_id is required")

        params = {
            "project_id": self.project_id,
            "dataset_id": dataset,
            "commodity": commodity,
            "limit": limit,
        }

        logger.info(f"Testing read from daily_prices: commodity={commodity}, limit={limit}")
        return self._load_and_execute_query("prices.test_read_daily_prices", params)

    def get_daily_prices(
        self,
        commodity: str,
        start_date: str,
        end_date: str,
        dataset_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get price data from daily_prices table for a specific commodity

        SQL 파일: prices/get_price_history.sql (동일 쿼리 재사용)

        Args:
            commodity: Commodity name (e.g., 'corn', 'wheat', 'soybean')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            dataset_id: Dataset ID (if None, uses instance default)

        Returns:
            pd.DataFrame: Price data with all columns

        Example:
            >>> df = bq.get_daily_prices(
            ...     commodity="corn",
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-20"
            ... )
        """
        dataset = dataset_id or self.dataset_id
        if not dataset:
            raise ValueError("dataset_id is required")

        # Validate date formats
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        params = {
            "project_id": self.project_id,
            "dataset_id": dataset,
            "commodity": commodity,
            "start_date": start_date,
            "end_date": end_date,
        }

        logger.info(f"Getting daily prices: commodity={commodity}, date_range={start_date} to {end_date}")
        return self._load_and_execute_query("prices.get_price_history", params)

    # def get_filtered_articles(
    #     self,
    #     start_date: Optional[str] = None,
    #     end_date: Optional[str] = None,
    #     limit: Optional[int] = None,
    #     dataset_id: Optional[str] = None,
    #     table_id: str = "news_article",
    # ) -> pd.DataFrame:
    #     """
    #     Get filtered news articles (filter_status='T')

    #     SQL 파일: news/get_filtered_articles.sql

    #     Args:
    #         start_date: Start date (YYYY-MM-DD, optional)
    #         end_date: End date (YYYY-MM-DD, optional)
    #         limit: Result limit (optional)
    #         dataset_id: Dataset ID (if None, uses instance default)
    #         table_id: News table ID (default 'news_article')

    #     Returns:
    #         pd.DataFrame: Filtered news articles

    #     Example:
    #         >>> df = bq.get_filtered_articles(
    #         ...     start_date="2025-01-01",
    #         ...     end_date="2025-01-20",
    #         ...     limit=100
    #         ... )
    #     """
    #     dataset = dataset_id or self.dataset_id
    #     if not dataset:
    #         raise ValueError("dataset_id is required")

    #     # Build optional date filter
    #     date_conditions = []
    #     if start_date:
    #         date_conditions.append(f"AND publish_date >= '{start_date}'")
    #     if end_date:
    #         date_conditions.append(f"AND publish_date <= '{end_date}'")
    #     date_filter = " ".join(date_conditions)

    #     # Build optional limit clause
    #     limit_clause = f"LIMIT {limit}" if limit else ""

    #     params = {
    #         "project_id": self.project_id,
    #         "dataset_id": dataset,
    #         "table_id": table_id,
    #         "date_filter": date_filter,
    #         "limit_clause": limit_clause,
    #     }

    #     logger.info(
    #         f"Getting filtered articles: "
    #         f"date_range={start_date or 'N/A'} to {end_date or 'N/A'}, "
    #         f"limit={limit or 'N/A'}"
    #     )
    #     return self._load_and_execute_query("news.get_filtered_articles", params)

    def list_available_queries(self, domain: Optional[str] = None) -> Dict[str, list]:
        """
        List available SQL query files

        Args:
            domain: Filter by domain ('prices', 'news') - if None, lists all

        Returns:
            Dict[str, list]: Dictionary mapping domain to query names

        Example:
            >>> queries = bq.list_available_queries()
            >>> print(queries)
            {'prices': ['get_prophet_features', 'get_price_history', ...], ...}
        """
        return self._sql_loader.list_queries(domain)
