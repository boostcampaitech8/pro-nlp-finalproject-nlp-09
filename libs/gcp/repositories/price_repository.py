"""
Price Repository - 가격 데이터 접근 레이어

daily_prices 테이블에 대한 읽기/쓰기 작업을 캡슐화합니다.
SQL 쿼리는 libs/gcp/sql/prices/ 디렉토리에서 로드합니다.
"""

import logging
import pandas as pd
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from ..bigquery import BigQueryService


logger = logging.getLogger(__name__)


# 허용된 commodity 값 (SQL injection 방지)
VALID_COMMODITIES = {"corn", "wheat", "soybean"}


class PriceRepository:
    """
    가격 데이터 Repository

    daily_prices 테이블에 대한 CRUD 작업을 제공합니다.
    모든 메서드는 SQL 파일을 로드하여 실행합니다.

    Example:
        >>> from libs.gcp import GCPServiceFactory
        >>> factory = GCPServiceFactory()
        >>> bq = factory.get_bigquery_service()
        >>> repo = PriceRepository(bq)
        >>> df = repo.get_prices("corn", "2025-01-01", "2025-01-31")
    """

    def __init__(self, bq_service: BigQueryService):
        """
        Repository 초기화

        Args:
            bq_service: BigQueryService 인스턴스
        """
        self._bq = bq_service
        self._project_id = bq_service.project_id
        self._dataset_id = bq_service.dataset_id

    def _validate_commodity(self, commodity: str) -> str:
        """commodity 값 검증 (SQL injection 방지)"""
        commodity = commodity.lower().strip()
        if commodity not in VALID_COMMODITIES:
            raise ValueError(f"Invalid commodity: {commodity}. Must be one of: {VALID_COMMODITIES}")
        return commodity

    def _validate_date(self, date_str: str, param_name: str = "date") -> str:
        """날짜 형식 검증"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            raise ValueError(f"Invalid {param_name} format: {date_str}. Use YYYY-MM-DD")

    def _base_params(self) -> Dict[str, str]:
        """공통 파라미터 반환"""
        return {
            "project_id": self._project_id,
            "dataset_id": self._dataset_id,
        }

    # =========================================================================
    # READ 메서드 - 데이터 조회
    # =========================================================================

    def get_prices(
        self,
        commodity: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        날짜 범위로 가격 데이터 조회

        Args:
            commodity: 상품명 (corn, wheat, soybean)
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            pd.DataFrame: 가격 데이터
                columns: commodity, date, open, high, low, close, ema, volume, ingested_at
        """
        commodity = self._validate_commodity(commodity)
        start_date = self._validate_date(start_date, "start_date")
        end_date = self._validate_date(end_date, "end_date")

        params = {
            **self._base_params(),
            "commodity": commodity,
            "start_date": start_date,
            "end_date": end_date,
        }

        logger.info(f"Getting prices: {commodity}, {start_date} ~ {end_date}")
        return self._bq._load_and_execute_query("prices.get_prices_by_date_range", params)

    def get_latest_prices(
        self,
        commodity: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        최근 N일간의 가격 데이터 조회

        Args:
            commodity: 상품명 (corn, wheat, soybean)
            days: 조회할 일수 (기본값: 30)

        Returns:
            pd.DataFrame: 최근 가격 데이터 (날짜 내림차순)
        """
        commodity = self._validate_commodity(commodity)
        if days <= 0:
            raise ValueError("days must be positive")

        params = {
            **self._base_params(),
            "commodity": commodity,
            "days": days,
        }

        logger.info(f"Getting latest {days} days of {commodity} prices")
        return self._bq._load_and_execute_query("prices.get_latest_prices", params)

    def get_price_at_date(
        self,
        commodity: str,
        target_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        특정 날짜의 가격 데이터 조회 (단일 레코드)

        Args:
            commodity: 상품명 (corn, wheat, soybean)
            target_date: 조회할 날짜 (YYYY-MM-DD)

        Returns:
            Optional[Dict]: 가격 데이터 딕셔너리, 없으면 None
        """
        commodity = self._validate_commodity(commodity)
        target_date = self._validate_date(target_date, "target_date")

        params = {
            **self._base_params(),
            "commodity": commodity,
            "target_date": target_date,
        }

        df = self._bq._load_and_execute_query("prices.get_price_at_date", params)
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_prophet_features(
        self,
        commodity: str,
        target_date: str,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """
        Prophet 모델용 피처 데이터 조회

        Args:
            commodity: 상품명 (corn, wheat, soybean)
            target_date: 타겟 날짜 (YYYY-MM-DD)
            lookback_days: lookback 일수 (기본값: 60)

        Returns:
            pd.DataFrame: Prophet 형식 데이터
                columns: ds (날짜), y (close), open, high, low, ema, volume
        """
        commodity = self._validate_commodity(commodity)
        target_date = self._validate_date(target_date, "target_date")

        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = target_dt - pd.Timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        params = {
            **self._base_params(),
            "commodity": commodity,
            "start_date": start_date,
            "end_date": target_date,
        }

        logger.info(f"Getting prophet features: {commodity}, {start_date} ~ {target_date}")
        return self._bq._load_and_execute_query("prices.get_prophet_features", params)

    # =========================================================================
    # WRITE 메서드 - 데이터 적재 (Airflow용)
    # =========================================================================

    def merge_from_staging(self) -> None:
        """
        스테이징 테이블에서 메인 테이블로 MERGE (Upsert)

        stg_prices의 데이터를 daily_prices로 병합합니다.
        - 기존 레코드: UPDATE
        - 신규 레코드: INSERT

        Note:
            Airflow DAG에서 호출됩니다.
        """
        params = self._base_params()

        logger.info("Merging staging data to daily_prices")
        query = self._bq._sql_loader.load_with_params("prices.merge_from_staging", **params)

        # DML 쿼리는 execute_query 대신 직접 실행
        job = self._bq.client.query(query)
        job.result()  # 완료 대기

        logger.info(f"Merge completed: {job.num_dml_affected_rows} rows affected")

    def truncate_staging(self) -> None:
        """
        스테이징 테이블 비우기

        MERGE 후 호출하여 스테이징 테이블을 비웁니다.

        Note:
            Airflow DAG에서 호출됩니다.
        """
        params = self._base_params()

        logger.info("Truncating staging table")
        query = self._bq._sql_loader.load_with_params("prices.truncate_staging", **params)

        job = self._bq.client.query(query)
        job.result()
        logger.info("Staging table truncated")

    def count_staging_rows(self) -> pd.DataFrame:
        """
        스테이징 테이블의 레코드 수 조회

        Returns:
            pd.DataFrame: commodity별 row_count, min_date, max_date
        """
        params = self._base_params()
        return self._bq._load_and_execute_query("prices.count_staging_rows", params)

    def load_to_staging(self, df: pd.DataFrame) -> None:
        """
        DataFrame을 스테이징 테이블에 적재

        Args:
            df: 적재할 데이터
                required columns: commodity, date, open, high, low, close, ema, volume

        Note:
            Airflow DAG에서 호출됩니다.
        """
        required_cols = {"commodity", "date", "open", "high", "low", "close", "ema", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # commodity 검증
        invalid_commodities = set(df["commodity"].unique()) - VALID_COMMODITIES
        if invalid_commodities:
            raise ValueError(f"Invalid commodities: {invalid_commodities}")

        table_ref = f"{self._project_id}.{self._dataset_id}.stg_prices"

        logger.info(f"Loading {len(df)} rows to staging table")

        job_config = self._bq.client.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        )

        job = self._bq.client.load_table_from_dataframe(
            df[list(required_cols)],
            table_ref,
            job_config=job_config,
        )
        job.result()

        logger.info(f"Loaded {job.output_rows} rows to {table_ref}")

    def save_prediction(self, prediction_data: Dict[str, Any], commodity: str) -> None:
        """
        시계열 예측 결과를 prediction_timeseries 테이블에 적재

        Args:
            prediction_data: 시계열 모델의 반환값 (JSON/Dict)
            commodity: 상품명 (corn, soybean, wheat)
        """
        if not prediction_data or "error" in prediction_data:
            logger.error("Skipping save: Invalid prediction data")
            return

        table_id = "prediction_timeseries"
        
        # 데이터에 commodity 추가
        row = prediction_data.copy()
        row["commodity"] = commodity
        
        logger.info(f"Saving timeseries prediction for {commodity} on date: {row.get('target_date')}")
        
        errors = self._bq.insert_rows_json(table_id, [row])
        
        if errors:
            raise RuntimeError(f"Failed to save timeseries prediction for {commodity}: {errors}")
