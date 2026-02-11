"""
News Repository - 뉴스 데이터 접근 레이어

news_articles, news_entities, news_triples 테이블에 대한 읽기/쓰기 작업을 캡슐화합니다.
SQL 쿼리는 libs/gcp/sql/news/ 디렉토리에서 로드합니다.
"""

import logging
import pandas as pd
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..bigquery import BigQueryService


logger = logging.getLogger(__name__)


# 유효한 filter_status 값
VALID_FILTER_STATUS = {"T", "F", "E"}


class NewsRepository:
    """
    뉴스 데이터 Repository

    news_articles 테이블에 대한 CRUD 작업을 제공합니다.
    모든 메서드는 SQL 파일을 로드하여 실행합니다.

    Example:
        >>> from libs.gcp import GCPServiceFactory
        >>> factory = GCPServiceFactory()
        >>> bq = factory.get_bigquery_service()
        >>> repo = NewsRepository(bq)
        >>> df = repo.get_filtered_articles("2025-01-01", "2025-01-31")
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

    def save_prediction(self, prediction_data: Dict[str, Any], commodity: str) -> None:
        """
        뉴스 감성 분석 결과를 prediction_news_sentiment 테이블에 적재

        Args:
            prediction_data: 뉴스 모델의 반환값 (JSON/Dict)
            commodity: 상품명 (corn, soybean, wheat)
        """
        if not prediction_data or "error" in prediction_data:
            logger.error("Skipping save: Invalid news prediction data")
            return

        table_id = "prediction_news_sentiment"
        
        # JSON 필드 직렬화 및 commodity 추가
        row = prediction_data.copy()
        row["commodity"] = commodity
        
        if "features_summary" in row and isinstance(row["features_summary"], dict):
            row["features_summary"] = json.dumps(row["features_summary"], ensure_ascii=False)
        
        if "evidence_news" in row and isinstance(row["evidence_news"], list):
            row["evidence_news"] = json.dumps(row["evidence_news"], ensure_ascii=False)

        logger.info(f"Saving news prediction for {commodity} on date: {row.get('target_date')}")
        
        errors = self._bq.insert_rows_json(table_id, [row])
        
        if errors:
            raise RuntimeError(f"Failed to save news prediction for {commodity}: {errors}")

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

    def get_filtered_articles(
        self,
        start_date: str,
        end_date: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        필터링된 뉴스 기사 조회 (filter_status='T')

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            limit: 결과 제한 (optional)

        Returns:
            pd.DataFrame: 뉴스 기사 데이터
        """
        start_date = self._validate_date(start_date, "start_date")
        end_date = self._validate_date(end_date, "end_date")

        limit_clause = f"LIMIT {limit}" if limit else ""

        params = {
            **self._base_params(),
            "start_date": start_date,
            "end_date": end_date,
            "limit_clause": limit_clause,
        }

        logger.info(f"Getting filtered articles: {start_date} ~ {end_date}")
        return self._bq._load_and_execute_query("news.get_filtered_articles_by_date", params)

    def get_articles_for_prediction(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        예측용 뉴스 데이터 조회 (임베딩 포함)

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            pd.DataFrame: 예측 입력용 뉴스 데이터
                columns: id, publish_date, title, description,
                         article_embedding, triples, 감성점수들
        """
        start_date = self._validate_date(start_date, "start_date")
        end_date = self._validate_date(end_date, "end_date")

        params = {
            **self._base_params(),
            "start_date": start_date,
            "end_date": end_date,
        }

        logger.info(f"Getting articles for prediction: {start_date} ~ {end_date}")
        return self._bq._load_and_execute_query("news.get_articles_for_prediction", params)

    def get_articles_for_lookback(
        self,
        target_date: str,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """
        특정 날짜 기준 lookback 기간의 뉴스 조회

        감성 분석 모델 입력용입니다.

        Args:
            target_date: 타겟 날짜 (YYYY-MM-DD)
            lookback_days: lookback 일수 (기본값: 7)

        Returns:
            pd.DataFrame: 뉴스 기사 데이터
        """
        target_date = self._validate_date(target_date, "target_date")

        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = target_dt - pd.Timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        return self.get_articles_for_prediction(start_date, target_date)
