"""
BigQuery 데이터 가져오기 모듈
Google Cloud BigQuery에서 시계열, 뉴스, 가격 데이터를 가져옵니다.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.auth import default
from google.auth.transport.requests import Request

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config.settings import (
    VERTEX_AI_PROJECT_ID,
    BIGQUERY_DATASET_ID,
    BIGQUERY_TABLE_ID,
    BIGQUERY_DATE_COLUMN,
    BIGQUERY_VALUE_COLUMN,
    BIGQUERY_BASE_DATE,
    BIGQUERY_DAYS,
)


class BigQueryClient:
    """BigQuery 클라이언트"""

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        date_column: Optional[str] = None,
        value_column: Optional[Union[str, List[str]]] = None,
        base_date: Optional[str] = None,
        days: Optional[int] = None,
    ):
        """
        BigQuery 클라이언트 초기화
        """
        self.project_id = project_id or VERTEX_AI_PROJECT_ID
        if not self.project_id:
            raise ValueError("프로젝트 ID가 필요합니다.")

        self.dataset_id = dataset_id or BIGQUERY_DATASET_ID or "tilda"
        self.table_id = table_id or BIGQUERY_TABLE_ID
        self.date_column = date_column or BIGQUERY_DATE_COLUMN or "time"

        # value_column 설정
        if value_column is None:
            self.value_columns = [BIGQUERY_VALUE_COLUMN] if BIGQUERY_VALUE_COLUMN else ["value"]
        elif isinstance(value_column, str):
            self.value_columns = [value_column]
        else:
            self.value_columns = value_column

        self.base_date = base_date if base_date is not None else BIGQUERY_BASE_DATE
        self.days = days if days is not None else (BIGQUERY_DAYS or 30)

        # 인증 설정
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/bigquery"])
        if not credentials.valid:
            credentials.refresh(Request())

        self.client = bigquery.Client(project=self.project_id, credentials=credentials)

    def get_prophet_features(
        self,
        target_date: str,
        lookback_days: int = 1500,
        commodity: str = "corn",
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        date_column: str = "ds",
    ) -> pd.DataFrame:
        """
        Prophet/XGBoost 모델용 피처 데이터를 가져옵니다.
        """
        dataset = dataset_id or self.dataset_id
        if not table_id:
            table_id = f"prophet_{commodity}"

        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {target_date}")

        start_dt = target_dt - timedelta(days=lookback_days)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        query = f"""
            SELECT *
            FROM `{self.project_id}.{dataset}.{table_id}`
            WHERE {date_column} >= '{start_date_str}'
              AND {date_column} <= '{target_date}'
            ORDER BY {date_column} ASC
        """
        return self.client.query(query).to_dataframe()

    def get_news_for_prediction(
        self,
        target_date: str,
        lookback_days: int = 7,
        commodity: str = "corn",
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        filter_status: str = "T",
    ) -> pd.DataFrame:
        """
        품목별 뉴스 감성 예측용 데이터를 가져옵니다. 
        팀원이 추가한 앙상블 모델 파이프라인에서 요구하는 모든 컬럼을 포함합니다.
        """
        dataset = dataset_id or self.dataset_id
        if not table_id:
            table_id = f"{commodity}_all_news_with_sentiment"

        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {target_date}")

        start_dt = target_dt - timedelta(days=lookback_days)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        # 앙상블 모델용 전체 컬럼 조회
        query = f"""
            SELECT
                id, title, doc_url, all_text, authors, publish_date,
                meta_site_name, key_word, filter_status, description,
                named_entities, triples, article_embedding, combined_text,
                sentiment, sentiment_confidence, positive_score,
                negative_score, neutral_score, price_impact_score
            FROM `{self.project_id}.{dataset}.{table_id}`
            WHERE publish_date >= '{start_date_str}'
              AND publish_date <= '{target_date}'
              AND filter_status = '{filter_status}'
            ORDER BY publish_date ASC
        """
        return self.client.query(query).to_dataframe()

    def get_price_history(
        self,
        target_date: str,
        lookback_days: int = 30,
        commodity: str = "corn",
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        품목별 가격 데이터를 가져옵니다.
        """
        dataset = dataset_id or self.dataset_id
        if not table_id:
            table_id = f"{commodity}_price"

        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {target_date}")

        start_dt = target_dt - timedelta(days=lookback_days)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        query = f"""
            SELECT 
                time as date,
                time,
                close
            FROM `{self.project_id}.{dataset}.{table_id}`
            WHERE time >= '{start_date_str}'
              AND time <= '{target_date}'
            ORDER BY time ASC
        """
        return self.client.query(query).to_dataframe()

    def get_timeseries_data(
        self,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        date_column: Optional[str] = None,
        value_column: Optional[Union[str, List[str]]] = None,
        base_date: Optional[str] = None,
        days: Optional[int] = None,
        order_by: Optional[str] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        특정 날짜 기준으로 최근 N일치 시계열 데이터 가져오기 (일반 범용 조회용)
        """
        dataset_id = dataset_id or self.dataset_id
        table_id = table_id or self.table_id
        date_column = date_column or self.date_column

        if value_column is None:
            value_cols = self.value_columns
        elif isinstance(value_column, str):
            value_cols = [value_column]
        else:
            value_cols = value_column

        base_date = base_date if base_date is not None else self.base_date
        days = days if days is not None else 30

        if not dataset_id or not table_id:
            return []

        if base_date:
            base_dt = datetime.strptime(base_date, "%Y-%m-%d")
        else:
            base_dt = datetime.now()

        start_date = base_dt - timedelta(days=days - 1)
        
        select_clause = ", ".join([date_column] + value_cols)
        where_cond = f"{date_column} >= '{start_date.strftime('%Y-%m-%d')}' AND {date_column} <= '{base_dt.strftime('%Y-%m-%d')}'"
        if where_clause:
            where_cond += f" AND ({where_clause})"

        query = f"""
            SELECT {select_clause}
            FROM `{self.project_id}.{dataset_id}.{table_id}`
            WHERE {where_cond}
            ORDER BY {order_by or (date_column + ' ASC')}
            {f'LIMIT {limit}' if limit else ''}
        """
        
        results = self.client.query(query).result()
        data = []
        for row in results:
            item = {"date": row[date_column]}
            for col in value_cols:
                item[col] = row[col]
            data.append(item)
        return data


def get_bigquery_timeseries(
    dataset_id: Optional[str] = None,
    table_id: Optional[str] = None,
    project_id: Optional[str] = None,
    date_column: Optional[str] = None,
    value_column: Optional[Union[str, List[str]]] = None,
    base_date: Optional[str] = None,
    days: Optional[int] = None,
) -> Union[List[float], Dict[str, List[float]]]:
    """
    BigQuery에서 시계열 데이터 값 리스트를 가져오는 편의 함수
    """
    client = BigQueryClient(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        date_column=date_column,
        value_column=value_column,
        base_date=base_date,
        days=days,
    )
    
    data = client.get_timeseries_data(
        dataset_id=dataset_id,
        table_id=table_id,
        date_column=date_column,
        value_column=value_column,
        base_date=base_date,
        days=days,
    )

    if not data:
        return []

    # 첫 번째 행에서 컬럼 확인 (date 제외)
    first_row_keys = [k for k in data[0].keys() if k != "date"]

    if len(first_row_keys) == 1:
        col_name = first_row_keys[0]
        return [float(item[col_name]) for item in data]
    else:
        result = {}
        for col_name in first_row_keys:
            result[col_name] = [float(item[col_name]) for item in data]
        return result