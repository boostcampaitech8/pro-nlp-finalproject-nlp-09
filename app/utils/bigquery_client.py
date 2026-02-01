"""
BigQuery 데이터 가져오기 모듈
Google Cloud BigQuery에서 시계열 데이터를 가져옵니다.
"""

import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가 (스크립트 직접 실행 시)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.auth import default
from google.auth.transport.requests import Request
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

        Args:
            project_id: GCP 프로젝트 ID (None이면 환경변수에서 가져옴)
            dataset_id: 데이터셋 ID (None이면 환경변수에서 가져옴)
            table_id: 테이블 ID (None이면 환경변수에서 가져옴)
            date_column: 날짜 컬럼명 (None이면 환경변수에서 가져옴, 기본값: "time")
            value_column: 값 컬럼명 (문자열 또는 리스트, None이면 환경변수에서 가져옴, 기본값: "value")
                - 문자열: 단일 컬럼 지정 (예: "price")
                - 리스트: 여러 컬럼 지정 (예: ["price", "volume", "market_cap"])
            base_date: 기준 날짜 (None이면 환경변수 또는 오늘)
            days: 기본 일수 (None이면 환경변수에서 가져옴, 기본값: 30)
        """
        self.project_id = project_id or VERTEX_AI_PROJECT_ID

        if not self.project_id:
            raise ValueError(
                "프로젝트 ID가 필요합니다. VERTEX_AI_PROJECT_ID 환경변수를 설정하거나 project_id를 전달하세요."
            )

        # 기본값 설정 (환경변수 또는 파라미터)
        self.dataset_id = dataset_id or BIGQUERY_DATASET_ID
        self.table_id = table_id or BIGQUERY_TABLE_ID
        self.date_column = date_column or BIGQUERY_DATE_COLUMN

        # value_column을 리스트로 통일 (문자열이면 리스트로 변환)
        if value_column is None:
            value_col = BIGQUERY_VALUE_COLUMN
            if value_col:
                # 쉼표로 구분된 문자열인 경우 리스트로 변환
                if "," in value_col:
                    self.value_columns = [col.strip() for col in value_col.split(",") if col.strip()]
                else:
                    self.value_columns = [value_col]
            else:
                self.value_columns = ["value"]
        elif isinstance(value_column, str):
            # 쉼표로 구분된 문자열인 경우 리스트로 변환
            if "," in value_column:
                self.value_columns = [col.strip() for col in value_column.split(",") if col.strip()]
            else:
                self.value_columns = [value_column]
        elif isinstance(value_column, list):
            self.value_columns = value_column
        else:
            raise TypeError(f"value_column은 문자열 또는 리스트여야 합니다. 받은 타입: {type(value_column)}")

        self.base_date = base_date if base_date is not None else BIGQUERY_BASE_DATE
        self.days = days if days is not None else BIGQUERY_DAYS

        # 인증 설정
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/bigquery"])
        if not credentials.valid:
            credentials.refresh(Request())

        self.client = bigquery.Client(project=self.project_id, credentials=credentials)

    def get_prophet_features(
        self,
        target_date: str,
        lookback_days: int = 60,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        date_column: str = "ds",
    ) -> pd.DataFrame:
        """
        Prophet/XGBoost 모델용 피처 데이터를 DataFrame 형태로 가져옵니다.
        지정된 날짜(target_date) 포함 과거 N일(lookback_days) 데이터를 조회합니다.

        Args:
            target_date: 기준 날짜 (YYYY-MM-DD)
            lookback_days: 과거 조회 기간 (일 단위, 기본값 60)
            dataset_id: 데이터셋 ID (옵션)
            table_id: 테이블 ID (옵션)
            date_column: 날짜 컬럼명 (기본값 'ds')

        Returns:
            pd.DataFrame: 피처 데이터프레임 (날짜 오름차순 정렬됨)
        """
        dataset = dataset_id or self.dataset_id
        table = table_id or self.table_id

        if not dataset or not table:
            raise ValueError("dataset_id와 table_id가 설정되지 않았습니다.")

        # 날짜 계산
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {target_date}")

        start_dt = target_dt - timedelta(days=lookback_days)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        query = f"""
            SELECT *
            FROM `{self.project_id}.{dataset}.{table}`
            WHERE {date_column} >= '{start_date_str}'
              AND {date_column} <= '{target_date}'
            ORDER BY {date_column} ASC
        """

        # DataFrame으로 반환 (db-dtypes 필요)
        return self.client.query(query).to_dataframe()

    def get_news_for_prediction(
        self, target_date: str, lookback_days: int = 7, dataset_id: Optional[str] = None, table_id: str = "news_article"
    ) -> pd.DataFrame:
        """
        뉴스 감성 모델 예측용 뉴스 데이터를 가져옵니다.

        Args:
            target_date: 기준 날짜 (YYYY-MM-DD)
            lookback_days: 조회할 과거 일수 (기본 7일)
            dataset_id: 데이터셋 ID
            table_id: 뉴스 테이블 ID

        Returns:
            pd.DataFrame: 뉴스 데이터 (publish_date, title, article_embedding, scores...)
        """
        dataset = dataset_id or self.dataset_id or BIGQUERY_DATASET_ID

        # 날짜 계산
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {target_date}")

        start_dt = target_dt - timedelta(days=lookback_days)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        # 필요한 컬럼들 조회
        query = f"""
            SELECT 
                publish_date, 
                title, 
                description as all_text,
                article_embedding,
                price_impact_score,
                sentiment_confidence,
                positive_score,
                negative_score,
                triples,
                filter_status
            FROM `{self.project_id}.{dataset}.{table_id}`
            WHERE publish_date >= '{start_date_str}'
              AND publish_date <= '{target_date}'
              AND filter_status = 'T'
            ORDER BY publish_date ASC
        """

        return self.client.query(query).to_dataframe()

    def get_price_history(
        self, target_date: str, lookback_days: int = 30, dataset_id: Optional[str] = None, table_id: str = "corn_price"
    ) -> pd.DataFrame:
        """
        뉴스 감성 모델 예측용 가격 데이터를 가져옵니다.

        Args:
            target_date: 기준 날짜
            lookback_days: 조회할 과거 일수 (기본 30일)
            dataset_id: 데이터셋 ID
            table_id: 가격 테이블 ID

        Returns:
            pd.DataFrame: 가격 데이터 (date, close, ret_1d)
        """
        dataset = dataset_id or self.dataset_id or BIGQUERY_DATASET_ID

        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {target_date}")

        start_dt = target_dt - timedelta(days=lookback_days)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        # time 컬럼을 date로 별칭 지정하여 호환성 확보
        # ret_1d는 테이블에 없을 수 있으므로 쿼리에서 제외 (preprocessing.py에서 자동 계산됨)
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
        특정 날짜 기준으로 최근 N일치 시계열 데이터 가져오기

        다른 테이블을 조회하려면 table_id와 value_column을 지정하면 됩니다.

        Args:
            dataset_id: 데이터셋 ID (None이면 초기화 시 설정한 값 사용)
            table_id: 테이블 ID (None이면 초기화 시 설정한 값 사용)
            date_column: 날짜 컬럼명 (None이면 초기화 시 설정한 값 사용)
            value_column: 값 컬럼명 (문자열 또는 리스트, None이면 초기화 시 설정한 값 사용)
                - 문자열: 단일 컬럼 지정 (예: "price")
                - 리스트: 여러 컬럼 지정 (예: ["price", "volume", "market_cap"])
                - 쉼표로 구분된 문자열도 가능 (예: "price,volume,market_cap")
            base_date: 기준 날짜 (YYYY-MM-DD 형식, None이면 초기화 시 설정한 값 또는 오늘)
            days: 가져올 일수 (None이면 초기화 시 설정한 값 사용)
            order_by: 정렬 기준 (None이면 "{date_column} ASC" 사용, 최신 데이터가 마지막에 옴)
            where_clause: 추가 WHERE 조건 (예: "filter_status = 'T'")
            limit: 최대 조회 개수 제한 (None이면 제한 없음)

        Returns:
            시계열 데이터 리스트 (각 항목은 dict)
            - 단일 컬럼인 경우: {"date": ..., "column_name": ...}
            - 여러 컬럼인 경우: {"date": ..., "column1": ..., "column2": ..., ...}

        Example:
            >>> data = client.get_timeseries_data()
            >>> data = client.get_timeseries_data(table_id="other_table", value_column="price")
        """
        # 기본값 사용
        dataset_id = dataset_id or self.dataset_id
        table_id = table_id or self.table_id
        date_column = date_column or self.date_column

        # value_column을 리스트로 통일 (문자열이면 리스트로 변환)
        if value_column is None:
            value_cols = self.value_columns
        elif isinstance(value_column, str):
            # 쉼표로 구분된 문자열인 경우 리스트로 변환
            if "," in value_column:
                value_cols = [col.strip() for col in value_column.split(",") if col.strip()]
            else:
                value_cols = [value_column]
        elif isinstance(value_column, list):
            value_cols = value_column
        else:
            raise TypeError(f"value_column은 문자열 또는 리스트여야 합니다. 받은 타입: {type(value_column)}")

        base_date = base_date if base_date is not None else self.base_date
        days = days if days is not None else (self.days if self.days is not None else 30)

        if not dataset_id or not table_id:
            raise ValueError("dataset_id와 table_id가 필요합니다. 환경변수 또는 파라미터로 설정하세요.")

        if not value_cols:
            raise ValueError("value_column이 필요합니다.")

        # 기준 날짜 설정
        if base_date:
            try:
                base_dt = datetime.strptime(base_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요: {base_date}")
        else:
            base_dt = datetime.now()

        # 시작 날짜 계산 (base_date 기준으로 days일 전)
        start_date = base_dt - timedelta(days=days - 1)
        end_date = base_dt

        # 정렬 기준 설정
        if order_by is None:
            order_by = f"{date_column} ASC" if date_column else "1"

        # SELECT 절 구성 (날짜 컬럼이 있으면 포함)
        if date_column:
            select_columns = [date_column] + value_cols
        else:
            select_columns = value_cols
        select_clause = ", ".join(select_columns)

        where_conditions = []
        if date_column:
            where_conditions.append(f"{date_column} >= '{start_date.strftime('%Y-%m-%d')}'")
            where_conditions.append(f"{date_column} <= '{end_date.strftime('%Y-%m-%d')}'")
        if where_clause:
            where_conditions.append(where_clause)

        where_clause_sql = " AND ".join(where_conditions) if where_conditions else "1=1"

        # 쿼리 작성
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT 
            {select_clause}
        FROM `{self.project_id}.{dataset_id}.{table_id}`
        WHERE {where_clause_sql}
        ORDER BY {order_by}
        {limit_clause}
        """

        # 쿼리 실행
        query_job = self.client.query(query)
        results = query_job.result()

        # 결과를 리스트로 변환
        data = []
        for row in results:
            row_dict = {}
            if date_column:
                row_dict["date"] = row[date_column]
            # 여러 컬럼 추가
            for col in value_cols:
                row_dict[col] = row[col]
            data.append(row_dict)

        # 데이터가 뒤죽박죽일 수 있으므로 날짜 기준으로 다시 정렬 (안전장치, 날짜 컬럼이 있는 경우만)
        if date_column:

            def parse_date(date_value):
                """날짜 값을 datetime 객체로 변환"""
                if isinstance(date_value, datetime):
                    return date_value
                elif isinstance(date_value, str):
                    # 여러 날짜 형식 시도 (밀리초 포함 형식도 지원)
                    for fmt in [
                        "%Y-%m-%d %H:%M:%S.%f",  # 2025-11-14 23:26:13.000
                        "%Y-%m-%d %H:%M:%S",  # 2025-11-14 23:26:13
                        "%Y-%m-%d",  # 2025-11-14
                        "%Y-%m-%dT%H:%M:%S",  # ISO 형식
                    ]:
                        try:
                            return datetime.strptime(date_value, fmt)
                        except ValueError:
                            continue
                    # 파싱 실패 시 원본 반환
                    return date_value
                else:
                    return date_value

            try:
                data.sort(key=lambda x: parse_date(x.get("date")))
            except Exception:
                # 정렬 실패 시 원본 순서 유지
                pass

        return data

    def get_timeseries_values(
        self,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        date_column: Optional[str] = None,
        value_column: Optional[Union[str, List[str]]] = None,
        base_date: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Union[List[float], Dict[str, List[float]]]:
        """
        특정 날짜 기준으로 최근 N일치 시계열 데이터의 값만 리스트로 가져오기

        다른 테이블을 조회하려면 table_id와 value_column을 지정하면 됩니다.

        Args:
            dataset_id: 데이터셋 ID (None이면 초기화 시 설정한 값 사용)
            table_id: 테이블 ID (None이면 초기화 시 설정한 값 사용)
            date_column: 날짜 컬럼명 (None이면 초기화 시 설정한 값 사용)
            value_column: 값 컬럼명 (문자열 또는 리스트, None이면 초기화 시 설정한 값 사용)
                - 문자열: 단일 컬럼 지정 (예: "price")
                - 리스트: 여러 컬럼 지정 (예: ["price", "volume", "market_cap"])
                - 쉼표로 구분된 문자열도 가능 (예: "price,volume")
            base_date: 기준 날짜 (YYYY-MM-DD 형식, None이면 초기화 시 설정한 값 또는 오늘)
            days: 가져올 일수 (None이면 초기화 시 설정한 값 사용)

        Returns:
            단일 컬럼인 경우: 값 리스트 (List[float])
            여러 컬럼인 경우: 컬럼별 값 리스트 딕셔너리 (Dict[str, List[float]])

        Example:
            >>> values = client.get_timeseries_values()
            >>> values = client.get_timeseries_values(table_id="other_table", value_column="price")
        """
        data = self.get_timeseries_data(
            dataset_id=dataset_id,
            table_id=table_id,
            date_column=date_column,
            value_column=value_column,
            base_date=base_date,
            days=days,
        )

        if not data:
            # value_column이 리스트인지 확인
            if value_column is None:
                value_cols = self.value_columns
            elif isinstance(value_column, str):
                value_cols = [value_column]
            else:
                value_cols = value_column
            return [] if len(value_cols) == 1 else {}

        # 첫 번째 행에서 컬럼 확인 (date 제외)
        first_row_keys = [k for k in data[0].keys() if k != "date"]

        # 단일 컬럼인 경우 (하위 호환성)
        if len(first_row_keys) == 1:
            col_name = first_row_keys[0]
            return [float(item[col_name]) for item in data]
        else:
            # 여러 컬럼인 경우
            result = {}
            for col_name in first_row_keys:
                result[col_name] = [float(item[col_name]) for item in data]
            return result

    def get_custom_query(self, query: str) -> List[Dict[str, Any]]:
        """
        커스텀 SQL 쿼리 실행

        Args:
            query: SQL 쿼리 문자열

        Returns:
            쿼리 결과 리스트
        """
        query_job = self.client.query(query)
        results = query_job.result()

        data = []
        for row in results:
            row_dict = {}
            for key, value in row.items():
                row_dict[key] = value
            data.append(row_dict)

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
    편의 함수: BigQuery에서 시계열 데이터 값 리스트 가져오기
    (환경변수에서 기본값을 읽어옴)

    Args:
        dataset_id: 데이터셋 ID (None이면 환경변수 BIGQUERY_DATASET_ID 사용)
        table_id: 테이블 ID (None이면 환경변수 BIGQUERY_TABLE_ID 사용)
        project_id: GCP 프로젝트 ID (None이면 환경변수 VERTEX_AI_PROJECT_ID 사용)
        date_column: 날짜 컬럼명 (None이면 환경변수 BIGQUERY_DATE_COLUMN 사용, 기본값: "time")
        value_column: 값 컬럼명 (문자열 또는 리스트, None이면 환경변수 BIGQUERY_VALUE_COLUMN 사용, 기본값: "value")
            - 문자열: 단일 컬럼 지정 (예: "price")
            - 리스트: 여러 컬럼 지정 (예: ["price", "volume", "market_cap"])
        base_date: 기준 날짜 (YYYY-MM-DD 형식, None이면 환경변수 BIGQUERY_BASE_DATE 또는 오늘)
        days: 가져올 일수 (None이면 환경변수 BIGQUERY_DAYS 사용, 기본값: 30)

    Returns:
        단일 컬럼인 경우: 값 리스트 (List[float])
        여러 컬럼인 경우: 컬럼별 값 리스트 딕셔너리 (Dict[str, List[float]])

    Example:
        >>> values = get_bigquery_timeseries()
        >>> values = get_bigquery_timeseries(days=60)
        >>> values = get_bigquery_timeseries(value_column="price")
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
    return client.get_timeseries_values()


if __name__ == "__main__":
    """스크립트 직접 실행 시 간단 테스트"""
    try:
        client = BigQueryClient()

        # corn_price 테이블에서 close와 EMA 가져오기
        print("\n[corn_price 테이블 조회]")
        corn_data = client.get_timeseries_data(table_id="corn_price", value_column=["close", "EMA"])
        print(f"✅ corn_price 데이터 조회 성공: {len(corn_data)}개 레코드")
        if corn_data:
            print(f"   첫 번째 날짜: {corn_data[0]['date']}, 마지막 날짜: {corn_data[-1]['date']}")
            print(f"   첫 번째 close: {corn_data[0].get('close', 'N/A')}, EMA: {corn_data[0].get('EMA', 'N/A')}")

        # news_article 테이블에서 description 컬럼 가져오기 (filter_status='T'만)
        print("\n[news_article 테이블 조회]")
        article_data = client.get_timeseries_data(
            table_id="news_article",
            value_column="description",
            date_column="publish_date",
            where_clause="filter_status = 'T'",
            days=3,
        )
        print(f"✅ news_article 데이터 조회 성공: {len(article_data)}개 레코드")
        if article_data:
            desc = article_data[0].get("description", "N/A")
            publish_date = article_data[0].get("date", "N/A")
            print(f"   첫 번째 날짜: {publish_date}")
            print(f"   첫 번째 설명: {str(desc)[:200]}...")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback

        traceback.print_exc()
