"""
데이터 적재 유틸리티
Airflow 등의 파이프라인에서 분석 결과를 BigQuery나 GCS에 저장할 때 사용하는 함수 모음입니다.
"""

from typing import Dict, Any, List, Optional
import sys
import os

# libs 경로 추가 (Airflow 등에서 실행 시 경로 문제 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from libs.gcp.base import GCPServiceFactory


def load_timeseries_prediction(prediction_data: Dict[str, Any], dataset_id: str = "market") -> None:
    """
    시계열 예측 결과를 BigQuery 'prediction_timeseries' 테이블에 적재합니다.

    Args:
        prediction_data (dict): 시계열 모델 예측 결과 (JSON 구조)
        dataset_id (str): 대상 BigQuery 데이터셋 ID (기본값: market)
    """
    if not prediction_data or "error" in prediction_data:
        print("❌ 적재 실패: 유효하지 않은 예측 데이터입니다.")
        return

    # 팩토리를 통해 BigQuery 서비스 생성 (인증 자동 처리)
    factory = GCPServiceFactory()
    bq_service = factory.get_bigquery_client(dataset_id=dataset_id)

    # 데이터 매핑 (필요한 경우) 및 리스트 감싸기
    # API는 리스트 형태의 row를 받음
    rows = [prediction_data]

    # 적재 수행
    table_id = "prediction_timeseries"
    errors = bq_service.insert_rows_json(table_id, rows)

    if errors:
        raise RuntimeError(f"BigQuery 적재 중 오류 발생: {errors}")
    else:
        print(f"✅ 시계열 예측 데이터 적재 완료: {prediction_data.get('target_date', 'Unknown Date')}")
