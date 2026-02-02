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
from libs.gcp.repositories.price_repository import PriceRepository
from app.config.settings import BIGQUERY_DATASET_ID


def load_timeseries_prediction(prediction_data: Dict[str, Any], dataset_id: Optional[str] = None) -> None:
    """
    시계열 예측 결과를 BigQuery 'prediction_timeseries' 테이블에 적재합니다.
    (PriceRepository 사용)

    Args:
        prediction_data (dict): 시계열 모델 예측 결과 (JSON 구조)
        dataset_id (str): 대상 BigQuery 데이터셋 ID (기본값: settings.BIGQUERY_DATASET_ID)
    """
    # 데이터셋 ID 결정 (인자값 -> 설정값 -> 기본값 'market')
    target_dataset = dataset_id or BIGQUERY_DATASET_ID or "tilda"

    # 팩토리를 통해 BigQuery 서비스 생성
    factory = GCPServiceFactory()
    bq_service = factory.get_bigquery_client(dataset_id=target_dataset)
    
    # 리포지토리 초기화 및 저장 수행
    price_repo = PriceRepository(bq_service)
    
    try:
        price_repo.save_prediction(prediction_data)
        print(f"✅ 시계열 예측 데이터 적재 완료: {prediction_data.get('target_date', 'Unknown Date')}")
    except Exception as e:
        print(f"❌ 적재 실패: {str(e)}")
        raise e
