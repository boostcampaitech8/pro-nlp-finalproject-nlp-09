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
from libs.gcp.repositories.news_repository import NewsRepository
from libs.gcp.storage import StorageService
from app.config.settings import BIGQUERY_DATASET_ID
from datetime import datetime


def load_timeseries_prediction(prediction_data: Dict[str, Any], commodity: str, dataset_id: Optional[str] = None) -> None:
    """
    시계열 예측 결과를 BigQuery 'prediction_timeseries' 테이블에 적재합니다.
    (PriceRepository 사용)

    Args:
        prediction_data (dict): 시계열 모델 예측 결과 (JSON 구조)
        commodity (str): 상품명 (corn, soybean, wheat)
        dataset_id (str): 대상 BigQuery 데이터셋 ID (기본값: settings.BIGQUERY_DATASET_ID)
    """
    # 데이터셋 ID 결정 (인자값 -> 설정값 -> 기본값 'market')
    target_dataset = dataset_id or BIGQUERY_DATASET_ID or "market"

    # 팩토리를 통해 BigQuery 서비스 생성
    factory = GCPServiceFactory()
    bq_service = factory.get_bigquery_client(dataset_id=target_dataset)
    
    # 리포지토리 초기화 및 저장 수행
    price_repo = PriceRepository(bq_service)
    
    try:
        # 필요한 핵심 컬럼만 필터링 (스키마 불일치 방지)
        filtered_data = {
            "target_date": prediction_data.get("target_date"),
            "forecast_direction": prediction_data.get("forecast_direction"),
            "confidence_score": prediction_data.get("confidence_score")
        }
        
        price_repo.save_prediction(filtered_data, commodity=commodity)
        print(f"✅ [{commodity}] 시계열 예측 데이터 적재 완료: {filtered_data.get('target_date', 'Unknown Date')}")
    except Exception as e:
        print(f"❌ [{commodity}] 적재 실패: {str(e)}")
        raise e


def load_news_prediction(prediction_data: Dict[str, Any], commodity: str, dataset_id: Optional[str] = None) -> None:
    """
    뉴스 감성 분석 결과를 BigQuery 'prediction_news_sentiment' 테이블에 적재합니다.
    (NewsRepository 사용)

    Args:
        prediction_data (dict): 뉴스 모델 예측 결과 (JSON 구조)
        commodity (str): 상품명 (corn, soybean, wheat)
        dataset_id (str): 대상 BigQuery 데이터셋 ID
    """
    target_dataset = dataset_id or BIGQUERY_DATASET_ID or "market"

    factory = GCPServiceFactory()
    bq_service = factory.get_bigquery_client(dataset_id=target_dataset)
    
    news_repo = NewsRepository(bq_service)
    
    try:
        news_repo.save_prediction(prediction_data, commodity=commodity)
        print(f"✅ [{commodity}] 뉴스 예측 데이터 적재 완료: {prediction_data.get('target_date', 'Unknown Date')}")
    except Exception as e:
        print(f"❌ [{commodity}] 뉴스 적재 실패: {str(e)}")
        raise e


def upload_report_to_gcs(report_text: str, target_date: str, commodity: str, bucket_name: str = "team-blue-raw-data") -> None:
    """
    최종 리포트 텍스트를 GCS에 업로드합니다.
    경로: reports/{YYYY}/{MM}/{commodity}_report_{YYYY-MM-DD}.txt

    Args:
        report_text (str): 리포트 내용
        target_date (str): 분석 날짜 (YYYY-MM-DD)
        commodity (str): 상품명 (corn, soybean, wheat)
        bucket_name (str): GCS 버킷 이름
    """
    if not report_text:
        print(f"⚠️ [{commodity}] 리포트 내용이 비어있어 업로드를 건너뜁니다.")
        return

    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        year = dt.strftime("%Y")
        month = dt.strftime("%m")
        # 품목 중심 계층 구조로 경로 수정
        blob_path = f"reports/{commodity}/{year}/{month}/{commodity}_report_{target_date}.txt"

        factory = GCPServiceFactory()
        storage_service = factory.get_storage_client(bucket_name=bucket_name)
        
        storage_service.upload_from_string(report_text, blob_path)
        print(f"✅ [{commodity}] 리포트 GCS 업로드 완료: gs://{bucket_name}/{blob_path}")
        
    except Exception as e:
        print(f"❌ [{commodity}] 리포트 업로드 실패: {str(e)}")
        raise e
