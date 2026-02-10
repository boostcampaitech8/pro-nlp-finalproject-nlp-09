import sys
import os
import json
from datetime import datetime
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from model.timeseries_model.xg_inference import TimeSeriesXGBoostInference
except ImportError as e:
    print(f"경고: TimeSeriesXGBoostInference를 임포트할 수 없습니다. 오류: {e}")
    TimeSeriesXGBoostInference = None

try:
    from app.utils.bigquery_client import BigQueryClient
except ImportError as e:
    print(f"경고: BigQueryClient를 임포트할 수 없습니다. 오류: {e}")
    BigQueryClient = None

_inference_engine = None

def get_inference_engine():
    global _inference_engine
    if _inference_engine is None:
        if TimeSeriesXGBoostInference is None:
            raise ImportError("TimeSeriesXGBoostInference 모듈을 사용할 수 없습니다.")
        # config.yaml 경로 설정 (app/model/timeseries_model/config.yaml)
        # root_dir는 프로젝트 최상위이므로 app/ 을 경로에 추가해야 합니다.
        config_path = os.path.join(root_dir, 'app', 'model', 'timeseries_model', 'config.yaml')
        
        if not os.path.exists(config_path):
            print(f"⚠️ Warning: config.yaml을 찾을 수 없습니다: {config_path}")
            
        _inference_engine = TimeSeriesXGBoostInference(config_path=config_path)
    return _inference_engine

def predict_market_trend(target_date: str, commodity: str = "corn") -> str:
    """
    시계열 모델을 사용하여 특정 날짜의 금융 시장 추세를 예측합니다.
    BigQuery에서 필요한 피처 데이터를 가져옵니다.
    
    Args:
        target_date (str): 분석할 날짜 ('YYYY-MM-DD' 형식).
        commodity (str): 분석할 품목명 (기본값: "corn").
        
    Returns:
        str: 상세 예측 지표를 포함한 JSON 형식의 문자열.
    """
    try:
        dt = datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        return json.dumps({
            "error": f"잘못된 날짜 형식입니다: '{target_date}'. YYYY-MM-DD 형식을 사용해주세요."
        })

    # 추론 엔진 가져오기
    try:
        engine = get_inference_engine()
    except Exception as e:
        return json.dumps({
            "error": f"추론 엔진 초기화 실패: {str(e)}"
        })
        
    # BigQuery에서 데이터 가져오기
    if BigQueryClient is None:
        return json.dumps({
            "error": "BigQueryClient를 사용할 수 없어 피처 데이터를 가져올 수 없습니다."
        })
        
    try:
        bq_client = BigQueryClient() # 환경변수의 프로젝트/데이터셋 설정을 사용
        
        # 과거 데이터 조회 (타겟 날짜 + 문맥 정보 + 학습 데이터)
        # 품목명(commodity)을 쿼리에 전달하여 해당 데이터만 가져옴
        history_df = bq_client.get_prophet_features(
            target_date=target_date, 
            lookback_days=3000,
            commodity=commodity
        )
        
        if history_df.empty:
            return json.dumps({
                "error": f"BigQuery에서 {commodity}의 {target_date} (및 이전)에 대한 데이터를 찾을 수 없습니다."
            })
            
    except Exception as e:
        traceback.print_exc() # 서버 로그 확인용
        return json.dumps({
            "error": f"BigQuery 데이터 조회 실패: {str(e)}"
        })

    # 예측 수행
    try:
        result = engine.predict(history_df, target_date)
        
        # LLM용 출력 포맷팅
        return json.dumps(result, ensure_ascii=False)
        
    except ValueError as ve:
        return json.dumps({
            "error": str(ve)
        })
    except Exception as e:
        return json.dumps({
            "error": f"예측 중 예기치 않은 오류가 발생했습니다: {str(e)}"
        })

if __name__ == "__main__":
    # 함수 테스트 (BQ 자격 증명 필요)
    print("'2025-11-26' 날짜로 predict_market_trend 테스트 중...")
    print(predict_market_trend("2025-11-26"))