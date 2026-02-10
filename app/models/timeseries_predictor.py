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
        config_path = os.path.join(root_dir, 'app', 'model', 'timeseries_model', 'config.yaml')
            
        _inference_engine = TimeSeriesXGBoostInference(config_path=config_path)
    return _inference_engine

def predict_market_trend(target_date: str, commodity: str = "corn") -> str:
    dt = datetime.strptime(target_date, '%Y-%m-%d')

    engine = get_inference_engine()
        
    bq_client = BigQueryClient() 

    history_df = bq_client.get_prophet_features(
        target_date=target_date, 
        lookback_days=3000,
        commodity=commodity
    )
        
    result = engine.predict(history_df, target_date)
        
    return json.dumps(result, ensure_ascii=False)
        
if __name__ == "__main__":
    print(predict_market_trend("2025-11-26"))