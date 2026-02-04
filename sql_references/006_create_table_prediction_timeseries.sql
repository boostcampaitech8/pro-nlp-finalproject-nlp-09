-- 시계열 예측 결과 저장 테이블 (경량화 버전)
-- Parameters:
--   {project_id}: GCP Project ID
--   {dataset_id}: BigQuery Dataset ID

CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.prediction_timeseries` (
    target_date DATE NOT NULL,          -- 예측 기준일
    forecast_direction STRING,          -- 예측 방향 (Up/Down)
    confidence_score FLOAT64,           -- 예측 신뢰도 (%)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() -- 적재 시간
)
PARTITION BY target_date
CLUSTER BY forecast_direction;
