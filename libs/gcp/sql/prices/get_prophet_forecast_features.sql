-- prophet_forecast_features 테이블에서 Prophet 예측 피처 조회
-- 시계열 예측 모델 추론에 필요한 사전 계산된 Prophet 피처를 조회합니다.
--
-- Parameters:
--   {project_id}: GCP 프로젝트 ID
--   {dataset_id}: BigQuery 데이터셋 ID (예: market)
--   {commodity}: 원자재 종류 (예: corn, wheat, soybean)
--   {start_date}: 시작 날짜 (YYYY-MM-DD)
--   {end_date}: 종료 날짜 (YYYY-MM-DD)

SELECT
    ds,
    yhat,
    yhat_lower,
    yhat_upper,
    trend,
    weekly,
    yearly,
    extra_regressors_multiplicative,
    volume_lag1_effect,
    ema_lag1_effect,
    y,
    volume,
    ema,
    volume_lag1,
    ema_lag1,
    y_change,
    direction
FROM `{project_id}.{dataset_id}.prophet_forecast_features`
WHERE commodity = '{commodity}'
  AND ds >= '{start_date}'
  AND ds <= '{end_date}'
ORDER BY ds ASC
