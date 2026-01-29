-- Get Prophet features for time series prediction from daily_prices
-- This query retrieves historical data for Prophet model feature extraction
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID (e.g., market)
--   {commodity}: Commodity name (e.g., corn, wheat, soybean)
--   {start_date}: Start date (YYYY-MM-DD)
--   {end_date}: End date (YYYY-MM-DD)

SELECT
    date as ds,
    close as y,
    open,
    high,
    low,
    ema,
    volume
FROM `{project_id}.{dataset_id}.daily_prices`
WHERE commodity = '{commodity}'
  AND date >= '{start_date}'
  AND date <= '{end_date}'
ORDER BY date ASC
