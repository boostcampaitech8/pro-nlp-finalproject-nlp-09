-- Get price history for model analysis from daily_prices table
-- This query retrieves price data for a specific commodity
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID (e.g., market)
--   {commodity}: Commodity name (e.g., corn, wheat, soybean)
--   {start_date}: Start date (YYYY-MM-DD)
--   {end_date}: End date (YYYY-MM-DD)

SELECT
    commodity,
    date,
    open,
    high,
    low,
    close,
    ema,
    volume,
    ingested_at
FROM `{project_id}.{dataset_id}.daily_prices`
WHERE commodity = '{commodity}'
  AND date >= '{start_date}'
  AND date <= '{end_date}'
ORDER BY date ASC
