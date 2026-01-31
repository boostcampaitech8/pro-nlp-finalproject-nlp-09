-- Test query to read daily_prices table (safe query with LIMIT)
-- This query is for testing table connectivity without scanning too much data
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID (e.g., market)
--   {commodity}: Commodity name (e.g., corn)
--   {limit}: Number of rows to fetch (default: 10)

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
ORDER BY date DESC
LIMIT {limit}
