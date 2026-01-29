-- Get Prophet features for time series prediction
-- This query retrieves historical data for Prophet model feature extraction
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {table_id}: Table name (e.g., corn_price)
--   {date_column}: Date column name (e.g., ds, time)
--   {start_date}: Start date (YYYY-MM-DD)
--   {end_date}: End date (YYYY-MM-DD)

SELECT *
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE {date_column} >= '{start_date}'
  AND {date_column} <= '{end_date}'
ORDER BY {date_column} ASC
