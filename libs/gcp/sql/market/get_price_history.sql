-- Get price history for model analysis
-- This query retrieves price data with returns for analysis
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {table_id}: Table name (e.g., corn_price)
--   {date_column}: Date column name (e.g., time)
--   {start_date}: Start date (YYYY-MM-DD)
--   {end_date}: End date (YYYY-MM-DD)

SELECT
    {date_column} as date,
    {date_column},
    close,
    ret_1d
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE {date_column} >= '{start_date}'
  AND {date_column} <= '{end_date}'
ORDER BY {date_column} ASC
