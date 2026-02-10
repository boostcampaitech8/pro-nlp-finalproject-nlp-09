-- Get generic time-series data
-- This query retrieves time-series data with flexible column selection
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {table_id}: Table name
--   {select_columns}: Columns to select (comma-separated)
--   {date_column}: Date column name (optional)
--   {start_date}: Start date (YYYY-MM-DD, optional)
--   {end_date}: End date (YYYY-MM-DD, optional)
--   {where_clause}: Additional WHERE conditions (optional)
--   {order_by}: ORDER BY clause (optional)
--   {limit}: LIMIT clause (optional)

SELECT {select_columns}
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE {where_clause}
{order_by}
{limit}
