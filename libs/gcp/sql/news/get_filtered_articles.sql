-- Get filtered news articles
-- This query retrieves news articles that passed filtering (filter_status='T')
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {table_id}: News table name (e.g., news_article)
--   {start_date}: Start date (YYYY-MM-DD, optional)
--   {end_date}: End date (YYYY-MM-DD, optional)
--   {limit}: Result limit (optional)

SELECT
    publish_date,
    title,
    description,
    source,
    url,
    filter_status,
    triples
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE filter_status = 'T'
{date_filter}
ORDER BY publish_date DESC
{limit_clause}
