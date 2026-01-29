-- Get news articles for sentiment prediction
-- This query retrieves filtered news articles with embeddings and scores
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {table_id}: News table name (e.g., news_article)
--   {start_date}: Start date (YYYY-MM-DD)
--   {end_date}: End date (YYYY-MM-DD)

SELECT
    publish_date,
    title,
    description as all_text,
    article_embedding,
    price_impact_score,
    sentiment_confidence,
    positive_score,
    negative_score,
    triples,
    filter_status
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE publish_date >= '{start_date}'
  AND publish_date <= '{end_date}'
  AND filter_status = 'T'
ORDER BY publish_date ASC
