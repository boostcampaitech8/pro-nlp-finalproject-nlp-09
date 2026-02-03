-- 뉴스 기사 + 임베딩 조회 (감성 분석 모델용)
-- 필터링된 뉴스 기사와 임베딩을 함께 조회합니다.
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {start_date}: 시작 날짜 (YYYY-MM-DD)
--   {end_date}: 종료 날짜 (YYYY-MM-DD)

SELECT
    a.article_id,
    a.publish_date,
    a.title,
    a.description,
    a.key_word,
    e.embedding as article_embedding
FROM `{project_id}.{dataset_id}.news_articles` a
LEFT JOIN `{project_id}.{dataset_id}.article_embeddings` e
    ON a.article_id = e.article_id
    AND a.publish_date = e.publish_date
WHERE a.publish_date >= '{start_date}'
  AND a.publish_date <= '{end_date}'
  AND a.filter_status = 'T'
ORDER BY a.publish_date DESC
