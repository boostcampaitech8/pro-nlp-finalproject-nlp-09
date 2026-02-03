-- 뉴스 기사 본문 조회
-- article_id 목록에 해당하는 본문을 조회합니다.
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {article_ids}: article_id 목록 (예: "1, 2, 3")

SELECT
    article_id,
    publish_date,
    all_text
FROM `{project_id}.{dataset_id}.news_article_texts`
WHERE article_id IN ({article_ids})
ORDER BY article_id
