-- 뉴스 기사 메타 정보 조회
-- 필터링된 뉴스 기사의 기본 정보를 조회합니다.
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {start_date}: 시작 날짜 (YYYY-MM-DD)
--   {end_date}: 종료 날짜 (YYYY-MM-DD)
--   {filter_status}: 필터 상태 (T/F/E)
--   {limit_clause}: LIMIT 절 (선택적, 예: "LIMIT 100" 또는 "")

SELECT
    article_id,
    publish_date,
    meta_site_name,
    key_word,
    filter_status,
    title,
    description,
    doc_url,
    authors,
    ingested_at
FROM `{project_id}.{dataset_id}.news_articles`
WHERE publish_date >= '{start_date}'
  AND publish_date <= '{end_date}'
  AND filter_status = '{filter_status}'
ORDER BY publish_date DESC
{limit_clause}
