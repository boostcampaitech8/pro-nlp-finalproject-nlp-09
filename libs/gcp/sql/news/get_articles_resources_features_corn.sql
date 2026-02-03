-- 옥수수 뉴스 감성 모델 예측용 뉴스 데이터 조회
-- corn_all_news_with_sentiment 테이블에서 필요한 피처를 조회합니다.
--
-- Parameters:
--   {project_id}: GCP project ID
--   {dataset_id}: BigQuery dataset ID
--   {start_date}: 시작 날짜 (YYYY-MM-DD)
--   {end_date}: 종료 날짜 (YYYY-MM-DD)

-- TODO description as all_text를 왜 썼는지 확인할 것
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
FROM `{project_id}.{dataset_id}.corn_all_news_with_sentiment`
WHERE publish_date >= '{start_date}'
  AND publish_date <= '{end_date}'
  AND filter_status = 'T'
ORDER BY publish_date ASC
