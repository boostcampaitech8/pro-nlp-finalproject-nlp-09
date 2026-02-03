-- 뉴스 감성 분석 결과 저장 테이블
-- Parameters:
--   {project_id}: GCP Project ID
--   {dataset_id}: BigQuery Dataset ID

CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.prediction_news_sentiment` (
    target_date DATE NOT NULL,          -- 예측 기준일
    prediction INT64,                   -- 예측 결과 (0: 하락, 1: 상승)
    probability FLOAT64,                -- 상승 확률 (0~1)
    confidence FLOAT64,                 -- 예측 확신도 (probability와 동일)
    features_summary JSON,              -- 분석에 사용된 피처 요약 (news_count, avg_sentiment 등)
    evidence_news JSON,                 -- 근거 뉴스 리스트 (제목, 점수, 본문 일부 등 배열)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() -- 적재 시간
)
PARTITION BY target_date
CLUSTER BY prediction;
