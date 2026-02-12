-- 뉴스 감성 분석 결과 저장 테이블 (다중 품목 대응)
-- Parameters:
--   {project_id}: GCP Project ID
--   {dataset_id}: BigQuery Dataset ID

CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.prediction_news_sentiment` (
    commodity STRING NOT NULL,          -- 상품명 (corn, soybean, wheat)
    target_date DATE NOT NULL,          -- 예측 기준일
    prediction INT64,                   -- 예측 결과 (0: 하락, 1: 상승)
    probability FLOAT64,                -- 상승할 확률 (0~1)
    confidence FLOAT64,                 -- 예측 확신도
    features_summary JSON,              -- 분석 피처 요약
    evidence_news JSON,                 -- 근거 뉴스 리스트
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() -- 적재 시간
)
PARTITION BY target_date
CLUSTER BY commodity, prediction;