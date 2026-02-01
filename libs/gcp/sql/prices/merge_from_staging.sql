-- 스테이징 테이블에서 메인 테이블로 MERGE (Upsert)
-- Airflow DAG에서 사용
--
-- Parameters:
--   @project_id: GCP 프로젝트 ID
--   @dataset_id: BigQuery 데이터셋 ID

MERGE `{project_id}.{dataset_id}.daily_prices` T
USING `{project_id}.{dataset_id}.stg_prices` S
ON T.commodity = S.commodity AND T.date = S.date
WHEN MATCHED THEN
  UPDATE SET
    open = S.open,
    high = S.high,
    low = S.low,
    close = S.close,
    ema = S.ema,
    volume = S.volume,
    ingested_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
  INSERT (commodity, date, open, high, low, close, ema, volume)
  VALUES (S.commodity, S.date, S.open, S.high, S.low, S.close, S.ema, S.volume)
