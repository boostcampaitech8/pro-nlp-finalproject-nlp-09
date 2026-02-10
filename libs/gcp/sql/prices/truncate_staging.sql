-- 스테이징 테이블 비우기
-- MERGE 후 Airflow DAG에서 사용
--
-- Parameters:
--   @project_id: GCP 프로젝트 ID
--   @dataset_id: BigQuery 데이터셋 ID

TRUNCATE TABLE `{project_id}.{dataset_id}.stg_prices`
