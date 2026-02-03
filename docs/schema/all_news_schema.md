<!-- ## 임베딩 분리(하지만 쉽게 다시 합칠 수 있게 View 제공), 그리고 파티션/클러스터 + 비용 가드레일까지 반영한 BigQuery 스키마 초안입니다.

-- =========================================================
-- (2) 뉴스 기사 — bq.core.news_articles (권장 스키마)
--  - commodity 컬럼은 두지 않음: key_word를 그대로 사용
--  - 임베딩/긴 본문/대용량 추출물(named_entities, triples)은 분리
--  - join(re-hydration)을 위한 View 제공
--  - ingested_at은 bq.core.news_articles에만 둠 (SSOT)
-- =========================================================

-- 1) 기사 메타 (narrow table)  ✅ ingested_at 유지(유일)
CREATE TABLE IF NOT EXISTS `bq.core.news_articles` (
  article_id     INT64  NOT NULL,        -- 원본 id
  publish_date   DATE   NOT NULL,        -- 파티션 키
  meta_site_name STRING,                 -- 매체명
  key_word       STRING,                 -- 기사 검색 키워드
  filter_status  STRING,                 -- 'T'/'F'/'E'
  title          STRING,
  description    STRING,
  doc_url        STRING,
  authors        STRING,                 -- 1단계에서는 STRING 유지(원본 그대로)
  ingested_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY publish_date
CLUSTER BY filter_status, key_word, meta_site_name
OPTIONS (
  description = "News article metadata. Keep narrow for cheap scans. key_word is used as the commodity-like tag.",
  require_partition_filter = TRUE
);

-- 2) 본문(긴 텍스트) 분리: all_text는 스캔 비용을 크게 올릴 수 있어 별도 테이블 권장
CREATE TABLE IF NOT EXISTS `bq.core.news_article_texts` (
  article_id   INT64 NOT NULL,
  publish_date DATE  NOT NULL,
  all_text     STRING
)
PARTITION BY publish_date
CLUSTER BY article_id
OPTIONS (
  description = "Long article body (all_text). Separated to reduce scan cost.",
  require_partition_filter = TRUE
);

-- 3) 엔티티/트리플 추출 결과(1단계: 원본 CSV 그대로 STRING 적재)
--    (filter_status == 'T' 일 때만 존재한다는 전제를 두되, null 허용)
CREATE TABLE IF NOT EXISTS `bq.core.news_article_enrichments_raw` (
  article_id          INT64 NOT NULL,
  publish_date        DATE  NOT NULL,
  named_entities_json STRING,  -- 예: ["India","Food Ministry",...]
  triples_json        STRING   -- 예: [["A","pred","B"], ...]
)
PARTITION BY publish_date
CLUSTER BY article_id
OPTIONS (
  description = "Raw extractions. Only present when filter_status='T'. Kept as STRING first (CSV JSON-string friendly).",
  require_partition_filter = TRUE
);

-- 4) 기사 임베딩(512차원) 분리 테이블
--    - Vector Search에서 자주 쓰는 필터 컬럼(filter_status, key_word, meta_site_name)을 같이 넣어
--      '임베딩 테이블 단독'으로도 필터링 + ANN 검색이 가능하게 설계
CREATE TABLE IF NOT EXISTS `bq.core.article_embeddings` (
  article_id     INT64  NOT NULL,
  publish_date   DATE   NOT NULL,
  filter_status  STRING NOT NULL,    -- 보통 'T'만 저장(혹은 T/F/E 모두 저장 + embedding null)
  key_word       STRING,
  meta_site_name STRING,
  embedding      ARRAY<FLOAT64>      -- 길이 512 (filter_status='T'일 때만 비-null)
)
PARTITION BY publish_date
CLUSTER BY filter_status, key_word, meta_site_name
OPTIONS (
  description = "Article embedding (512-d). Derived from title+description with titan-embed-text-v2. Designed for BigQuery Vector Search.",
  require_partition_filter = TRUE
);

-- 5) 엔티티/트리플 임베딩(1024차원) 차원 테이블(=dimension)
CREATE TABLE IF NOT EXISTS `bq.core.entity_embeddings` (
  hash_id     STRING NOT NULL,        -- prefix + md5(entity_text)
  entity_text STRING,
  embedding   ARRAY<FLOAT64>          -- 길이 1024
)
OPTIONS (
  description = "Entity embedding (1024-d). hash_id = 'entity-' + md5(entity_text)."
);

CREATE TABLE IF NOT EXISTS `bq.core.triple_embeddings` (
  hash_id     STRING NOT NULL,        -- prefix + md5(triple_text)
  triple_text STRING,
  embedding   ARRAY<FLOAT64>          -- 길이 1024
)
OPTIONS (
  description = "Triple embedding (1024-d). hash_id = 'triple-' + md5(triple_text)."
);

-- 6) 링크 테이블(=map): article ↔ entity/triple
--    - 파티션/클러스터를 넣어 기간 필터 + 해시 조인 비용을 낮춤
CREATE TABLE IF NOT EXISTS `bq.core.article_entity_map` (
  article_id      INT64  NOT NULL,
  publish_date    DATE   NOT NULL,
  entity_hash_id  STRING NOT NULL
)
PARTITION BY publish_date
CLUSTER BY entity_hash_id, article_id
OPTIONS (
  description = "Article-to-entity link table. entity_hash_id references bq.core.entity_embeddings.hash_id.",
  require_partition_filter = TRUE
);

CREATE TABLE IF NOT EXISTS `bq.core.article_triple_map` (
  article_id      INT64  NOT NULL,
  publish_date    DATE   NOT NULL,
  triple_hash_id  STRING NOT NULL
)
PARTITION BY publish_date
CLUSTER BY triple_hash_id, article_id
OPTIONS (
  description = "Article-to-triple link table. triple_hash_id references bq.core.triple_embeddings.hash_id.",
  require_partition_filter = TRUE
);

-- 7) “다시 합치기(Join/Re-hydration)”를 위한 뷰
--    - 리포트/분석 시에는 이 뷰를 쓰고,
--    - Vector Search / 필터링 단계에서는 article_embeddings만 스캔하는 식으로 비용 제어
CREATE OR REPLACE VIEW `bq.mart.news_articles_full` AS
SELECT
  a.*,
  t.all_text,
  r.named_entities_json,
  r.triples_json,
  e.embedding AS article_embedding
FROM `bq.core.news_articles` a
LEFT JOIN `bq.core.news_article_texts` t
  ON a.article_id = t.article_id AND a.publish_date = t.publish_date
LEFT JOIN `bq.core.news_article_enrichments_raw` r
  ON a.article_id = r.article_id AND a.publish_date = r.publish_date
LEFT JOIN `bq.core.article_embeddings` e
  ON a.article_id = e.article_id AND a.publish_date = e.publish_date; -->
