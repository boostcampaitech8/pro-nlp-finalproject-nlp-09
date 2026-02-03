# news_articles_resources_native_raw_schema

## 1) Table overview
- **Purpose**: 다양한 온라인 소스에서 수집한 **뉴스 기사 raw 데이터** 저장 (NLP/후처리용)
- **Data location (Region/Multi-region)**: **US**
- **Table type**: **Native table (Partitioned)**
- **Partitioning**: **DAY**, partitioned on **`publish_date` (DATE)**
- **Partition filter**: **Required** (require_partition_filter 활성)
- **Clustering**: **`filter_status`, `key_word`, `meta_site_name`**
- **Row count**: **296,514**
- **Partition count**: **2,931**
- **Table description (Details 탭)**:
  - Typed native raw from `ext_news_articles_resources`.
  - Arrays parsed; non-T rows get empty arrays.
  - Embedding parsed via robust regex extraction.

## 2) Columns
| Column | Type | Mode | Description |
|---|---|---|---|
| id | STRING | NULLABLE | 원본 레코드 식별자(문자열) |
| title | STRING | NULLABLE | 기사 제목 |
| doc_url | STRING | NULLABLE | 기사 URL |
| all_text | STRING | NULLABLE | 기사 본문(원문 텍스트) |
| authors | STRING | NULLABLE | 저자 정보(원본 그대로) |
| publish_date | DATE | NULLABLE | 발행일(파티션 키) |
| meta_site_name | STRING | NULLABLE | 매체/사이트명 |
| key_word | STRING | NULLABLE | 수집 키워드 |
| filter_status | STRING | NULLABLE | 필터 상태(예: T/F/E 등) |
| description | STRING | NULLABLE | 요약/설명 |
| named_entities | STRING | REPEATED | 엔티티 텍스트 배열 (`ARRAY<STRING>`) |
| triples | RECORD | REPEATED | 트리플 구조체 배열 (`ARRAY<STRUCT<...>>`) *(하위 필드는 UI에서 펼쳐서 확인 필요)* |
| article_embedding | FLOAT | REPEATED | 문서 임베딩 벡터 (`ARRAY<FLOAT64>`) |

## 3) Keys / Constraints (실무적으로 이렇게 취급)
- BigQuery **PK 미설정** (Details에서 Primary key(s) 비어있음)
- **Natural key 후보**: `id`
  - 중복 적재 방지를 위해 ETL에서 `id` 기준 MERGE/UPSERT 권장

## 4) Query 작성 시 필수 팁
- **파티션 프루닝 필수**: `WHERE publish_date BETWEEN ...` 같은 날짜 조건을 항상 포함해야 함(Partition filter required).
- **클러스터링 최적화 패턴**
  - `WHERE publish_date BETWEEN ... AND filter_status='T' AND key_word='corn'`
  - `meta_site_name`까지 함께 필터하면 추가 이점 가능
- **REPEATED 컬럼 사용**
  - `named_entities`: `UNNEST(named_entities)`로 펼쳐서 분석/집계
  - `triples`: `UNNEST(triples)`로 펼쳐서 사용(STRUCT 하위 필드명은 스키마 펼쳐서 확인 후 적용)

## 5) Canonical DDL (요약)
```sql
CREATE TABLE IF NOT EXISTS
  `project-5b75bb04-485d-454e-af7.tilda_latest.news_articles_resources_native_raw` (
  id              STRING,
  title           STRING,
  doc_url         STRING,
  all_text        STRING,
  authors         STRING,
  publish_date    DATE,
  meta_site_name  STRING,
  key_word        STRING,
  filter_status   STRING,
  description     STRING,
  named_entities  ARRAY<STRING>,
  triples         ARRAY<STRUCT<subj STRING, pred STRING, obj STRING>>, -- *STRUCT 필드는 실제 스키마 펼쳐서 확인 권장*
  article_embedding ARRAY<FLOAT64>
)
PARTITION BY publish_date
CLUSTER BY filter_status, key_word, meta_site_name
OPTIONS (require_partition_filter = TRUE);
```

## 6) Storage / raw 데이터 기준 정보 (Details 탭)

* Number of rows: **296,514**
* Number of partitions: **2,931**
* Total logical bytes: **643.2 MB** (Active: 643.2 MB / Long term: 0 B)
* Total physical bytes: **690.25 MB** (Current: 382.71 MB / Time travel: 307.55 MB)
