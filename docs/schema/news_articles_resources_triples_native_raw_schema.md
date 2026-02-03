# news_articles_resources_triples_native_raw_schema

## 1) Table overview
- **Purpose**: 뉴스에서 추출된 **관계(triple) 텍스트 + triple 임베딩** 저장 (관계 기반 검색/Vector Search용)
- **Data location (Region/Multi-region)**: **US**
- **Table type**: **Native table (Non-partitioned)**
- **Partitioning**: 없음
- **Clustering**: **`hash_id`**
- **Row count**: **714,665**
- **Table description (Details 탭)**:
  - Typed native raw from `ext_news_articles_resources_triples`.
  - triple_text into `ARRAY<STRING>`, embedding into `ARRAY<FLOAT64>`.

## 2) Columns
| Column | Type | Mode | Description |
|---|---|---|---|
| hash_id | STRING | NULLABLE | 트리플(또는 레코드) 식별 해시 |
| triple_text | STRING | REPEATED | 트리플 텍스트 배열 (`ARRAY<STRING>`) |
| embedding | FLOAT | REPEATED | 트리플 임베딩 벡터 (`ARRAY<FLOAT64>`) |

## 3) Keys / Constraints (실무적으로 이렇게 취급)
- BigQuery **PK 미설정**
- **Natural key 후보**: `hash_id`
  - `hash_id`가 “한 트리플 단위”인지 “트리플 묶음 단위”인지는 데이터 생성 규칙에 따라 달라질 수 있음
  - (실무 권장) ETL에서 `hash_id` 유일성 규칙을 명확히 두고 적재

## 4) Query 작성 시 필수 팁
- **클러스터링(hash_id)** 최적 패턴
  - `WHERE hash_id = '...'`
- `triple_text`가 REPEATED이므로, 분석/집계 시:
  - `UNNEST(triple_text)`로 펼쳐서 사용
- Vector Search를 할 경우 `embedding` 대상으로 `VECTOR_SEARCH(...)` 적용 가능

## 5) Canonical DDL (요약)
```sql
CREATE TABLE IF NOT EXISTS
  `project-5b75bb04-485d-454e-af7.tilda_latest.news_articles_resources_triples_native_raw` (
  hash_id     STRING,
  triple_text ARRAY<STRING>,
  embedding   ARRAY<FLOAT64>
)
CLUSTER BY hash_id;
```

## 6) Storage / raw 데이터 기준 정보 (Details 탭)

* Number of rows: **714,665**
* Total logical bytes: **4.32 GB** (Active: 4.32 GB / Long term: 0 B)
* Total physical bytes: **1.93 GB** (Time travel: 0 B)