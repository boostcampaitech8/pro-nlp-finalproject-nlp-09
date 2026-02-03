# news_articles_resources_entities_native_raw_schema

## 1) Table overview
- **Purpose**: 뉴스에서 추출된 **엔티티 텍스트 + 엔티티 임베딩** 저장 (Vector Search/유사도 검색용)
- **Data location (Region/Multi-region)**: **US**
- **Table type**: **Native table (Non-partitioned)**
- **Partitioning**: 없음
- **Clustering**: **`hash_id`**
- **Row count**: **104,573**
- **Table description (Details 탭)**:
  - Typed native raw from `ext_news_articles_resources_entities`.
  - embedding parsed into `ARRAY<FLOAT64>`.

## 2) Columns
| Column | Type | Mode | Description |
|---|---|---|---|
| hash_id | STRING | NULLABLE | 엔티티(또는 레코드) 식별 해시 |
| entity_text | STRING | NULLABLE | 엔티티 텍스트 |
| embedding | FLOAT | REPEATED | 엔티티 임베딩 벡터 (`ARRAY<FLOAT64>`) |

## 3) Keys / Constraints (실무적으로 이렇게 취급)
- BigQuery **PK 미설정**
- **Natural key 후보**: `hash_id`
  - `hash_id` 생성 로직이 안정적이라는 전제에서, 중복 방지를 위해 ETL에서 `hash_id` 기준 upsert 권장

## 4) Query 작성 시 필수 팁
- **클러스터링(hash_id)** 덕분에 아래 패턴이 유리
  - `WHERE hash_id = '...'`
- Vector Search를 할 경우 `embedding` 컬럼을 대상으로 `VECTOR_SEARCH(...)` 사용 가능

## 5) Canonical DDL (요약)
```sql
CREATE TABLE IF NOT EXISTS
  `project-5b75bb04-485d-454e-af7.tilda_latest.news_articles_resources_entities_native_raw` (
  hash_id     STRING,
  entity_text STRING,
  embedding   ARRAY<FLOAT64>
)
CLUSTER BY hash_id;
```

## 6) Storage / raw 데이터 기준 정보 (Details 탭)

* Number of rows: **104,573**
* Total logical bytes: **632.01 MB** (Active: 632.01 MB / Long term: 0 B)
* Total physical bytes: **283.84 MB** (Time travel: 0 B)