# daily_prices (BigQuery) — Schema & Query Notes

## 1) Table overview
- **Purpose**: 일별 원자재 선물 가격 + 파생 지표(EMA) + 거래량(Volume)
- **Data location (Region/Multi-region)**: **US**
- **Table type**: **Native table (Partitioned)**
- **Partitioning**: **DAY**, partitioned on **`date` (DATE)**
- **Clustering**: **`commodity`**
- **Row count**: **9,232**
- **Partition count**: **3,824**

## 2) Columns
| Column | Type | Mode | Description |
|---|---|---|---|
| commodity | STRING | REQUIRED | 원자재 구분자 (예: corn / soybean / wheat) |
| date | DATE | REQUIRED | 거래 일자 (파티션 키) |
| open | FLOAT64 | NULLABLE | 시가 |
| high | FLOAT64 | NULLABLE | 고가 |
| low | FLOAT64 | NULLABLE | 저가 |
| close | FLOAT64 | NULLABLE | 종가 |
| ema | FLOAT64 | NULLABLE | 지수이동평균(EMA) |
| volume | INT64 | NULLABLE | 거래량 |
| ingested_at | TIMESTAMP | NULLABLE (default) | 적재 시각, 기본값 `CURRENT_TIMESTAMP()` |

## 3) Keys / Constraints (실무적으로 이렇게 취급)
- BigQuery **PK는 미설정** (UI에서도 Primary key(s) 비어있음)
- **Natural key 권장**: `(commodity, date)`  
  - 같은 `(commodity, date)`가 중복 적재되지 않도록 ETL에서 **MERGE/UPSERT 규칙**을 두는 걸 권장

## 4) Query 작성 시 필수 팁
- **파티션 프루닝**이 비용/성능에 가장 중요  
  - 항상 `WHERE date BETWEEN ...` 또는 `WHERE date >= ...` 형태로 날짜 필터를 넣는 습관 권장
- **클러스터링(`commodity`)** 덕분에 아래 패턴이 유리  
  - `WHERE commodity = 'corn' AND date BETWEEN ...`
- `ingested_at`은 “데이터가 생성된 시점”이 아니라 **BigQuery에 들어온 시점**(적재 시각)으로 사용

## 5) Canonical DDL (요약)
```sql
CREATE TABLE IF NOT EXISTS `...dataset.daily_prices` (
  commodity   STRING    NOT NULL,
  date        DATE      NOT NULL,
  open        FLOAT64,
  high        FLOAT64,
  low         FLOAT64,
  close       FLOAT64,
  ema         FLOAT64,
  volume      INT64,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY commodity;

## 6) raw 데이터 기준 정보
- 행 개수: 9,232
- 파티션 개수: 3,824