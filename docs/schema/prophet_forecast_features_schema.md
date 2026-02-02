# prophet_forecast_features (BigQuery) — Schema & Query Notes

## 1) Table overview
- **Purpose**: Prophet 예측 결과(yhat 계열) + 구성요소(trend/seasonality) + 외생변수/래그 피처 + 라벨(direction)
- **Data location (Region/Multi-region)**: **US**
- **Table type**: **Native table (Partitioned)**
- **Partitioning**: **DAY**, partitioned on **`ds` (DATE)**
- **Clustering**: **`commodity`**
- **Row count**: **1,254**
- **Partition count**: **1,254**

## 2) Columns
| Column | Type | Mode | Description |
|---|---|---|---|
| commodity | STRING | REQUIRED | 원자재 구분자 (예: corn / soybean / wheat) |
| ds | DATE | REQUIRED | 기준 일자 (파티션 키). Prophet의 날짜 컬럼 명명 관례(`ds`) |
| yhat | FLOAT64 | NULLABLE | 예측값 |
| yhat_lower | FLOAT64 | NULLABLE | 예측 하한 |
| yhat_upper | FLOAT64 | NULLABLE | 예측 상한 |
| trend | FLOAT64 | NULLABLE | Prophet 추세(trend) 컴포넌트 |
| weekly | FLOAT64 | NULLABLE | 주간 계절성(weekly seasonality) 컴포넌트 |
| yearly | FLOAT64 | NULLABLE | 연간 계절성(yearly seasonality) 컴포넌트 |
| extra_regressors_multiplicative | FLOAT64 | NULLABLE | 외생변수(회귀자) 효과(곱셈 형태) |
| volume_lag1_effect | FLOAT64 | NULLABLE | volume lag1이 예측에 기여한 효과(모델 산출) |
| ema_lag1_effect | FLOAT64 | NULLABLE | ema lag1이 예측에 기여한 효과(모델 산출) |
| y | FLOAT64 | NULLABLE | 실제값(타깃) |
| volume | INT64 | NULLABLE | 거래량(피처) |
| ema | FLOAT64 | NULLABLE | EMA(피처) |
| volume_lag1 | INT64 | NULLABLE | 전일 거래량(래그 피처) |
| ema_lag1 | FLOAT64 | NULLABLE | 전일 EMA(래그 피처) |
| y_change | FLOAT64 | NULLABLE | y 변화량(정의는 파이프라인 기준: 보통 `y - LAG(y)` 또는 수익률) |
| direction | INT64 | NULLABLE | 방향 라벨(예: 상승/하락을 1/0 등으로 인코딩) |
| ingested_at | TIMESTAMP | NULLABLE | 적재 시각 |

## 3) Keys / Constraints (실무적으로 이렇게 취급)
- BigQuery **PK는 미설정**
- **Natural key 권장**: `(commodity, ds)`
  - 같은 날짜/원자재의 피처가 재적재될 수 있으니 ETL에서 **MERGE 기준키**로 쓰기 좋음

## 4) Query 작성 시 필수 팁
- 비용/성능: **항상 `ds`로 기간 필터**를 거는 게 핵심 (파티션 프루닝)
  - `WHERE ds BETWEEN ...` 또는 `WHERE ds >= ...`
- **클러스터링(commodity)** 덕분에 아래 패턴이 가장 효율적
  - `WHERE commodity = 'corn' AND ds BETWEEN ...`
- `ds`는 이름이 `date`가 아니라 `ds`라서, daily_prices와 조인할 때 컬럼명 매핑 주의
  - `ON dp.commodity = pf.commodity AND dp.date = pf.ds`

## 5) Canonical DDL (요약)
```sql
CREATE TABLE IF NOT EXISTS `...dataset.prophet_forecast_features` (
  commodity STRING NOT NULL,
  ds DATE NOT NULL,
  yhat FLOAT64,
  yhat_lower FLOAT64,
  yhat_upper FLOAT64,
  trend FLOAT64,
  weekly FLOAT64,
  yearly FLOAT64,
  extra_regressors_multiplicative FLOAT64,
  volume_lag1_effect FLOAT64,
  ema_lag1_effect FLOAT64,
  y FLOAT64,
  volume INT64,
  ema FLOAT64,
  volume_lag1 INT64,
  ema_lag1 FLOAT64,
  y_change FLOAT64,
  direction INT64,
  ingested_at TIMESTAMP
)
PARTITION BY ds
CLUSTER BY commodity;

## 6) raw 데이터 기준 정보
- 행 개수: 1,254
- 파티션 개수: 1,254