# BigQuery 클라이언트

Google Cloud BigQuery에서 시계열 데이터를 가져오는 모듈입니다.

## 설정 (환경변수)

`.env` 파일에 다음을 설정하세요:

```bash
# 필수
BIGQUERY_DATASET_ID=your_dataset_id
BIGQUERY_TABLE_ID=your_table_id

# 선택사항 (기본값)
BIGQUERY_DATE_COLUMN=time          # 기본값: "time"
BIGQUERY_VALUE_COLUMN=value        # 기본값: "value"
BIGQUERY_BASE_DATE=2026-01-23      # None이면 오늘
BIGQUERY_DAYS=30                    # 기본값: 30
```

## 사용법

### 1. 간단한 사용법 (환경변수 기본값 사용) - 권장 ✅

```python
from utils.bigquery_client import get_bigquery_timeseries

# 환경변수에서 모든 설정을 읽어옴 (매우 간단!)
values = get_bigquery_timeseries()

# 일부만 오버라이드 가능
values = get_bigquery_timeseries(days=60)  # 일수만 변경
```

### 2. 클래스 사용법

```python
from utils.bigquery_client import BigQueryClient

# 환경변수 기본값 사용
client = BigQueryClient()

# 파라미터 없이 호출하면 환경변수 설정값 사용
data = client.get_timeseries_data()
values = client.get_timeseries_values()

# 일부만 오버라이드
values = client.get_timeseries_values(days=60)

# 클라이언트 생성 시 기본값 설정 (환경변수 오버라이드)
client = BigQueryClient(
    dataset_id="other_dataset",
    table_id="other_table"
)
data = client.get_timeseries_data()
```

### 3. 커스텀 쿼리 실행

```python
client = BigQueryClient()

results = client.get_custom_query("""
    SELECT time, price, volume
    FROM `project.dataset.table`
    WHERE time >= '2026-01-01'
    ORDER BY time ASC
""")
```

## 설치

```bash
pip install google-cloud-bigquery>=3.13.0
```

또는

```bash
pip install -r requirements.txt
```

## 환경 변수 (전체 목록)

`.env` 파일에 다음을 설정하세요:

```bash
# Google Cloud 프로젝트 (필수)
VERTEX_AI_PROJECT_ID=your-google-cloud-project-id

# BigQuery 설정 (필수)
BIGQUERY_DATASET_ID=your_dataset_id
BIGQUERY_TABLE_ID=your_table_id

# BigQuery 설정 (선택사항)
BIGQUERY_DATE_COLUMN=time
BIGQUERY_VALUE_COLUMN=value
BIGQUERY_BASE_DATE=2026-01-23  # 생략하면 오늘
BIGQUERY_DAYS=30
```

## 인증

Google Cloud 인증이 설정되어 있어야 합니다:

```bash
gcloud auth application-default login
```

또는 서비스 계정 키 파일 사용:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```
