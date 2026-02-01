import os
from dotenv import load_dotenv

load_dotenv()

# LSTM 설정 (추후 구현 예정)
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm_model.h5")
LSTM_INPUT_SHAPE = (60, 1)
LSTM_UNITS = 50

# FinBERT 설정 (추후 구현 예정)
FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
FINBERT_DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"

# 생성 모델 설정
GENERATE_MODEL_NAME = os.getenv(
    "GENERATE_MODEL_NAME", "meta/llama-3.1-70b-instruct-maas"
)
GENERATE_MODEL_TEMPERATURE = float(os.getenv("GENERATE_MODEL_TEMPERATURE", "0.7"))
GENERATE_MODEL_MAX_TOKENS = int(os.getenv("GENERATE_MODEL_MAX_TOKENS", "2048"))

# Vertex AI 설정 (플랫폼 관련)
VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")

# BigQuery 설정
# LangChain Agent의 Tool이 BigQuery에서 데이터를 조회할 때 사용하는 기본값
BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID")
BIGQUERY_TABLE_ID = os.getenv("BIGQUERY_TABLE_ID")
BIGQUERY_DATE_COLUMN = os.getenv("BIGQUERY_DATE_COLUMN", "time")
BIGQUERY_VALUE_COLUMN = os.getenv("BIGQUERY_VALUE_COLUMN", "close")
BIGQUERY_BASE_DATE = os.getenv("BIGQUERY_BASE_DATE")  # YYYY-MM-DD 형식, None이면 오늘
BIGQUERY_DAYS = int(os.getenv("BIGQUERY_DAYS", "30"))

# 참고: run_pipeline.py에서 Tool에 전달하는 파라미터로 테이블명 등을 지정할 수 있습니다.
# 기본값:
# - 시계열: table_id="corn_price", value_column="close", days=30
# - 뉴스: table_id="news_article", value_column="description", days=3

# 서버 설정
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
