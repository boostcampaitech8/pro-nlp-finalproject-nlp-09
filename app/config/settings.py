# import os
# import pathlib
# import logging
# from dotenv import load_dotenv

# logger = logging.getLogger(__name__)

# # load_dotenv(dotenv_path="/data/ephemeral/home/kdh/pro-nlp-finalproject-nlp-09/.env")
# load_dotenv(
#     dotenv_path="/root/boostcampAI_final/pro-nlp-finalproject-nlp-09/legacy/.env_legacy"
# )

# # LSTM 설정 (추후 구현 예정)
# LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm_model.h5")
# LSTM_INPUT_SHAPE = (60, 1)
# LSTM_UNITS = 50

# # FinBERT 설정 (추후 구현 예정)
# FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
# FINBERT_DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else None

# # 생성 모델 설정
# GENERATE_MODEL_NAME = os.getenv("GENERATE_MODEL_NAME", None)
# GENERATE_MODEL_TEMPERATURE = float(os.getenv("GENERATE_MODEL_TEMPERATURE", "0.7"))
# GENERATE_MODEL_MAX_TOKENS = int(os.getenv("GENERATE_MODEL_MAX_TOKENS", "2048"))

# # Vertex AI 설정 (플랫폼 관련)
# VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
# VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")

# # BigQuery 설정
# # LangChain Agent의 Tool이 BigQuery에서 데이터를 조회할 때 사용하는 기본값
# BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID")
# BIGQUERY_TABLE_ID = os.getenv("BIGQUERY_TABLE_ID")
# BIGQUERY_DATE_COLUMN = os.getenv("BIGQUERY_DATE_COLUMN", "time")
# BIGQUERY_VALUE_COLUMN = os.getenv("BIGQUERY_VALUE_COLUMN", "close")
# BIGQUERY_BASE_DATE = os.getenv("BIGQUERY_BASE_DATE")  # YYYY-MM-DD 형식, None이면 오늘
# BIGQUERY_DAYS = int(os.getenv("BIGQUERY_DAYS", "30"))

# TS_MODEL_PATH = os.getenv("TS_MODEL_PATH", "should_be_set_in_env")

# # 참고: run_pipeline.py에서 Tool에 전달하는 파라미터로 테이블명 등을 지정할 수 있습니다.
# # 기본값:
# # - 시계열: table_id="corn_price", value_column="close", days=30
# # - 뉴스: table_id="news_article", value_column="description", days=3

# # 서버 설정
# API_HOST = os.getenv("API_HOST", "0.0.0.0")
# API_PORT = int(os.getenv("API_PORT", "8000"))
# DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# logger.info("=" * 80)
# logger.info("Configuration Settings Loaded")
# logger.info("=" * 80)

# logger.info("--- LSTM Settings ---")
# logger.info(f"LSTM_MODEL_PATH: {LSTM_MODEL_PATH}")
# logger.info(f"LSTM_INPUT_SHAPE: {LSTM_INPUT_SHAPE}")
# logger.info(f"LSTM_UNITS: {LSTM_UNITS}")

# logger.info("--- FinBERT Settings ---")
# logger.info(f"FINBERT_MODEL_NAME: {FINBERT_MODEL_NAME}")
# logger.info(f"FINBERT_DEVICE: {FINBERT_DEVICE}")

# logger.info("--- Generate Model Settings ---")
# logger.info(f"GENERATE_MODEL_NAME: {GENERATE_MODEL_NAME}")
# logger.info(f"GENERATE_MODEL_TEMPERATURE: {GENERATE_MODEL_TEMPERATURE}")
# logger.info(f"GENERATE_MODEL_MAX_TOKENS: {GENERATE_MODEL_MAX_TOKENS}")

# logger.info("--- Vertex AI Settings ---")
# logger.info(f"VERTEX_AI_PROJECT_ID: {VERTEX_AI_PROJECT_ID}")
# logger.info(f"VERTEX_AI_LOCATION: {VERTEX_AI_LOCATION}")

# logger.info("--- BigQuery Settings ---")
# logger.info(f"BIGQUERY_DATASET_ID: {BIGQUERY_DATASET_ID}")
# logger.info(f"BIGQUERY_TABLE_ID: {BIGQUERY_TABLE_ID}")
# logger.info(f"BIGQUERY_DATE_COLUMN: {BIGQUERY_DATE_COLUMN}")
# logger.info(f"BIGQUERY_VALUE_COLUMN: {BIGQUERY_VALUE_COLUMN}")
# logger.info(f"BIGQUERY_BASE_DATE: {BIGQUERY_BASE_DATE}")
# logger.info(f"BIGQUERY_DAYS: {BIGQUERY_DAYS}")

# logger.info("--- Server Settings ---")
# logger.info(f"API_HOST: {API_HOST}")
# logger.info(f"API_PORT: {API_PORT}")
# logger.info(f"DEBUG: {DEBUG}")

# logger.info("=" * 80)
# logger.info("All configurations loaded successfully")
# logger.info("=" * 80)

####################################
####################################
####################################

# """
# app 설정 - libs/utils로 위임

# 이 모듈은 하위 호환성을 위해 유지됩니다.
# 새 코드에서는 libs/utils/config.py, libs/utils/constants.py를 직접 사용하세요.

# Example:
#     # 권장 (새 코드)
#     >>> from libs.utils.config import get_config
#     >>> from libs.utils.constants import GENERATE_MODEL_NAME
#     >>> config = get_config()
#     >>> project_id = config.vertex_ai.project_id

#     # 하위 호환성 (기존 코드)
#     >>> from app.config.settings import VERTEX_AI_PROJECT_ID
# """

# from libs.utils.config import get_config
# from libs.utils.constants import (
#     # 모델 상수
#     # BigQuery 상수
#     # API 상수
#     DEFAULT_API_HOST,
#     DEFAULT_API_PORT,
#     DEFAULT_DEBUG,
# )

# # 설정 인스턴스 (보안 정보만 .env에서 로드)
# _config = get_config()

# # =============================================================================
# # Vertex AI 설정
# # =============================================================================
# VERTEX_AI_PROJECT_ID = _config.vertex_ai.project_id

# # =============================================================================
# # BigQuery 설정
# # =============================================================================
# BIGQUERY_DATASET_ID = _config.bigquery.dataset_id

# # =============================================================================
# # 상수 (하위 호환성)
# # =============================================================================
# # constants.py에서 import한 값들을 그대로 노출
# API_HOST = DEFAULT_API_HOST
# API_PORT = DEFAULT_API_PORT
# DEBUG = DEFAULT_DEBUG
