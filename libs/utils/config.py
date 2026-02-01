"""
중앙 집중식 설정 관리

이 모듈은 환경 변수에서 타입 안전한 설정을 로드합니다.
모든 설정 모델은 Pydantic을 사용하여 검증과 타입 체크를 수행합니다.

환경 변수는 libs/gcp/.env 파일에서 로드됩니다.
필수/민감한 값만 .env에서 로드하고, 나머지는 적절한 기본값을 사용합니다.

순수 상수는 constants.py에서 정의됩니다.

Example:
    >>> from libs.utils.config import get_config
    >>> config = get_config()
    >>> project_id = config.vertex_ai.project_id
    >>> dataset_id = config.bigquery.dataset_id
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import DATE_FORMAT

# .env 파일 위치: libs/gcp/.env
_ENV_FILE = Path(__file__).parent.parent / "gcp" / ".env"


class GCPConfig(BaseSettings):
    """
    Common GCP configuration

    This configuration is shared across all GCP services.
    """

    project_id: Optional[str] = Field(default=None, alias="GCP_PROJECT_ID")
    location: str = Field(default="us-east1", alias="GCP_LOCATION")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class VertexAIConfig(BaseSettings):
    """Vertex AI configuration"""

    project_id: Optional[str] = Field(default=None, alias="VERTEX_AI_PROJECT_ID")
    location: str = Field(default="us-central1", alias="VERTEX_AI_LOCATION")
    model_name: str = Field(
        default="meta/llama-3.1-70b-instruct-maas",
        alias="GENERATE_MODEL_NAME",
    )
    temperature: float = Field(default=0.7, alias="GENERATE_MODEL_TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="GENERATE_MODEL_MAX_TOKENS")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class BigQueryConfig(BaseSettings):
    """
    BigQuery configuration for daily_prices table

    Table schema: commodity, date, open, high, low, close, ema, volume, ingested_at
    """

    dataset_id: Optional[str] = Field(default=None, alias="BIGQUERY_DATASET_ID")
    table_id: str = Field(default="daily_prices", alias="BIGQUERY_TABLE_ID")
    date_column: str = Field(default="date", alias="BIGQUERY_DATE_COLUMN")
    value_column: str = Field(default="close", alias="BIGQUERY_VALUE_COLUMN")
    commodity: str = Field(default="corn", alias="BIGQUERY_COMMODITY")
    base_date: Optional[str] = Field(default=None, alias="BIGQUERY_BASE_DATE")
    days: int = Field(default=30, alias="BIGQUERY_DAYS")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("days must be positive")
        return v

    @field_validator("base_date")
    @classmethod
    def validate_base_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v:
            from datetime import datetime

            try:
                datetime.strptime(v, DATE_FORMAT)
            except ValueError:
                raise ValueError(f"base_date must be in {DATE_FORMAT} format")
        return v


class StorageConfig(BaseSettings):
    """Google Cloud Storage configuration"""

    bucket_name: Optional[str] = Field(default=None, alias="GCS_BUCKET_NAME")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class APIConfig(BaseSettings):
    """API server configuration"""

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @field_validator("debug", mode="before")
    @classmethod
    def validate_debug(cls, v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)


# TODO __init__ 메서드 생성 방식 데코레이를 활용한 패턴으로 변경
class AppConfig:
    """
    Complete application configuration container

    This class combines all configuration sections and provides
    easy access to all settings.
    """

    def __init__(self):
        self.gcp = GCPConfig()
        self.vertex_ai = VertexAIConfig()
        self.bigquery = BigQueryConfig()
        self.storage = StorageConfig()
        self.api = APIConfig()

    def __repr__(self) -> str:
        return (
            f"AppConfig(\n"
            f"  gcp={self.gcp},\n"
            f"  vertex_ai={self.vertex_ai},\n"
            f"  bigquery={self.bigquery},\n"
            f"  storage={self.storage},\n"
            f"  api={self.api}\n"
            f")"
        )


class ConfigManager:
    """
    Singleton configuration manager with lazy loading

    Usage:
        config = ConfigManager.get_instance()
        project_id = config.vertex_ai.project_id
    """

    _instance: Optional[AppConfig] = None

    @classmethod
    def get_instance(cls) -> AppConfig:
        """Get or create the singleton configuration instance"""
        if cls._instance is None:
            cls._instance = AppConfig()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)"""
        cls._instance = None


def get_config() -> AppConfig:
    """
    Convenience function to get the application configuration

    Returns:
        AppConfig: The application configuration instance

    Example:
        >>> from libs.utils.config import get_config
        >>> config = get_config()
        >>> project_id = config.vertex_ai.project_id
    """
    return ConfigManager.get_instance()
