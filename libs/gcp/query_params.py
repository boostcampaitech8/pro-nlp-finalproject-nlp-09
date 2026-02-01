"""
쿼리 파라미터 검증 모듈

BigQuery 쿼리에 사용되는 파라미터를 Pydantic으로 검증합니다.
SQL injection 방지 및 타입 안전성을 보장합니다.
"""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from libs.utils.constants import VALID_COMMODITIES, VALID_FILTER_STATUS, DATE_FORMAT


class PriceQueryParams(BaseModel):
    """
    가격 쿼리 파라미터 검증

    Example:
        >>> params = PriceQueryParams(
        ...     commodity="corn",
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31"
        ... )
    """

    commodity: str
    start_date: str
    end_date: str

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """commodity 값 검증 (allowlist 기반)"""
        v_lower = v.lower().strip()
        if v_lower not in VALID_COMMODITIES:
            raise ValueError(
                f"Invalid commodity: {v}. Must be one of: {sorted(VALID_COMMODITIES)}"
            )
        return v_lower

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """YYYY-MM-DD 형식 검증"""
        try:
            datetime.strptime(v, DATE_FORMAT)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")


class ProphetFeaturesParams(BaseModel):
    """
    Prophet 피처 조회 파라미터 검증

    Example:
        >>> params = ProphetFeaturesParams(
        ...     commodity="corn",
        ...     target_date="2025-01-31",
        ...     lookback_days=60
        ... )
    """

    commodity: str
    target_date: str
    lookback_days: int = 60

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """commodity 값 검증"""
        v_lower = v.lower().strip()
        if v_lower not in VALID_COMMODITIES:
            raise ValueError(
                f"Invalid commodity: {v}. Must be one of: {sorted(VALID_COMMODITIES)}"
            )
        return v_lower

    @field_validator("target_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """YYYY-MM-DD 형식 검증"""
        try:
            datetime.strptime(v, DATE_FORMAT)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")

    @field_validator("lookback_days")
    @classmethod
    def validate_lookback_days(cls, v: int) -> int:
        """lookback_days 양수 검증"""
        if v <= 0:
            raise ValueError("lookback_days must be positive")
        if v > 365 * 5:  # 5년 초과 방지
            raise ValueError("lookback_days cannot exceed 5 years (1825 days)")
        return v


class NewsQueryParams(BaseModel):
    """
    뉴스 쿼리 파라미터 검증

    Example:
        >>> params = NewsQueryParams(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-07",
        ...     filter_status="T"
        ... )
    """

    start_date: str
    end_date: str
    filter_status: Literal["T", "F", "E"] = "T"
    key_word: Optional[str] = None
    limit: Optional[int] = None

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """YYYY-MM-DD 형식 검증"""
        try:
            datetime.strptime(v, DATE_FORMAT)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")

    @field_validator("filter_status")
    @classmethod
    def validate_filter_status(cls, v: str) -> str:
        """filter_status 값 검증"""
        if v not in VALID_FILTER_STATUS:
            raise ValueError(
                f"Invalid filter_status: {v}. Must be one of: {sorted(VALID_FILTER_STATUS)}"
            )
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: Optional[int]) -> Optional[int]:
        """limit 양수 검증"""
        if v is not None and v <= 0:
            raise ValueError("limit must be positive")
        return v


class NewsForPredictionParams(BaseModel):
    """
    예측용 뉴스 조회 파라미터 검증

    Example:
        >>> params = NewsForPredictionParams(
        ...     target_date="2025-01-31",
        ...     lookback_days=7
        ... )
    """

    target_date: str
    lookback_days: int = 7

    @field_validator("target_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """YYYY-MM-DD 형식 검증"""
        try:
            datetime.strptime(v, DATE_FORMAT)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")

    @field_validator("lookback_days")
    @classmethod
    def validate_lookback_days(cls, v: int) -> int:
        """lookback_days 양수 검증"""
        if v <= 0:
            raise ValueError("lookback_days must be positive")
        if v > 90:  # 3개월 초과 방지
            raise ValueError("lookback_days cannot exceed 90 days for news")
        return v
