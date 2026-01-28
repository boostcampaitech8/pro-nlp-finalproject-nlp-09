"""
Pydantic 모델 정의
FastAPI 요청/응답 스키마
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TimeSeriesPrediction(BaseModel):
    """시계열 예측 결과"""
    prediction: float = Field(..., description="예측값")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0 ~ 1.0)")
    timestamp: str = Field(..., description="예측 시각 (ISO 형식)")


class SentimentAnalysis(BaseModel):
    """감성분석 결과"""
    text: str = Field(..., description="분석된 텍스트")
    sentiment: str = Field(..., description="감성 (positive/negative/neutral)")
    scores: dict = Field(..., description="감성 점수 딕셔너리")


class OrchestratorInput(BaseModel):
    """Orchestrator 입력 데이터
    
    권장 방식: target_date를 사용하여 특정 시점의 데이터를 분석합니다.
    하위 호환성: timeseries_data와 news_articles는 유지됩니다.
    """
    target_date: Optional[str] = Field(None, description="분석 기준 날짜 (YYYY-MM-DD), 미입력시 최근 데이터 사용")
    
    # 하위 호환성 필드 (권장하지 않음)
    timeseries_data: Optional[List[float]] = Field(None, description="시계열 데이터 리스트 (하위 호환성, 권장하지 않음)")
    news_articles: Optional[List[str]] = Field(None, description="뉴스 기사 텍스트 리스트 (하위 호환성, 권장하지 않음)")
    context: Optional[str] = Field(None, description="분석 맥락")
    
    # BigQuery 파라미터 (권장 방식)
    timeseries_table_id: Optional[str] = Field(None, description="시계열 데이터 테이블명 (기본값: 'corn_price')")
    timeseries_value_column: Optional[str] = Field(None, description="시계열 값 컬럼명 (기본값: 'close')")
    timeseries_days: Optional[int] = Field(None, description="시계열 데이터 가져올 일수 (기본값: 30)")
    news_table_id: Optional[str] = Field(None, description="뉴스 테이블명 (기본값: 'news_article')")
    news_value_column: Optional[str] = Field(None, description="뉴스 텍스트 컬럼명 (기본값: 'description')")
    news_days: Optional[int] = Field(None, description="뉴스 가져올 일수 (기본값: 3)")


class OrchestratorOutput(BaseModel):
    """Orchestrator 출력 데이터"""
    timeseries_prediction: TimeSeriesPrediction = Field(..., description="시계열 예측 결과")
    sentiment_analysis: List[SentimentAnalysis] = Field(default_factory=list, description="감성분석 결과 리스트")
    llm_summary: str = Field(..., description="LLM 종합 요약")
