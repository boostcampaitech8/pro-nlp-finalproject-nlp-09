"""
FastAPI 메인 애플리케이션
금융 분석 파이프라인 API 서버
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.orchestrator import router as orchestrator_router
from config.settings import API_HOST, API_PORT, DEBUG

# FastAPI 앱 생성
app = FastAPI(
    title="금융 분석 파이프라인 API",
    description="LSTM 시계열 예측 + FinBERT 감성분석 + LLM 요약 통합 API",
    version="1.0.0",
    debug=DEBUG
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(orchestrator_router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "금융 분석 파이프라인 API",
        "version": "1.0.0",
        "endpoints": {
            "orchestrator": "/api/orchestrator/summarize",
            "health": "/api/orchestrator/health"
        }
    }


@app.get("/health")
async def health():
    """전체 시스템 상태 확인"""
    return {
        "status": "healthy",
        "service": "금융 분석 파이프라인 API"
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    
    # app 디렉토리를 Python 경로에 추가 (프로젝트 루트에서 실행하는 경우 대비)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG
    )
