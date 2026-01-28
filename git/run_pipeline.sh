#!/bin/bash
# Prophet + XGBoost 전체 파이프라인 실행

echo "========================================"
echo "Prophet + XGBoost 파이프라인 시작"
echo "========================================"
echo ""

# 1. Prophet 실행
echo "1️⃣  Prophet features 추출 중..."
python run_prophet.py
if [ $? -ne 0 ]; then
    echo "❌ Prophet 실행 실패!"
    exit 1
fi
echo ""

# 2. XGBoost 실행
echo "2️⃣  XGBoost 학습 및 예측 중..."
python run_xg.py
if [ $? -ne 0 ]; then
    echo "❌ XGBoost 실행 실패!"
    exit 1
fi
echo ""

echo "========================================"
echo "✅ 전체 파이프라인 완료!"
echo "========================================"
echo ""
echo "생성된 파일:"
echo "  - prophet_features.csv (Prophet 출력)"
echo "  - xgboost_results.csv (최종 결과)"
echo "  - xgboost_model.pkl (학습된 모델)"
