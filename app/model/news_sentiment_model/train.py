"""
옥수수 선물 가격 예측 모델 학습 코드
학습된 모델, PCA 객체, 피처 컬럼을 저장합니다.
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
import pickle

warnings.filterwarnings("ignore")

# 모델링 라이브러리
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.decomposition import PCA
import xgboost as xgb

# 공통 전처리 모듈
from preprocessing import (
    preprocess_news_data,
    aggregate_daily_news,
    align_news_to_trading_days,
    create_target_labels,
    merge_news_and_price,
    create_lag_features,
    get_feature_columns,
    preprocess_price_data,
)


def load_data(news_path="corn_all_news_with_sentiment.csv", price_path="corn_future_price.csv"):
    """
    데이터 로드

    Args:
        news_path: 뉴스 데이터 경로
        price_path: 가격 데이터 경로

    Returns:
        news_df, price_df
    """
    print("=" * 80)
    print("데이터 로딩 중...")
    print("=" * 80)

    news_df = pd.read_csv(news_path)
    price_df = pd.read_csv(price_path)

    print(f"뉴스 데이터 shape: {news_df.shape}")
    print(f"가격 데이터 shape: {price_df.shape}")

    return news_df, price_df


def prepare_training_data(news_df, price_df):
    """
    학습 데이터 준비

    Args:
        news_df: 원본 뉴스 데이터
        price_df: 원본 가격 데이터

    Returns:
        merged_df: 병합 및 전처리된 데이터프레임
        pca: 학습된 PCA 객체
    """
    print("\n" + "=" * 80)
    print("데이터 전처리 시작...")
    print("=" * 80)

    # 1. 뉴스 데이터 전처리
    print("\n뉴스 데이터 전처리 중...")
    news_processed = preprocess_news_data(news_df, filter_status=True)
    print(f"필터링 후 뉴스 데이터 shape: {news_processed.shape}")

    # 2. 일별 뉴스 집계
    print("\n일별 뉴스 집계 중...")
    daily_news = aggregate_daily_news(news_processed, embedding_dim=512)
    print(f"집계된 일별 뉴스 데이터 shape: {daily_news.shape}")

    # 3. 가격 데이터 전처리 (ret_1d 계산 포함)
    print("\n가격 데이터 전처리 중...")
    price_df = preprocess_price_data(price_df, time_column="time")
    print(f"가격 데이터 기간: {price_df['date'].min()} ~ {price_df['date'].max()}")
    print("✓ ret_1d (일일 수익률) 계산 완료")

    # 4. 날짜 보정 (주말/휴일 뉴스 처리)
    print("\n날짜 보정 (주말/휴일 뉴스 → 다음 거래일 반영)...")
    daily_news_aligned = align_news_to_trading_days(daily_news, price_df)
    print(f"날짜 정렬 후 뉴스 데이터 shape: {daily_news_aligned.shape}")

    # 5. 타겟 라벨링
    print("\n타겟 라벨링 (0.5% 임계값 적용)...")
    price_df = create_target_labels(price_df, threshold=0.005, method="fixed")
    print(f"타겟 분포:\n{price_df['target'].value_counts()}")
    print(f"상승(1) 비율: {price_df['target'].mean():.2%}")

    # 6. 뉴스와 가격 데이터 병합
    print("\n데이터 병합 중...")
    merged_df = merge_news_and_price(price_df, daily_news_aligned)
    print(f"병합된 데이터 shape: {merged_df.shape}")

    # 7. PCA를 통한 임베딩 차원 축소
    print("\nPCA를 통한 임베딩 차원 축소 (512 → 50)...")
    emb_raw_cols = [f"emb_raw_{i}" for i in range(512)]

    # 임베딩 컬럼 누락 시 0으로 채우기
    for col in emb_raw_cols:
        if col not in merged_df.columns:
            merged_df[col] = 0.0

    embeddings_matrix = merged_df[emb_raw_cols].fillna(0).values
    print(f"임베딩 매트릭스 shape: {embeddings_matrix.shape}")

    # PCA 적용 (fit_transform)
    pca = PCA(n_components=50, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_matrix)

    print(f"PCA 후 shape: {embeddings_pca.shape}")
    print(f"설명된 분산 비율: {pca.explained_variance_ratio_.sum():.2%}")

    # PCA 결과를 DataFrame에 추가
    for i in range(50):
        merged_df[f"emb_pca_{i}"] = embeddings_pca[:, i]

    # 8. Lag 피처 생성
    print("\n시계열 윈도우 적용 (Lag 및 이동평균)...")
    merged_df = create_lag_features(merged_df, embedding_dim=50)
    print(f"최종 데이터 shape: {merged_df.shape}")
    print(f"데이터 기간: {merged_df['date'].min()} ~ {merged_df['date'].max()}")

    return merged_df, pca


def train_model(X_train, y_train, X_test, y_test):
    """
    XGBoost 모델 학습

    Args:
        X_train, y_train: 학습 데이터
        X_test, y_test: 테스트 데이터

    Returns:
        trained_model: 학습된 XGBoost 모델
        metrics: 평가 메트릭 딕셔너리
    """
    print("\n" + "=" * 80)
    print("XGBoost 모델 학습 중...")
    print("=" * 80)

    # 클래스 불균형 처리
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 예측 및 평가
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    print("\n[XGBoost 모델 결과]")
    print("-" * 80)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (상승 예측 정확도): {metrics['precision']:.4f}")
    print(f"Recall (상승 탐지율): {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["하락(0)", "상승(1)"], zero_division=0))

    return xgb_model, metrics


def save_model_artifacts(model, pca, feature_columns, output_dir="models"):
    """
    모델, PCA 객체, 피처 컬럼 저장

    Args:
        model: 학습된 XGBoost 모델
        pca: 학습된 PCA 객체
        feature_columns: 피처 컬럼 리스트
        output_dir: 저장 디렉토리
    """
    print("\n" + "=" * 80)
    print("모델 및 전처리 객체 저장 중...")
    print("=" * 80)

    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 1. XGBoost 모델 저장 (JSON 형식)
    model_path = os.path.join(output_dir, "xgb_model.json")
    model.save_model(model_path)
    print(f"✓ 모델 저장: {model_path}")

    # 2. PCA 객체 저장 (pickle)
    pca_path = os.path.join(output_dir, "pca_transformer.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"✓ PCA 객체 저장: {pca_path}")

    # 3. 피처 컬럼 저장 (JSON)
    feature_path = os.path.join(output_dir, "feature_columns.json")
    with open(feature_path, "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"✓ 피처 컬럼 저장: {feature_path}")

    print("\n모든 파일이 성공적으로 저장되었습니다!")
    print(f"저장 위치: {os.path.abspath(output_dir)}/")


def main():
    """메인 학습 파이프라인"""

    # 1. 데이터 로드
    news_df, price_df = load_data()

    # 2. 학습 데이터 준비
    merged_df, pca = prepare_training_data(news_df, price_df)

    # 3. 피처와 타겟 분리
    print("\n" + "=" * 80)
    print("피처 준비 중...")
    print("=" * 80)

    feature_cols = get_feature_columns(embedding_dim=50)

    X = merged_df[feature_cols].values
    y = merged_df["target"].values
    dates = merged_df["date"].values

    print(f"피처 shape: {X.shape}")
    print(f"타겟 shape: {y.shape}")
    print(f"총 피처 개수: {len(feature_cols)}")

    # 4. Train/Test 분할 (시계열 기반)
    print("\nTrain/Test 분할 중...")
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    print(f"Train 데이터: {X_train.shape}, 기간: {dates_train[0]} ~ {dates_train[-1]}")
    print(f"Test 데이터: {X_test.shape}, 기간: {dates_test[0]} ~ {dates_test[-1]}")
    print(f"\nTrain 타겟 분포: {np.bincount(y_train)}")
    print(f"Test 타겟 분포: {np.bincount(y_test)}")

    # 5. 모델 학습
    model, metrics = train_model(X_train, y_train, X_test, y_test)

    # 6. Feature Importance 분석
    print("\n" + "=" * 80)
    print("주요 피처 분석 (Top 20)")
    print("=" * 80)

    feature_importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    print("\n" + feature_importance.head(20).to_string(index=False))

    # 7. 모델 및 전처리 객체 저장
    save_model_artifacts(model, pca, feature_cols, output_dir="models")

    # 8. 학습 완료 요약
    print("\n" + "=" * 80)
    print("학습 완료!")
    print("=" * 80)
    print(f"총 데이터 포인트: {len(merged_df)}")
    print(f"학습 데이터: {len(X_train)}, 테스트 데이터: {len(X_test)}")
    print(f"총 피처 개수: {len(feature_cols)}")
    print("\n최종 성능 (Test Set):")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
    print("\n핵심 개선 사항:")
    print("  1. 타겟 라벨링: 0.5% 임계값 적용으로 노이즈 감소")
    print("  2. Lag 피처: T, T-1, T-2 감성 지표 추가")
    print("  3. 이동평균: 3일, 5일 MA로 트렌드 포착")
    print("  4. PCA 임베딩: 512 → 50차원 축소 + Lag 적용")
    print("  5. 날짜 보정: merge_asof로 주말/휴일 뉴스 처리")
    print("=" * 80)


if __name__ == "__main__":
    main()
