"""
공통 전처리 모듈
학습과 추론에서 동일하게 사용되는 전처리 로직
"""

import pandas as pd
import numpy as np
import ast


def parse_embedding(embedding_str):
    """문자열 형태의 임베딩을 numpy 배열로 변환"""
    try:
        if isinstance(embedding_str, str):
            embedding_list = ast.literal_eval(embedding_str)
            return np.array(embedding_list, dtype=np.float32)
        elif isinstance(embedding_str, (list, np.ndarray)):
            return np.array(embedding_str, dtype=np.float32)
        else:
            return np.zeros(512, dtype=np.float32)
    except:
        return np.zeros(512, dtype=np.float32)


def preprocess_news_data(news_df, filter_status=True):
    """
    뉴스 데이터 전처리

    Args:
        news_df: 원본 뉴스 데이터프레임
        filter_status: filter_status가 'T'인 데이터만 필터링할지 여부

    Returns:
        전처리된 뉴스 데이터프레임
    """
    news_df = news_df.copy()

    # filter_status가 'T'인 데이터만 필터링
    if filter_status and "filter_status" in news_df.columns:
        news_df = news_df[news_df["filter_status"] == "T"].copy()

    # 날짜 변환
    news_df["publish_date"] = pd.to_datetime(news_df["publish_date"])
    news_df["date"] = news_df["publish_date"].dt.date

    # article_embedding 문자열을 리스트로 변환
    if "article_embedding" in news_df.columns:
        news_df["embedding_array"] = news_df["article_embedding"].apply(parse_embedding)

    return news_df


def aggregate_daily_news(news_df, embedding_dim=512):
    """
    일별 뉴스 집계

    Args:
        news_df: 전처리된 뉴스 데이터프레임
                 (※ finbert.py로 감성 분석이 완료된 데이터)
        embedding_dim: 임베딩 차원 (기본 512)

    Returns:
        일별로 집계된 뉴스 데이터프레임

    Note:
        다음 컬럼들은 finbert.py를 통해 미리 생성되어 있어야 합니다:
        - price_impact_score: positive_score - negative_score
        - sentiment_confidence: 감성 분석 신뢰도
        - positive_score: 긍정 점수 (0~1)
        - negative_score: 부정 점수 (0~1)
        - article_embedding: 512차원 임베딩 벡터
    """
    # 날짜별 그룹화
    daily_news = (
        news_df.groupby("date")
        .agg(
            {
                "price_impact_score": "mean",
                "sentiment_confidence": "mean",
                "positive_score": "mean",
                "negative_score": "mean",
                "article_embedding": "count",
            }
        )
        .reset_index()
    )

    daily_news.rename(columns={"article_embedding": "news_count"}, inplace=True)

    # 임베딩 평균 계산
    def calculate_mean_embedding(group):
        """그룹의 임베딩 평균 벡터 계산"""
        embeddings = np.stack(group["embedding_array"].values)
        return embeddings.mean(axis=0)

    embedding_mean = (
        news_df.groupby("date").apply(calculate_mean_embedding).reset_index()
    )
    embedding_mean.columns = ["date", "mean_embedding"]

    # 임베딩을 512개의 개별 컬럼으로 즉시 분리
    embedding_df = pd.DataFrame(
        embedding_mean["mean_embedding"].tolist(),
        columns=[f"emb_raw_{i}" for i in range(embedding_dim)],
    )
    embedding_df["date"] = embedding_mean["date"].values

    # daily_news에 임베딩 개별 컬럼 병합
    daily_news = daily_news.merge(embedding_df, on="date", how="left")

    return daily_news


def preprocess_price_data(price_df, time_column="time"):
    """
    가격 데이터 전처리 및 수익률 계산

    Args:
        price_df: 원본 가격 데이터프레임
        time_column: 시간 컬럼명 (기본: 'time')

    Returns:
        전처리된 가격 데이터프레임 (ret_1d 컬럼 추가됨)

    Note:
        ret_1d는 다음 공식으로 계산됩니다:
        ret_1d = log(close_today / close_yesterday)

        즉, 어제 종가 대비 오늘 종가의 로그 수익률입니다.
    """
    price_df = price_df.copy()

    # 시간 컬럼 변환
    if time_column in price_df.columns:
        price_df["time"] = pd.to_datetime(price_df[time_column])
        price_df["date"] = price_df["time"].dt.date
    elif "date" in price_df.columns:
        price_df["date"] = pd.to_datetime(price_df["date"])
    else:
        raise ValueError(f"'{time_column}' 또는 'date' 컬럼이 필요합니다.")

    # 날짜 기준으로 정렬 (중요!)
    price_df = price_df.sort_values("date").reset_index(drop=True)

    # ret_1d가 없으면 계산
    if "ret_1d" not in price_df.columns:
        # 일일 수익률 계산: log(close_t / close_t-1)
        # shift(-1)이 아닌 그냥 계산 (현재가 / 이전가)
        price_df["ret_1d"] = np.log(price_df["close"] / price_df["close"].shift(1))

        # 첫 번째 행은 NaN이므로 제거하거나 0으로 채우기
        price_df["ret_1d"] = price_df["ret_1d"].fillna(0)

    # 날짜를 datetime으로 통일
    price_df["date"] = pd.to_datetime(price_df["date"])

    return price_df


def align_news_to_trading_days(daily_news, price_df):
    """
    뉴스 데이터를 거래일에 맞춰 정렬 (주말/휴일 뉴스를 다음 거래일에 반영)

    Args:
        daily_news: 일별 집계된 뉴스 데이터프레임
        price_df: 가격 데이터프레임

    Returns:
        거래일에 정렬된 뉴스 데이터프레임
    """
    # 가격 데이터의 모든 거래일을 기준으로 리인덱싱
    all_trading_days = pd.DataFrame({"date": price_df["date"].unique()})
    all_trading_days = all_trading_days.sort_values("date").reset_index(drop=True)

    # 뉴스 데이터를 거래일 기준으로 병합
    daily_news["date"] = pd.to_datetime(daily_news["date"])
    all_trading_days["date"] = pd.to_datetime(all_trading_days["date"])

    # merge_asof를 사용하여 뉴스 날짜를 다음 거래일에 매칭
    daily_news_aligned = pd.merge_asof(
        all_trading_days.sort_values("date"),
        daily_news.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("7 days"),  # 최대 7일 이내의 뉴스만 반영
    )

    return daily_news_aligned


def create_target_labels(price_df, threshold=0.005, method="fixed"):
    """
    타겟 라벨 생성

    Args:
        price_df: 가격 데이터프레임
        threshold: 임계값 (기본 0.5% = 0.005)
        method: 'fixed' 또는 'dynamic'

    Returns:
        타겟이 추가된 가격 데이터프레임
    """
    price_df = price_df.copy()

    # 다음 날 수익률 계산
    price_df["next_day_ret"] = price_df["ret_1d"].shift(-1)

    if method == "fixed":
        # 고정 임계값
        price_df["target"] = (price_df["next_day_ret"] > threshold).astype(int)
    else:
        # 동적 임계값 (최근 20일 표준편차의 0.5배)
        rolling_std = price_df["ret_1d"].rolling(window=20, min_periods=10).std()
        price_df["threshold_dynamic"] = rolling_std * 0.5
        price_df["target"] = (
            price_df["next_day_ret"] > price_df["threshold_dynamic"]
        ).astype(int)

    # 마지막 행은 target이 NaN이므로 제거
    price_df = price_df[price_df["target"].notna()].copy()

    return price_df


def merge_news_and_price(price_df, daily_news_aligned):
    """
    뉴스와 가격 데이터 병합

    Args:
        price_df: 가격 데이터프레임
        daily_news_aligned: 거래일에 정렬된 뉴스 데이터프레임

    Returns:
        병합된 데이터프레임
    """
    # 날짜 형식 통일
    price_df["date"] = pd.to_datetime(price_df["date"])
    daily_news_aligned["date"] = pd.to_datetime(daily_news_aligned["date"])

    # 병합
    merged_df = price_df[["date", "close", "ret_1d", "target"]].merge(
        daily_news_aligned, on="date", how="inner"
    )

    # 결측치 처리 (forward fill로 누락된 뉴스 데이터 채우기)
    sentiment_cols = [
        "price_impact_score",
        "sentiment_confidence",
        "positive_score",
        "negative_score",
        "news_count",
    ]
    merged_df[sentiment_cols] = (
        merged_df[sentiment_cols].fillna(method="ffill").fillna(0)
    )

    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    return merged_df


def create_lag_features(df, embedding_dim=50):
    """
    Lag 피처 및 이동평균 생성

    Args:
        df: 데이터프레임 (PCA 임베딩이 포함되어 있어야 함)
        embedding_dim: PCA 후 임베딩 차원 (기본 50)

    Returns:
        Lag 피처가 추가된 데이터프레임
    """
    df = df.copy()

    # Lag 피처 생성 (T, T-1, T-2)
    lag_cols = ["price_impact_score", "positive_score", "negative_score", "news_count"]

    for col in lag_cols:
        for lag in [1, 2]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # 이동 평균 (3일, 5일)
    for col in ["price_impact_score", "positive_score", "negative_score"]:
        df[f"{col}_ma3"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_ma5"] = df[col].rolling(window=5, min_periods=1).mean()

    # PCA 임베딩에 대한 Lag (T, T-1만 사용)
    for i in range(embedding_dim):
        df[f"emb_pca_{i}_lag1"] = df[f"emb_pca_{i}"].shift(1)

    # 초기 Lag로 인한 결측치 제거
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(embedding_dim=50):
    """
    전체 피처 컬럼 리스트 반환

    Args:
        embedding_dim: PCA 후 임베딩 차원 (기본 50)

    Returns:
        피처 컬럼 리스트
    """
    # 기본 컬럼
    base_cols = [
        "price_impact_score",
        "positive_score",
        "negative_score",
        "news_count",
        "sentiment_confidence",
    ]

    # Lag 및 MA 컬럼
    lag_feature_cols = []
    lag_base = ["price_impact_score", "positive_score", "negative_score", "news_count"]
    for col in lag_base:
        for lag in [1, 2]:
            lag_feature_cols.append(f"{col}_lag{lag}")

    for col in ["price_impact_score", "positive_score", "negative_score"]:
        lag_feature_cols.append(f"{col}_ma3")
        lag_feature_cols.append(f"{col}_ma5")

    # PCA 컬럼
    pca_cols = [f"emb_pca_{i}" for i in range(embedding_dim)] + [
        f"emb_pca_{i}_lag1" for i in range(embedding_dim)
    ]

    # 전체 피처 컬럼
    feature_cols = base_cols + lag_feature_cols + pca_cols

    return feature_cols


def prepare_inference_features(
    news_data, price_history, pca_transformer, feature_columns
):
    """
    추론용 데이터 준비 (단일 시점 예측용)

    Args:
        news_data: 최근 뉴스 데이터 (DataFrame, 최소 3일치)
        price_history: 최근 가격 데이터 (DataFrame, 최소 5일치)
        pca_transformer: 학습된 PCA 객체
        feature_columns: 학습 시 사용한 피처 컬럼 리스트

    Returns:
        추론용 피처 배열 (numpy array, shape: (1, n_features))
    """
    # 뉴스 데이터 전처리
    news_processed = preprocess_news_data(news_data, filter_status=False)

    # 일별 뉴스 집계
    daily_news = aggregate_daily_news(news_processed)

    # 가격 데이터 전처리 (ret_1d 계산 포함)
    price_history = preprocess_price_data(price_history, time_column="time")

    # 뉴스와 가격 데이터 병합 (타겟 없이)
    daily_news["date"] = pd.to_datetime(daily_news["date"])
    merged_df = price_history[["date", "close", "ret_1d"]].merge(
        daily_news, on="date", how="left"
    )

    # 결측치 처리
    sentiment_cols = [
        "price_impact_score",
        "sentiment_confidence",
        "positive_score",
        "negative_score",
        "news_count",
    ]
    merged_df[sentiment_cols] = (
        merged_df[sentiment_cols].fillna(method="ffill").fillna(0)
    )
    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    # PCA 적용 (transform만 수행)
    emb_raw_cols = [f"emb_raw_{i}" for i in range(512)]

    # 임베딩 컬럼 누락 시 0으로 채우기
    for col in emb_raw_cols:
        if col not in merged_df.columns:
            merged_df[col] = 0.0

    embeddings_matrix = merged_df[emb_raw_cols].fillna(0).values
    embeddings_pca = pca_transformer.transform(embeddings_matrix)

    # PCA 결과를 DataFrame에 추가
    for i in range(50):
        merged_df[f"emb_pca_{i}"] = embeddings_pca[:, i]

    # Lag 피처 생성
    merged_df = create_lag_features(merged_df, embedding_dim=50)

    # 가장 최근 데이터만 추출 (마지막 row)
    if len(merged_df) == 0:
        raise ValueError("전처리 후 데이터가 없습니다. 입력 데이터를 확인해주세요.")

    latest_data = merged_df.iloc[-1:][feature_columns].values

    return latest_data
