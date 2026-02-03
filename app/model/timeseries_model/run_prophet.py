"""
Prophet 모델 실행 및 Features 추출
=====================================
원본 데이터를 읽어서 Prophet으로 예측하고,
생성된 features를 CSV로 저장합니다.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import yaml
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

# 재현성을 위한 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_config(config_path="config.yaml"):
    """YAML 설정 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_and_preprocess_data(filepath):
    """데이터 로드 및 전처리"""
    print("📊 데이터 로딩 중...")
    df = pd.read_csv(filepath)

    df["ds"] = pd.to_datetime(df["time"])
    df["y"] = df["close"]

    df = df[["ds", "y", "Volume", "EMA"]].copy()

    df = df.sort_values("ds").reset_index(drop=True)

    print(f"✅ 데이터 로드 완료: {len(df)} 행")
    print(f"기간: {df['ds'].min()} ~ {df['ds'].max()}\n")

    return df


def create_lag_features(df, lag_periods):
    df = df.copy()

    for lag in lag_periods:
        df[f"Volume_lag{lag}"] = df["Volume"].shift(lag)
        df[f"EMA_lag{lag}"] = df["EMA"].shift(lag)

    # NaN 제거
    df = df.dropna().reset_index(drop=True)

    print(f"✅ Lag Features 생성 완료: {len(df)} 행\n")

    return df


def extract_prophet_features_walkforward(df, config):
    """
    Walk-Forward 방식으로 Prophet Features 추출
    각 시점마다 과거 데이터로 학습하고 다음 시점을 예측
    """
    prophet_config = config["prophet"]
    train_window_days = int(prophet_config["train_window_years"] * 365)
    regressor_columns = prophet_config["regressors"]

    print("Walk-Forward")
    print(
        f"학습 윈도우: {prophet_config['train_window_years']}년 ({train_window_days}일)"
    )

    prophet_features_list = []
    start_idx = train_window_days

    with tqdm(total=len(df) - start_idx - 1, desc="Prophet") as pbar:
        for i in range(start_idx, len(df) - 1):
            train_start_idx = max(0, i - train_window_days)
            train_end_idx = i
            train_subset = df.iloc[train_start_idx:train_end_idx].copy()

            test_subset = df.iloc[i + 1 : i + 2].copy()

            # Prophet 모델 생성
            model = Prophet(
                seasonality_mode=prophet_config["seasonality_mode"],
                changepoint_prior_scale=prophet_config["changepoint_prior_scale"],
                yearly_seasonality=prophet_config["yearly_seasonality"],
                weekly_seasonality=prophet_config["weekly_seasonality"],
                daily_seasonality=prophet_config["daily_seasonality"],
            )
            for col in regressor_columns:
                model.add_regressor(col, mode=prophet_config["regressor_mode"])

            # 모델 학습
            model.fit(train_subset[["ds", "y"] + regressor_columns])

            # 예측
            future = test_subset[["ds"] + regressor_columns].copy()
            forecast = model.predict(future)

            # Features 추출
            prophet_feat = forecast[
                ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
            ].copy()

            if "weekly" in forecast.columns:
                prophet_feat["weekly"] = forecast["weekly"]
            if "yearly" in forecast.columns:
                prophet_feat["yearly"] = forecast["yearly"]

            if "extra_regressors_multiplicative" in forecast.columns:
                prophet_feat["extra_regressors_multiplicative"] = forecast[
                    "extra_regressors_multiplicative"
                ]

            for reg in regressor_columns:
                if reg in forecast.columns:
                    prophet_feat[f"{reg}_effect"] = forecast[reg]

            prophet_feat["y"] = test_subset["y"].values[0]

            for col in test_subset.columns:
                if col not in prophet_feat.columns and col not in ["ds", "y"]:
                    prophet_feat[col] = test_subset[col].values[0]

            prophet_features_list.append(prophet_feat)
            pbar.update(1)

    # DataFrame으로 변환
    prophet_features_df = pd.concat(prophet_features_list, ignore_index=True)
    print(f"\n✅ Prophet features 추출 완료: {len(prophet_features_df)} 행")

    return prophet_features_df


def create_target_variable(df):
    """타겟 변수 생성 (전날 대비 상승=1, 하락=0)"""
    df = df.copy()

    # 전날 대비 변화량
    df["y_change"] = df["y"].diff()

    # 방향 (상승=1, 하락=0)
    df["direction"] = (df["y_change"] > 0).astype(int)

    # 보합(변화 없음) 제거
    df = df[df["y_change"] != 0].copy()

    # 첫 행(NaN) 제거
    df = df.dropna(subset=["y_change"]).reset_index(drop=True)

    print(f"✅ 타겟 변수 생성 완료: {len(df)} 행 (보합 제외)")

    return df


def main():
    config = load_config("config.yaml")

    df = load_and_preprocess_data(config["data"]["input_csv"])

    df = create_lag_features(df, config["prophet"]["lag_periods"])

    validation_mode = config["validation"]["mode"]

    if validation_mode == "walk_forward":
        prophet_features_df = extract_prophet_features_walkforward(df, config)

    prophet_features_df = create_target_variable(prophet_features_df)

    output_path = config["data"]["prophet_output_csv"]
    prophet_features_df.to_csv(output_path, index=False)
    print(f"\nResult saved: {output_path}")

    print("\n" + "=" * 70)
    print("📊 추출된 Features 요약")
    print("=" * 70)
    print(f"총 행 수: {len(prophet_features_df)}")
    print(f"총 컬럼 수: {len(prophet_features_df.columns)}")
    print(f"\nFeatures: {list(prophet_features_df.columns)}")
    print(
        f"\n상승(1): {(prophet_features_df['direction'] == 1).sum()}개 ({(prophet_features_df['direction'] == 1).mean() * 100:.1f}%)"
    )
    print(
        f"하락(0): {(prophet_features_df['direction'] == 0).sum()}개 ({(prophet_features_df['direction'] == 0).mean() * 100:.1f}%)"
    )

    return prophet_features_df


if __name__ == "__main__":
    results = main()
