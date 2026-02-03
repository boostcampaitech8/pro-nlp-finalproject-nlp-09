import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import yaml
import pickle
from tqdm import tqdm
import warnings
import random
import os
from typing import Dict, Any
warnings.filterwarnings('ignore')

# 재현성을 위한 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def load_config(config_path=None):
    """YAML 설정 파일 로드"""
    if config_path is None:
        # 기본 경로: 현재 파일과 같은 디렉토리의 config.yaml
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_prophet_features(filepath):
    """Prophet features CSV 로드"""
    print("📂 Prophet features 로딩 중...")
    df = pd.read_csv(filepath)
    df['ds'] = pd.to_datetime(df['ds'])
    print(f"✅ 로드 완료: {len(df)} 행\n")
    return df


def train_xgboost_walkforward(df, config):
    """
    Walk-Forward 방식으로 XGBoost 학습 및 예측
    매 시점마다 모델을 재학습
    """
    xgb_config = config['xgboost']
    validation_config = config['validation']
    min_train_samples = validation_config['min_train_samples']
    window_size = validation_config.get('window_size', None)  
    
    print("\n🚀 Walk-Forward 방식으로 XGBoost 학습 시작...")

    feature_columns = [
        col
        for col in df.columns
        if col not in ["ds", "y", "direction", "y_change", "yhat_lower", "yhat_upper"]
    ]

    print(f"사용할 Features ({len(feature_columns)}개):")
    for col in feature_columns:
        print(f"  - {col}")
    print()
    
    predictions = []
    final_model = None
    
    with tqdm(total=len(df) - min_train_samples, desc="XGBoost 학습 및 예측") as pbar:
        for i in range(min_train_samples, len(df)):
            # Sliding Window 적용
            if window_size is None:
                train_val_start = 0
            else:
                train_val_start = max(0, i - window_size)

            available_samples = i - train_val_start
            train_size_relative = int(available_samples * xgb_config['train_val_split'])
            train_end = train_val_start + train_size_relative
            
            X_train = df.iloc[train_val_start:train_end][feature_columns]
            y_train = df.iloc[train_val_start:train_end]['direction']
            
            X_val = df.iloc[train_end:i][feature_columns]
            y_val = df.iloc[train_end:i]['direction']
            
            X_test = df.iloc[i:i+1][feature_columns]
            y_test = df.iloc[i:i+1]['direction'].values[0]

            n_positive = (y_train == 1).sum()
            n_negative = (y_train == 0).sum()
            scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1

            xgb_params = {
                'objective': xgb_config['objective'],
                'max_depth': xgb_config['max_depth'],
                'learning_rate': xgb_config['learning_rate'],
                'n_estimators': xgb_config['n_estimators'],
                'min_child_weight': xgb_config['min_child_weight'],
                'subsample': xgb_config['subsample'],
                'colsample_bytree': xgb_config['colsample_bytree'],
                'gamma': xgb_config['gamma'],
                'reg_alpha': xgb_config['reg_alpha'],
                'reg_lambda': xgb_config['reg_lambda'],
                'scale_pos_weight': scale_pos_weight,
                'random_state': xgb_config['random_state'],
                'verbosity': 0
            }

            early_stopping_rounds = xgb_config.get('early_stopping_rounds')

            if len(X_val) > 0 and early_stopping_rounds is not None:
                xgb_params['early_stopping_rounds'] = early_stopping_rounds
                xgb_model = XGBClassifier(**xgb_params)
                xgb_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False,
                )
            else:
                xgb_model = XGBClassifier(**xgb_params)
                xgb_model.fit(X_train, y_train)
            
            final_model = xgb_model
            
            y_pred = xgb_model.predict(X_test)[0]
            y_pred_proba = xgb_model.predict_proba(X_test)[0]
            
            train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
            val_acc = (
                accuracy_score(y_val, xgb_model.predict(X_val))
                if len(X_val) > 0
                else 0.0
            )

            result = {
                'ds': df.iloc[i]['ds'],
                'y': df.iloc[i]['y'],
                'y_actual_direction': y_test,
                'y_pred_direction': y_pred,
                'pred_proba_down': y_pred_proba[0],
                'pred_proba_up': y_pred_proba[1],
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'n_estimators_used': xgb_model.get_booster().num_boosted_rounds(),
                'train_size': len(X_train),
                'val_size': len(X_val),
                'scale_pos_weight': scale_pos_weight,
            }

            for col in feature_columns:
                result[col] = df.iloc[i][col]
            
            predictions.append(result)
            pbar.update(1)
    
    results_df = pd.DataFrame(predictions)
    print(f"✅ XGBoost 예측 완료: {len(results_df)} 행")
    
    return results_df, final_model


def calculate_metrics(results_df):
    """성능 지표 계산"""
    y_true = results_df['y_actual_direction'].values
    y_pred = results_df['y_pred_direction'].values
    
    metrics = {
        'test_accuracy': accuracy_score(y_true, y_pred) * 100,
        'test_precision': precision_score(y_true, y_pred, zero_division=0) * 100,
        'test_recall': recall_score(y_true, y_pred, zero_division=0) * 100,
        'test_f1_score': f1_score(y_true, y_pred, zero_division=0) * 100,
    }
    
    # Train/Val 정확도 평균
    if 'train_accuracy' in results_df.columns:
        metrics['train_accuracy_mean'] = results_df['train_accuracy'].mean() * 100
    if 'val_accuracy' in results_df.columns:
        metrics['val_accuracy_mean'] = results_df['val_accuracy'].mean() * 100
    
    # 과적합 갭
    if "train_accuracy" in results_df.columns:
        metrics["overfit_gap"] = (
            metrics["train_accuracy_mean"] - metrics["test_accuracy"]
        )

    # 평균 사용 트리
    if 'n_estimators_used' in results_df.columns:
        metrics['avg_n_estimators_used'] = results_df['n_estimators_used'].mean()
    
    return metrics


def analyze_feature_importance(model, feature_columns, top_n=20):
    print("🔍 Feature Importance 분석")

    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)

    total_importance = importance_df["importance"].sum()
    importance_df["importance_pct"] = (
        importance_df["importance"] / total_importance
    ) * 100
    importance_df["cumulative_pct"] = importance_df["importance_pct"].cumsum()

    # 상위 N개 출력
    print(f"\n상위 {min(top_n, len(importance_df))}개 중요 Features:")
    print("-" * 70)
    print(f"{'순위':<6} {'Feature':<30} {'중요도':<12} {'비율':<10} {'누적':<10}")
    print("-" * 70)
    
    for idx, row in importance_df.head(top_n).iterrows():
        rank = importance_df.index.get_loc(idx) + 1
        print(f"{rank:<6} {row['feature']:<30} {row['importance']:<12.6f} {row['importance_pct']:>8.2f}% {row['cumulative_pct']:>8.2f}%")
    
    # 상위 80% 중요도를 차지하는 feature 개수
    n_80pct = (importance_df['cumulative_pct'] <= 80).sum()
    print(f"\n💡 상위 {n_80pct}개 feature가 전체 중요도의 80%를 차지합니다.")
    
    # 중요도 타입별 분석
    print("\n" + "=" * 70)
    print("📊 중요도 상세 분석")
    print("=" * 70)
    
    try:
        # get_score()로 다양한 중요도 지표 확인
        booster = model.get_booster()
        
        # weight: 해당 feature가 트리 분할에 사용된 횟수
        score_weight = booster.get_score(importance_type='weight')
        # gain: 해당 feature로 분할할 때 평균 gain(손실 감소량)
        score_gain = booster.get_score(importance_type='gain')
        # cover: 해당 feature가 커버하는 샘플 수
        score_cover = booster.get_score(importance_type='cover')
        
        print("\n중요도 계산 방식 비교 (상위 5개):")
        print("-" * 70)
        
        for i, row in importance_df.head(5).iterrows():
            feat = row['feature']
            # XGBoost 내부에서는 f0, f1, ... 형식으로 저장됨
            feat_idx = f"f{feature_columns.index(feat)}"
            
            print(f"\n{i+1}. {feat}")
            print(f"   - Default (gain):  {row['importance']:.6f}")
            if feat_idx in score_weight:
                print(f"   - Weight (횟수):   {score_weight[feat_idx]:.0f}")
            if feat_idx in score_gain:
                print(f"   - Gain (손실감소): {score_gain[feat_idx]:.6f}")
            if feat_idx in score_cover:
                print(f"   - Cover (샘플수):  {score_cover[feat_idx]:.0f}")
    
    except Exception as e:
        print(f"\n⚠️  상세 분석 중 오류: {e}")
    
    return importance_df


def analyze_yearly_performance(results_df):
    """연도별 성능 분석"""
    # 연도 추출
    results_df['year'] = pd.to_datetime(results_df['ds']).dt.year
    
    yearly_stats = []
    for year in sorted(results_df['year'].unique()):
        year_data = results_df[results_df['year'] == year]
        y_true = year_data['y_actual_direction'].values
        y_pred = year_data['y_pred_direction'].values
        
        accuracy = accuracy_score(y_true, y_pred) * 100
        count = len(year_data)

        yearly_stats.append(
            {
                "year": year,
                "accuracy": accuracy,
                "count": count,
                "correct": (y_true == y_pred).sum(),
            }
        )

    return pd.DataFrame(yearly_stats)


def print_results(results_df, metrics, config):
    """결과 출력"""
    print("\n" + "=" * 70)
    print("📊 XGBoost 분류 성능 지표")
    print("=" * 70)
    
    # Test 성능
    print("\n[Test 성능]")
    print(f"  Accuracy:  {metrics['test_accuracy']:.2f}%")
    print(f"  Precision: {metrics['test_precision']:.2f}%")
    print(f"  Recall:    {metrics['test_recall']:.2f}%")
    print(f"  F1-Score:  {metrics['test_f1_score']:.2f}%")
    
    # Train/Val 성능
    if 'train_accuracy_mean' in metrics:
        print("\n[Train/Val 성능]")
        print(f"  Train Accuracy (평균): {metrics['train_accuracy_mean']:.2f}%")
        if 'val_accuracy_mean' in metrics:
            print(f"  Val Accuracy (평균):   {metrics['val_accuracy_mean']:.2f}%")
        if 'overfit_gap' in metrics:
            gap = metrics['overfit_gap']
            print(f"  Overfit Gap:           {gap:+.2f}%p", end="")
            if gap > 10:
                print("  ⚠️  과적합 의심!")
            elif gap > 5:
                print("  ⚠️  약간 과적합")
            else:
                print("  ✅ 정상")
    
    # 모델 정보
    if "avg_n_estimators_used" in metrics:
        print("\n[모델 정보]")
        print(
            f"  평균 사용 트리 개수: {metrics['avg_n_estimators_used']:.1f}/{config['xgboost']['n_estimators']}"
        )

    # 혼동 행렬
    y_true = results_df['y_actual_direction'].values
    y_pred = results_df['y_pred_direction'].values
    
    print("\n" + "=" * 70)
    print("📋 혼동 행렬")
    print("=" * 70)
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n실제 하락(0) / 예측 하락(0): {cm[0][0]}")
    print(f"실제 하락(0) / 예측 상승(1): {cm[0][1]}")
    print(f"실제 상승(1) / 예측 하락(0): {cm[1][0]}")
    print(f"실제 상승(1) / 예측 상승(1): {cm[1][1]}")
    
    print("\n" + "=" * 70)
    print("📈 상세 분류 리포트")
    print("=" * 70)
    print(
        "\n"
        + classification_report(
            y_true, y_pred, target_names=["하락(0)", "상승(1)"], digits=4
        )
    )

    # 연도별 성능 분석
    print("\n" + "=" * 70)
    print("📅 연도별 성능 분석")
    print("=" * 70)
    
    yearly_df = analyze_yearly_performance(results_df)

    print(
        f"\n{'연도':<8} {'정확도':<12} {'예측 횟수':<12} {'정답 횟수':<12} {'트렌드':<10}"
    )
    print("-" * 70)
    
    for idx, row in yearly_df.iterrows():
        year = int(row['year'])
        acc = row['accuracy']
        count = int(row['count'])
        correct = int(row['correct'])
        
        # 트렌드 표시
        if idx > 0:
            prev_acc = yearly_df.iloc[idx-1]['accuracy']
            diff = acc - prev_acc
            if diff > 2:
                trend = f"↗️ +{diff:.1f}%"
            elif diff < -2:
                trend = f"↘️ {diff:.1f}%"
            else:
                trend = "→ 유사"
        else:
            trend = "-"

        print(
            f"{year:<8} {acc:>7.2f}%    {count:>8}개    {correct:>8}개    {trend:<10}"
        )

    # 초반/후반 비교
    if len(yearly_df) >= 2:
        print("\n" + "-" * 70)
        n_years = len(yearly_df)
        split_point = n_years // 2
        
        early_years = yearly_df.iloc[:split_point]
        late_years = yearly_df.iloc[split_point:]

        early_acc = early_years["correct"].sum() / early_years["count"].sum() * 100
        late_acc = late_years["correct"].sum() / late_years["count"].sum() * 100

        early_period = (
            f"{int(early_years.iloc[0]['year'])}~{int(early_years.iloc[-1]['year'])}"
        )
        late_period = (
            f"{int(late_years.iloc[0]['year'])}~{int(late_years.iloc[-1]['year'])}"
        )

        print(f"\n초반 ({early_period}): {early_acc:.2f}%")
        print(f"후반 ({late_period}): {late_acc:.2f}%")
        
        diff = late_acc - early_acc
        if diff > 2:
            print(f"\n💡 후반으로 갈수록 성능 향상! (+{diff:.2f}%p)")
        elif diff < -2:
            print(f"\n⚠️  후반으로 갈수록 성능 저하... ({diff:.2f}%p)")
        else:
            print(f"\n→ 초반과 후반 성능 비슷함 ({diff:+.2f}%p)")
    
    # 최고/최저 연도
    if len(yearly_df) > 0:
        best_year = yearly_df.loc[yearly_df["accuracy"].idxmax()]
        worst_year = yearly_df.loc[yearly_df["accuracy"].idxmin()]

        print(
            f"\n✅ 최고 성능: {int(best_year['year'])}년 ({best_year['accuracy']:.2f}%)"
        )
        print(
            f"❌ 최저 성능: {int(worst_year['year'])}년 ({worst_year['accuracy']:.2f}%)"
        )
        print(f"   성능 편차: {best_year['accuracy'] - worst_year['accuracy']:.2f}%p")


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("XGBoost 분류 모델 실행")
    print("=" * 70)
    
    # 1. 설정 로드
    config = load_config('config.yaml')
    print("✅ 설정 로드 완료\n")
    
    # 2. Prophet features 로드
    df = load_prophet_features(config['data']['prophet_output_csv'])
    
    # Feature 컬럼 정의 (나중에 importance 분석에 사용)
    feature_columns = [
        col
        for col in df.columns
        if col not in ["ds", "y", "direction", "y_change", "yhat_lower", "yhat_upper"]
    ]

    # 3. XGBoost 학습 및 예측 (검증 방식에 따라)
    validation_mode = config['validation']['mode']
    
    if validation_mode == 'walk_forward':
        results_df, model = train_xgboost_walkforward(df, config)
    elif validation_mode == 'fixed_test':
        raise NotImplementedError("fixed_test 모드는 아직 구현되지 않았습니다. walk_forward 모드를 사용해주세요.")
    else:
        raise ValueError(f"지원하지 않는 validation_mode: {validation_mode}")
    
    # 4. 성능 지표 계산
    metrics = calculate_metrics(results_df)
    
    # 5. 결과 출력
    print_results(results_df, metrics, config)
    
    # 6. Feature Importance 분석
    importance_df = analyze_feature_importance(model, feature_columns, top_n=20)
    
    # 8. 모델 저장
    if config['output']['save_model']:
        model_path = config['data']['model_output_pkl']
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 모델 저장 완료: {model_path}")
    
    print("\n" + "=" * 70)
    print("✅ XGBoost 작업 완료!")
    print("=" * 70)
    
    return results_df, model, importance_df


class TimeSeriesXGBoostInference:
    """
    BigQuery에서 가져온 DataFrame을 사용하여 Walk-Forward 방식으로 학습하고 예측하는 클래스
    inference.py와 동일한 인터페이스를 제공합니다.
    """
    def __init__(self, config_path=None):
        """
        추론 엔진 초기화
        
        Args:
            config_path (str, optional): config.yaml 파일 경로. None이면 기본 경로 사용.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = load_config(config_path)
        self.xgb_config = self.config['xgboost']
        self.validation_config = self.config['validation']
        
    def predict(self, history_df: pd.DataFrame, target_date: str) -> Dict[str, Any]:
        """
        제공된 과거 데이터를 사용하여 Walk-Forward 방식으로 학습하고 시장 방향을 예측합니다.
        
        Args:
            history_df (pd.DataFrame): Prophet 피처가 포함된 데이터프레임.
                                    'ds' 컬럼과 학습에 사용된 모든 피처가 포함되어야 합니다.
                                    target_date를 포함한 과거 데이터가 있어야 합니다.
            target_date (str): 예측할 날짜 문자열 ('YYYY-MM-DD' 형식).
            
        Returns:
            Dict: 예측 상세 결과 사전 (inference.py와 동일한 형식).
        """
        try:
            target_ts = pd.Timestamp(target_date)
        except ValueError:
            raise ValueError(f"잘못된 날짜 형식입니다: {target_date}. YYYY-MM-DD 형식을 사용하세요.")
        
        # 'ds' 컬럼이 datetime 형식인지 확인
        if not pd.api.types.is_datetime64_any_dtype(history_df['ds']):
            history_df['ds'] = pd.to_datetime(history_df['ds'])
        
        # 타겟 날짜에 해당하는 행 찾기
        target_idx = history_df[history_df['ds'] == target_ts].index
        
        if len(target_idx) == 0:
            raise ValueError(f"제공된 데이터프레임에서 해당 날짜({target_date})의 데이터를 찾을 수 없습니다.")
        
        target_idx = target_idx[0]
        target_row_idx = history_df.index.get_loc(target_idx)
        
        # 피처 컬럼 정의
        exclude_cols = ['ds', 'y', 'direction', 'y_change', 'yhat_lower', 'yhat_upper']
        feature_columns = [col for col in history_df.columns if col not in exclude_cols]
        
        # Walk-Forward 학습을 위한 데이터 준비
        min_train_samples = self.validation_config['min_train_samples']
        window_size = self.validation_config.get('window_size', None)
        train_val_split = self.xgb_config['train_val_split']
        
        # 타겟 날짜까지의 데이터만 사용 (미래 데이터는 사용하지 않음)
        df_until_target = history_df.iloc[:target_row_idx + 1].copy()
        
        if len(df_until_target) < min_train_samples:
            raise ValueError(
                f"학습에 필요한 최소 샘플 수({min_train_samples})보다 적습니다. "
                f"현재 사용 가능한 샘플 수: {len(df_until_target)}"
            )
        
        # Sliding Window 적용
        if window_size is None:
            train_val_start = 0
        else:
            train_val_start = max(0, target_row_idx - window_size)
        
        available_samples = target_row_idx - train_val_start
        train_size_relative = int(available_samples * train_val_split)
        train_end = train_val_start + train_size_relative
        
        # Train/Val/Test 분리
        X_train = df_until_target.iloc[train_val_start:train_end][feature_columns]
        y_train = df_until_target.iloc[train_val_start:train_end]['direction']
        
        X_val = df_until_target.iloc[train_end:target_row_idx][feature_columns]
        y_val = df_until_target.iloc[train_end:target_row_idx]['direction']
        
        X_test = df_until_target.iloc[target_row_idx:target_row_idx+1][feature_columns]
        row = df_until_target.iloc[target_row_idx:target_row_idx+1]
        
        # 클래스 불균형 처리
        n_positive = (y_train == 1).sum()
        n_negative = (y_train == 0).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1
        
        # XGBoost 파라미터 설정
        xgb_params = {
            'objective': self.xgb_config['objective'],
            'max_depth': self.xgb_config['max_depth'],
            'learning_rate': self.xgb_config['learning_rate'],
            'n_estimators': self.xgb_config['n_estimators'],
            'min_child_weight': self.xgb_config['min_child_weight'],
            'subsample': self.xgb_config['subsample'],
            'colsample_bytree': self.xgb_config['colsample_bytree'],
            'gamma': self.xgb_config['gamma'],
            'reg_alpha': self.xgb_config['reg_alpha'],
            'reg_lambda': self.xgb_config['reg_lambda'],
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.xgb_config['random_state'],
            'verbosity': 0
        }
        
        # 모델 학습
        early_stopping_rounds = self.xgb_config.get('early_stopping_rounds')
        
        if len(X_val) > 0 and early_stopping_rounds is not None:
            xgb_params['early_stopping_rounds'] = early_stopping_rounds
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        else:
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(X_train, y_train)
        
        # 예측 수행
        prediction_prob = xgb_model.predict_proba(X_test)[0]  # [하락확률, 상승확률]
        prediction = xgb_model.predict(X_test)[0]  # 0 또는 1
        
        confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
        
        # Prophet 예측값 (yhat)
        yhat = row['yhat'].values[0]
        
        # 문맥 통계 (추세 분석용)
        # 제공된 과거 데이터에서 최근 7일 평균 계산
        recent_7_days = df_until_target.tail(7)
        recent_mean = recent_7_days['yhat'].mean()
        
        # 전 기간 평균 계산
        all_time_mean = df_until_target['yhat'].mean()
        
        # inference.py와 동일한 형식으로 결과 반환
        return {
            "target_date": target_date,
            "forecast_value": float(yhat),         # Prophet 예측값
            "forecast_direction": "Up" if prediction == 1 else "Down",
            "confidence_score": float(confidence) * 100,  # 신뢰도 (%)
            "recent_mean_7d": float(recent_mean),  # 최근 7일 평균
            "all_time_mean": float(all_time_mean), # 전체 기간 평균
            "trend_analysis": "Rising" if yhat > recent_mean else "Falling",  # 단순 추세
            "volatility_index": float(recent_7_days['yhat'].std()),  # 변동성 지표 (표준편차)
            "last_observed_value": float(row['y'].values[0]) if 'y' in row.columns and not pd.isna(row['y'].values[0]) else None  # 실제값 (있으면)
        }


if __name__ == "__main__":
    results, model, importance = main()