"""
옥수수 선물 가격 예측 모델 추론 코드
저장된 모델을 로드하여 새로운 데이터에 대한 예측을 수행합니다.
근거 뉴스 추출 기능을 포함하여 예측의 해석 가능성을 제공합니다.
"""

import pandas as pd
import json
import pickle
import warnings
import os
import ast
from datetime import timedelta

# TODO warnings 무시 설정 제거 필요할 수 있음
warnings.filterwarnings("ignore")

import xgboost as xgb

# 공통 전처리 모듈 임포트 (패키지 실행 vs 단독 실행 호환)
try:
    from .preprocessing import prepare_inference_features
except ImportError:
    try:
        from preprocessing import prepare_inference_features
    except ImportError:
        # 패키지 구조에 따른 절대 경로 시도
        from app.model.news_sentiment_model.preprocessing import (
            prepare_inference_features,
        )


# ============================================
# 근거 뉴스 추출 유틸리티 함수
# ============================================


def parse_triples(triples_str):
    """
    triples 문자열을 파싱하여 읽기 쉬운 텍스트로 변환

    Args:
        triples_str: triples 문자열 (리스트 형태의 문자열)

    Returns:
        list: 파싱된 triples 리스트
    """
    if pd.isna(triples_str) or triples_str == "":
        return []

    try:
        # 문자열을 리스트로 변환
        if isinstance(triples_str, str):
            triples = ast.literal_eval(triples_str)
        else:
            triples = triples_str

        if not isinstance(triples, list):
            return []

        return triples
    except:
        return []


def format_triples_as_text(triples):
    """
    triples를 사람이 읽기 쉬운 문장으로 변환

    Args:
        triples: 파싱된 triples 리스트

    Returns:
        str: 읽기 쉬운 텍스트
    """
    if not triples:
        return ""

    sentences = []
    for triple in triples:
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            subject, relation, obj = triple[0], triple[1], triple[2]
            sentence = f"{subject} {relation} {obj}"
            sentences.append(sentence)

    return "; ".join(sentences) if sentences else ""


def extract_evidence_news(news_df, news_data_used, prediction, top_k=2):
    """
    예측에 사용된 뉴스 데이터 날짜 범위 내에서 근거 뉴스 추출

    Args:
        news_df: 전체 뉴스 데이터프레임 (publish_date, title, all_text, price_impact_score, triples 포함)
        news_data_used: 예측에 실제 사용된 뉴스 데이터 (DataFrame)
        prediction: 예측 결과 (0: 하락, 1: 상승)
        top_k: 추출할 뉴스 개수 (기본 2개)

    Returns:
        list: 근거 뉴스 딕셔너리 리스트
            [
                {
                    'title': str,
                    'publish_date': str,
                    'price_impact_score': float,
                    'all_text': str,
                    'triples_text': str
                },
                ...
            ]
    """
    # 날짜 형식 통일
    news_df = news_df.copy()
    news_df["publish_date"] = pd.to_datetime(news_df["publish_date"])

    news_data_used = news_data_used.copy()
    news_data_used["publish_date"] = pd.to_datetime(news_data_used["publish_date"])

    # 예측에 사용된 뉴스의 날짜 범위 확인
    start_date = news_data_used["publish_date"].min()
    end_date = news_data_used["publish_date"].max()

    print(f"  근거 뉴스 검색 범위: {start_date.date()} ~ {end_date.date()}")

    # 해당 날짜 범위의 뉴스만 필터링
    target_news = news_df[
        (news_df["publish_date"] >= start_date) & (news_df["publish_date"] <= end_date)
    ].copy()

    if len(target_news) == 0:
        print(
            f"경고: {start_date.date()} ~ {end_date.date()}에 해당하는 뉴스가 없습니다."
        )
        return []

    print(f"  검색된 뉴스: {len(target_news)} 건")

    # price_impact_score 기준으로 정렬
    if prediction == 1:  # 상승 예측 → 높은 점수 뉴스
        target_news = target_news.sort_values("price_impact_score", ascending=False)
    else:  # 하락 예측 → 낮은 점수 뉴스
        target_news = target_news.sort_values("price_impact_score", ascending=True)

    # 상위 k개 추출
    top_news = target_news.head(top_k)

    # 결과 딕셔너리 생성
    evidence_list = []
    for _, row in top_news.iterrows():
        # triples 파싱 및 텍스트 변환
        triples = parse_triples(row.get("triples", ""))
        triples_text = format_triples_as_text(triples)

        evidence = {
            "title": row["title"] if pd.notna(row["title"]) else "",
            "publish_date": row["publish_date"].strftime("%Y-%m-%d"),
            "price_impact_score": float(row["price_impact_score"])
            if pd.notna(row["price_impact_score"])
            else 0.0,
            "all_text": row["all_text"][:500]
            if pd.notna(row["all_text"])
            else "",  # 첫 500자만
            "triples_text": triples_text,
        }
        evidence_list.append(evidence)

    return evidence_list


def load_news_dataframe(
    news_path="corn_all_news_with_sentiment.csv",
    start_date=None,
    end_date=None,
    window_days=7,
):
    """
    뉴스 데이터 로드 (메모리 효율적으로 필요한 날짜 범위만)

    Args:
        news_path: 뉴스 CSV 파일 경로
        start_date: 시작 날짜 (None이면 전체 로드)
        end_date: 종료 날짜 (None이면 전체 로드)
        window_days: 날짜 범위 전후로 추가로 로드할 일수

    Returns:
        DataFrame: 뉴스 데이터프레임
    """
    # 필요한 컬럼만 로드
    usecols = ["publish_date", "title", "all_text", "price_impact_score", "triples"]

    # 전체 로드
    news_df = pd.read_csv(news_path, usecols=usecols)
    news_df["publish_date"] = pd.to_datetime(news_df["publish_date"])

    # 날짜 범위가 지정되면 해당 기간만 필터링
    if start_date is not None and end_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # window_days만큼 범위 확장
        extended_start = start_date - timedelta(days=window_days)
        extended_end = end_date + timedelta(days=window_days)

        news_df = news_df[
            (news_df["publish_date"].dt.date >= extended_start.date())
            & (news_df["publish_date"].dt.date <= extended_end.date())
        ].copy()

    return news_df


class CornPricePredictor:
    """옥수수 가격 예측 모델 클래스"""

    def __init__(self, model_dir=None):
        """
        모델 초기화

        Args:
            model_dir: 모델 파일들이 저장된 디렉토리 (None이면 자동 탐색)
        """
        # 기본 경로 설정: 현재 파일의 디렉토리
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 우선순위: 1. 인자값 2. 환경변수 3. 로컬 기본값
        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = os.getenv("NEWS_MODEL_PATH", base_dir)

        self.model = None
        self.pca = None
        self.feature_columns = None

    def load_model(self):
        """저장된 모델, PCA 객체, 피처 컬럼 로드"""
        print("=" * 80)
        print("모델 로딩 중...")
        print("=" * 80)

        # 1. XGBoost 모델 로드
        model_path = os.path.join(self.model_dir, "xgb_model.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"✓ 모델 로드 완료: {model_path}")

        # 2. PCA 객체 로드
        pca_path = os.path.join(self.model_dir, "pca_transformer.pkl")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA 파일을 찾을 수 없습니다: {pca_path}")

        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        print(f"✓ PCA 객체 로드 완료: {pca_path}")

        # 3. 피처 컬럼 로드
        feature_path = os.path.join(self.model_dir, "feature_columns.json")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"피처 컬럼 파일을 찾을 수 없습니다: {feature_path}"
            )

        with open(feature_path, "r") as f:
            self.feature_columns = json.load(f)
        print(f"✓ 피처 컬럼 로드 완료: {feature_path}")
        print(f"  - 총 피처 개수: {len(self.feature_columns)}")

        print("\n모든 모델 구성요소가 성공적으로 로드되었습니다!")
        print("=" * 80)

    def predict_next_day(self, news_data, price_history):
        """
        다음 날 가격 상승 여부 예측

        Args:
            news_data: 최근 뉴스 데이터 (DataFrame)
                필수 컬럼: ['publish_date', 'article_embedding', 'price_impact_score',
                           'sentiment_confidence', 'positive_score', 'negative_score']
                최소 3일치 데이터 권장

            price_history: 최근 가격 데이터 (DataFrame)
                필수 컬럼: ['time' 또는 'date', 'close', 'ret_1d']
                최소 5일치 데이터 권장

        Returns:
            dict: 예측 결과
                {
                    "prediction": 0 or 1,  # 0: 하락, 1: 상승
                    "probability": float,   # 상승 확률 (0~1)
                    "features_summary": {
                        "latest_news_count": int,
                        "avg_sentiment": float,
                        "latest_price": float,
                        "data_points_used": int
                    }
                }
        """
        # 모델 로드 확인
        if self.model is None or self.pca is None or self.feature_columns is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요."
            )

        print("\n" + "=" * 80)
        print("예측 데이터 전처리 중...")
        print("=" * 80)

        # 입력 데이터 검증
        self._validate_input_data(news_data, price_history)

        # 전처리 및 피처 생성
        try:
            X = prepare_inference_features(
                news_data=news_data,
                price_history=price_history,
                pca_transformer=self.pca,
                feature_columns=self.feature_columns,
            )
            print(f"✓ 전처리 완료: 피처 shape = {X.shape}")
        except Exception as e:
            raise ValueError(f"데이터 전처리 중 오류 발생: {str(e)}")

        # 예측 수행
        print("\n예측 수행 중...")
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]  # 상승(1) 확률

        # 결과 요약 생성
        features_summary = self._create_features_summary(news_data, price_history)

        # 결과 딕셔너리 생성
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "features_summary": features_summary,
        }

        print("=" * 80)
        print("예측 완료!")
        print("=" * 80)
        print(f"예측 결과: {'상승(1)' if prediction == 1 else '하락(0)'}")
        print(f"상승 확률: {probability:.2%}")
        print("=" * 80)

        return result

    def predict_with_evidence(
        self,
        news_data,
        price_history,
        target_date,
        news_df_full=None,
        news_path="corn_all_news_with_sentiment.csv",
        top_k=2,
    ):
        """
        근거 뉴스를 포함한 예측 수행

        Args:
            news_data: 최근 뉴스 데이터 (DataFrame) - 모델 입력용
            price_history: 최근 가격 데이터 (DataFrame) - 모델 입력용
            target_date: 예측 타겟 날짜 (str 또는 datetime) - 출력용
            news_df_full: 전체 뉴스 데이터프레임 (None이면 파일에서 로드)
            news_path: 뉴스 CSV 파일 경로 (news_df_full이 None일 때 사용)
            top_k: 추출할 근거 뉴스 개수

        Returns:
            dict: 예측 결과 + 근거 뉴스
                {
                    "prediction": 0 or 1,
                    "probability": float,
                    "confidence": float,  # probability와 동일 (호환성)
                    "target_date": str,
                    "features_summary": {...},
                    "evidence_news": [
                        {
                            "title": str,
                            "publish_date": str,
                            "price_impact_score": float,
                            "all_text": str,
                            "triples_text": str
                        },
                        ...
                    ]
                }
        """
        # 1. 기본 예측 수행
        prediction_result = self.predict_next_day(news_data, price_history)

        # 2. 뉴스 데이터 로드 (필요시)
        if news_df_full is None:
            # 예측에 사용된 뉴스의 날짜 범위 확인
            news_data_copy = news_data.copy()
            news_data_copy["publish_date"] = pd.to_datetime(
                news_data_copy["publish_date"]
            )
            start_date = news_data_copy["publish_date"].min()
            end_date = news_data_copy["publish_date"].max()

            print(
                f"\n뉴스 데이터 로드 중... (범위: {start_date.date()} ~ {end_date.date()})"
            )
            news_df_full = load_news_dataframe(
                news_path=news_path,
                start_date=start_date,
                end_date=end_date,
                window_days=1,
            )

        # 3. 근거 뉴스 추출 (예측에 사용된 뉴스 날짜 범위에서)
        print("\n근거 뉴스 추출 중...")
        evidence_news = extract_evidence_news(
            news_df=news_df_full,
            news_data_used=news_data,  # 예측에 사용된 뉴스 데이터 전달
            prediction=prediction_result["prediction"],
            top_k=top_k,
        )

        # 4. 결과 통합
        result = {
            "prediction": prediction_result["prediction"],
            "probability": prediction_result["probability"],
            "confidence": prediction_result["probability"],  # 호환성
            "target_date": pd.to_datetime(target_date).strftime("%Y-%m-%d"),
            "features_summary": prediction_result["features_summary"],
            "evidence_news": evidence_news,
        }

        # 5. 결과 출력
        print("\n" + "=" * 80)
        print("근거 뉴스 추출 완료!")
        print("=" * 80)
        print(f"추출된 뉴스 개수: {len(evidence_news)}")
        if evidence_news:
            print("\n주요 근거 뉴스:")
            for i, news in enumerate(evidence_news, 1):
                print(f"\n[{i}] {news['title']}")
                print(f"    날짜: {news['publish_date']}")
                print(f"    가격 영향 점수: {news['price_impact_score']:.3f}")
                if news["triples_text"]:
                    print(f"    핵심 관계: {news['triples_text'][:100]}...")
        print("=" * 80)

        return result

    def _validate_input_data(self, news_data, price_history):
        """입력 데이터 검증"""
        # 뉴스 데이터 검증
        required_news_cols = [
            "publish_date",
            "article_embedding",
            "price_impact_score",
            "sentiment_confidence",
            "positive_score",
            "negative_score",
        ]
        missing_news_cols = [
            col for col in required_news_cols if col not in news_data.columns
        ]
        if missing_news_cols:
            raise ValueError(
                f"뉴스 데이터에 필수 컬럼이 누락되었습니다: {missing_news_cols}"
            )

        # 가격 데이터 검증
        if "time" not in price_history.columns and "date" not in price_history.columns:
            raise ValueError("가격 데이터에 'time' 또는 'date' 컬럼이 필요합니다.")

        # ret_1d는 preprocessing 단계에서 자동 계산되므로 필수 아님
        required_price_cols = ["close"]
        missing_price_cols = [
            col for col in required_price_cols if col not in price_history.columns
        ]
        if missing_price_cols:
            raise ValueError(
                f"가격 데이터에 필수 컬럼이 누락되었습니다: {missing_price_cols}"
            )

        # 데이터 개수 확인
        if len(news_data) < 3:
            print(
                f"경고: 뉴스 데이터가 {len(news_data)}개로 부족합니다. 최소 3일치 권장"
            )

        if len(price_history) < 5:
            print(
                f"경고: 가격 데이터가 {len(price_history)}개로 부족합니다. 최소 5일치 권장"
            )

    def _create_features_summary(self, news_data, price_history):
        """피처 요약 정보 생성"""
        summary = {
            "latest_news_count": int(len(news_data)),
            "avg_sentiment": float(news_data["sentiment_confidence"].mean()),
            "avg_price_impact": float(news_data["price_impact_score"].mean()),
            "latest_price": float(price_history["close"].iloc[-1]),
            "data_points_used": int(len(price_history)),
        }
        return summary


def predict_next_day(news_data, price_history, model_dir="models"):
    """
    편의 함수: 다음 날 가격 상승 여부 예측

    Args:
        news_data: 최근 뉴스 데이터 (DataFrame)
        price_history: 최근 가격 데이터 (DataFrame)
        model_dir: 모델 파일들이 저장된 디렉토리

    Returns:
        dict: 예측 결과
    """
    predictor = CornPricePredictor(model_dir=model_dir)
    predictor.load_model()
    result = predictor.predict_next_day(news_data, price_history)
    return result


def predict_with_evidence(
    news_data,
    price_history,
    target_date,
    news_df_full=None,
    news_path="corn_all_news_with_sentiment.csv",
    model_dir="models",
    top_k=2,
):
    """
    편의 함수: 근거 뉴스를 포함한 가격 예측

    Args:
        news_data: 최근 뉴스 데이터 (DataFrame) - 모델 입력용
        price_history: 최근 가격 데이터 (DataFrame) - 모델 입력용
        target_date: 예측 타겟 날짜 (str 또는 datetime) - 출력용
        news_df_full: 전체 뉴스 데이터프레임 (None이면 파일에서 로드)
        news_path: 뉴스 CSV 파일 경로
        model_dir: 모델 파일들이 저장된 디렉토리
        top_k: 추출할 근거 뉴스 개수

    Returns:
        dict: 예측 결과 + 근거 뉴스

    Note:
        근거 뉴스는 예측에 실제 사용된 뉴스(news_data)의 날짜 범위 내에서 추출됩니다.
        예: news_data가 11/12~11/14 뉴스라면, 근거 뉴스도 11/12~11/14에서 찾습니다.
    """
    predictor = CornPricePredictor(model_dir=model_dir)
    predictor.load_model()
    result = predictor.predict_with_evidence(
        news_data=news_data,
        price_history=price_history,
        target_date=target_date,
        news_df_full=news_df_full,
        news_path=news_path,
        top_k=top_k,
    )
    return result


# ============================================
# 사용 예시
# ============================================
if __name__ == "__main__":
    """
    추론 코드 사용 예시
    """
    print("\n" + "=" * 80)
    print("추론 코드 사용 예시")
    print("=" * 80)

    # 예시 데이터 생성 (실제로는 최근 데이터를 로드해야 함)
    print("\n※ 실제 사용 시에는 아래처럼 최근 데이터를 로드하세요:")
    print("\n※ 실제 사용 시에는 아래처럼 최근 데이터를 로드하세요:")
    print("""
from inference import predict_with_evidence
import pandas as pd

# 최근 뉴스 데이터 로드
news_data = pd.read_csv('recent_news.csv')
price_history = pd.read_csv('corn_future_price.csv')

# 근거 뉴스를 포함한 예측
result = predict_with_evidence(
    news_data=news_data,
    price_history=price_history,
    target_date='2024-01-27',
    news_path='corn_all_news_with_sentiment.csv',
    top_k=2
)

print(f"예측: {'상승' if result['prediction'] == 1 else '하락'}")
print(f"신뢰도: {result['confidence']:.2%}")
    """)

    print("\n" + "=" * 80)
