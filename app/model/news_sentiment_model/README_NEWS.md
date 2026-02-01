# 옥수수 선물 가격 예측 모델

옥수수 선물 가격의 다음날 상승/하락을 예측하는 머신러닝 모델입니다.
뉴스 감성 분석과 가격 시계열 데이터를 결합하여 예측을 수행합니다.

## 📁 프로젝트 구조

```
.
├── finbert.py                        # 감성 분석 모듈 (순수 함수/클래스)
├── run_sentiment_analysis.py         # 감성 분석 실행 스크립트
├── preprocessing.py                  # 공통 전처리 함수 모듈
├── train.py                          # 학습 코드
├── inference.py                      # 추론 코드
├── test_inference.py                 # inference 테스트 실행
│
├── models/                           # 학습된 모델 저장 디렉토리
│   ├── xgb_model.json                       # XGBoost 모델
│   ├── pca_transformer.pkl                  # PCA 객체
│   └── feature_columns.json                 # 피처 컬럼 리스트
│
├── README.md                         # 문서 (이 파일)
├── PIPELINE.md                       # 전체 파이프라인 상세 설명
├── FILTERING_GUIDE.md                # 뉴스 필터링 가이드
├── PRICE_DATA_UPDATE.md              # 가격 데이터 전처리 가이드
└── TEST_GUIDE.md                     # Inference 테스트 가이드
```

## 🔄 전체 파이프라인

```
Step 0: 감성 분석 (최초 1회)
   run_sentiment_analysis.py
   news_articles_resources.csv → corn_all_news_with_sentiment.csv
   
Step 1: 모델 학습
   train.py
   corn_all_news_with_sentiment.csv + corn_future_price.csv → models/
   
Step 2: 예측 수행
   inference.py
   최근 뉴스 + 최근 가격 → 가격 예측
```

**빠른 시작:**
```bash
# 1. 감성 분석
python run_sentiment_analysis.py

# 2. 모델 학습
python train.py

# 3. 예측
python inference.py
```

## 🚀 사용 방법

### 0. 감성 분석 (최초 1회 또는 새 뉴스 수집 시)

#### 방법 1: 스크립트 실행 (권장)
```bash
# 기본 사용 (필터링 자동 적용)
# - 입력: news_articles_resources.csv
# - 출력: corn_all_news_with_sentiment.csv
# - 필터: filter_status='T', keyword='corn and (price or demand or supply or inventory)'
python run_sentiment_analysis.py

# 커스텀 경로 지정
python run_sentiment_analysis.py --input my_news.csv --output result.csv

# 다른 키워드로 필터링
python run_sentiment_analysis.py --keyword "corn and price"

# filter_status 변경
python run_sentiment_analysis.py --filter-status "F"

# 필터링 없이 전체 분석
python run_sentiment_analysis.py --no-filter

# 진행상황 숨기기
python run_sentiment_analysis.py --no-progress
```

#### 방법 2: 모듈로 직접 사용
```python
from finbert import analyze_news_sentiment, prepare_text_for_analysis
import pandas as pd

# 데이터 로드
df = pd.read_csv('news_articles_corn.csv')

# 텍스트 준비
df = prepare_text_for_analysis(df)

# 감성 분석
df_result = analyze_news_sentiment(df, text_column='combined_text')

# 결과 저장
df_result.to_csv('corn_all_news_with_sentiment.csv', index=False)
```

**필요한 파일:**
- `news_articles_resources.csv`: 원본 뉴스 데이터
  - 필수 컬럼: `title`, `description`, `publish_date`, `filter_status`, `key_word`

**기본 필터링 조건:**
- `filter_status == 'T'`
- `key_word == 'corn and (price or demand or supply or inventory)'`

**출력:**
- `corn_all_news_with_sentiment.csv`: 감성 분석이 추가된 뉴스 데이터
  - `sentiment`: positive/negative/neutral
  - `sentiment_confidence`: 감성 분석 신뢰도
  - `positive_score`, `negative_score`, `neutral_score`: 각 감성 점수
  - `price_impact_score`: positive_score - negative_score
  - `article_embedding`: 512차원 임베딩 벡터

⚠️ **주의**: 이 단계는 GPU 환경에서 실행하는 것을 권장합니다 (transformers 모델 사용)

### 1. 모델 학습

```bash
python train.py
```

**필요한 파일:**
- `corn_all_news_with_sentiment.csv`: 뉴스 감성 분석 데이터 (Step 0에서 생성)
- `corn_future_price.csv`: 옥수수 선물 가격 데이터
  - 필수 컬럼: `time` (또는 `date`), `close`
  - `ret_1d` (일일 수익률)은 자동 계산됨

**출력:**
- `models/xgb_model.json`: 학습된 XGBoost 모델
- `models/pca_transformer.pkl`: 학습된 PCA 객체
- `models/feature_columns.json`: 피처 컬럼 리스트

### 2. 추론 (예측)

```python
import pandas as pd
from inference import predict_next_day

# 최근 뉴스 데이터 로드 (최소 3일치 권장)
news_data = pd.read_csv('recent_news.csv')

# 최근 가격 데이터 로드 (최소 5일치 권장)
price_history = pd.read_csv('recent_prices.csv')

# 예측 수행
result = predict_next_day(news_data, price_history, model_dir='models')

# 결과 확인
print(f"예측: {result['prediction']}")  # 0: 하락, 1: 상승
print(f"상승 확률: {result['probability']:.2%}")
print(f"피처 요약: {result['features_summary']}")
```

**출력 형식:**
```json
{
    "prediction": 1,
    "probability": 0.85,
    "features_summary": {
        "latest_news_count": 15,
        "avg_sentiment": 0.72,
        "avg_price_impact": 0.65,
        "latest_price": 425.50,
        "data_points_used": 10
    }
}
```

### 3. 테스트 (선택사항)

모델 성능을 검증하려면 `test_inference.py`를 사용하세요.

```bash
python test_inference.py
```

자세한 내용은 [TEST_GUIDE.md](TEST_GUIDE.md) 참조

## 📊 데이터 요구사항

### 뉴스 데이터 (news_data)
최소 3일치 데이터 권장

| 컬럼 | 타입 | 설명 |
|------|------|------|
| publish_date | datetime | 뉴스 발행일 |
| article_embedding | str/list | 512차원 임베딩 벡터 |
| price_impact_score | float | 가격 영향 점수 (0~1) |
| sentiment_confidence | float | 감성 신뢰도 (0~1) |
| positive_score | float | 긍정 점수 (0~1) |
| negative_score | float | 부정 점수 (0~1) |

### 가격 데이터 (price_history)
최소 5일치 데이터 권장

| 컬럼 | 타입 | 설명 |
|------|------|------|
| time 또는 date | datetime | 거래 날짜 |
| close | float | 종가 |

**참고**: `ret_1d` (일일 수익률)은 자동으로 계산됩니다.
- 계산식: `ret_1d = log(close_today / close_yesterday)`
- 즉, 전일 종가 대비 당일 종가의 로그 수익률

## 🔧 핵심 기술

### 1. 전처리 파이프라인
- **날짜 보정**: 주말/휴일 뉴스를 다음 거래일에 반영
- **임베딩 차원 축소**: PCA를 통해 512 → 50차원 축소
- **시계열 피처**: Lag(T-1, T-2) 및 이동평균(MA3, MA5)

### 2. 모델 아키텍처
- **알고리즘**: XGBoost (Gradient Boosting)
- **피처**: 감성 지표 + Lag 피처 + PCA 임베딩 (총 ~130개)
- **타겟**: 다음날 0.5% 이상 상승 여부 (0/1)

### 3. 성능 지표
- Accuracy: 정확도
- Precision: 상승 예측 정확도
- Recall: 상승 탐지율
- F1-Score: Precision과 Recall의 조화평균

## 🔗 LangChain 연동 예시

```python
from langchain.tools import Tool
from inference import predict_next_day

def corn_price_prediction_tool(input_data):
    """옥수수 가격 예측 도구"""
    # 최근 데이터 로드
    news_data = load_recent_news()
    price_history = load_recent_prices()
    
    # 예측 수행
    result = predict_next_day(news_data, price_history)
    
    # LLM에게 전달할 보고서 생성
    report = f"""
    옥수수 선물 가격 예측 결과:
    - 예측: {'상승' if result['prediction'] == 1 else '하락'}
    - 상승 확률: {result['probability']:.2%}
    - 분석 데이터:
      * 뉴스 기사 수: {result['features_summary']['latest_news_count']}개
      * 평균 감성 점수: {result['features_summary']['avg_sentiment']:.2f}
      * 최근 가격: ${result['features_summary']['latest_price']:.2f}
    """
    return report

# LangChain Tool 등록
prediction_tool = Tool(
    name="CornPricePrediction",
    func=corn_price_prediction_tool,
    description="옥수수 선물 가격의 다음날 상승/하락을 예측합니다."
)

# LLM 에이전트에 도구 추가
agent = initialize_agent(
    tools=[prediction_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# 보고서 생성 요청
response = agent.run("옥수수 가격 전망 보고서를 작성해주세요.")
```

## ⚠️ 주의사항

### 1. 추론 시 PCA 사용
- **학습 시**: `pca.fit_transform()` - PCA를 학습하고 적용
- **추론 시**: `pca.transform()` - 학습된 PCA로 변환만 수행
- ❌ 추론 시 `fit_transform` 사용하면 안 됨!

### 2. 데이터 일관성
- 학습과 추론에서 동일한 전처리 파이프라인 사용
- 피처 컬럼 순서가 동일해야 함 (feature_columns.json 참조)

### 3. 데이터 양
- 뉴스: 최소 3일치 (Lag 피처 생성을 위해)
- 가격: 최소 5일치 (이동평균 계산을 위해)

### 4. 가격 데이터 (중요!)
- `corn_future_price.csv`에는 `close` 컬럼만 필수
- `ret_1d` (일일 수익률)은 **자동 계산**됨
- 계산식: `ret_1d = log(close_today / close_yesterday)`
- 자세한 내용: [PRICE_DATA_UPDATE.md](PRICE_DATA_UPDATE.md)

### 5. 모델 재학습
- 새로운 데이터로 재학습 시 모든 아티팩트(모델, PCA, 피처) 재생성 필요

## 📚 추가 문서

- **[PIPELINE.md](PIPELINE.md)** - 전체 파이프라인 상세 설명
- **[FILTERING_GUIDE.md](FILTERING_GUIDE.md)** - 뉴스 필터링 가이드
- **[PRICE_DATA_UPDATE.md](PRICE_DATA_UPDATE.md)** - 가격 데이터 전처리 가이드
- **[TEST_GUIDE.md](TEST_GUIDE.md)** - Inference 테스트 가이드

## 📈 성능 개선 팁

1. **데이터 품질**: 고품질 뉴스 필터링 (filter_status='T')
2. **하이퍼파라미터 튜닝**: XGBoost의 n_estimators, max_depth 조정
3. **피처 엔지니어링**: 추가 Lag, 다양한 이동평균 윈도우
4. **앙상블**: XGBoost + LightGBM 결합

## 📝 라이센스

이 프로젝트는 교육 및 연구 목적으로만 사용됩니다.
실제 투자 결정에 사용하지 마세요.

## 🙋 문의

질문이나 문제가 있으시면 이슈를 등록해주세요.
