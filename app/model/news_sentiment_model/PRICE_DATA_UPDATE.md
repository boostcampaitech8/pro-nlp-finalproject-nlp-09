# 가격 데이터 전처리 업데이트 가이드

## 🔄 주요 변경사항

### 변경 이유
기존에는 `corn_future_price_processed.csv`에 `ret_1d` (일일 수익률)이 미리 계산되어 있다고 가정했습니다.
하지만 실제 원본 파일인 `corn_future_price.csv`에는 `ret_1d`가 없어서, 이를 자동으로 계산하도록 수정했습니다.

### 변경 전 (Before)
```python
# 수동으로 ret_1d를 계산해야 했음
price_df['ret_1d'] = np.log(price_df['close'].shift(-1) / price_df['close'])
```

### 변경 후 (After)
```python
# preprocessing.py에서 자동 계산
from preprocessing import preprocess_price_data

price_df = preprocess_price_data(price_df, time_column='time')
# ret_1d가 자동으로 추가됨!
```

## 📊 수익률 계산 공식

### 일일 수익률 (ret_1d)
```python
ret_1d = log(close_today / close_yesterday)
```

**설명:**
- 전일 종가 대비 당일 종가의 로그 수익률
- 예: 어제 종가 $100, 오늘 종가 $105 → ret_1d = log(105/100) = 0.0488 (약 4.88% 상승)

**왜 로그 수익률을 사용하나?**
1. **대칭성**: 50% 상승과 50% 하락이 비대칭적이지만, 로그는 대칭적
2. **시간 가산성**: 여러 날의 수익률을 단순히 더할 수 있음
3. **정규분포 근사**: 로그 수익률은 정규분포에 가까움 (금융 모델링에 유리)

## 🆕 추가된 함수

### preprocessing.py에 추가
```python
def preprocess_price_data(price_df, time_column='time'):
    """
    가격 데이터 전처리 및 수익률 계산
    
    Args:
        price_df: 원본 가격 데이터프레임
        time_column: 시간 컬럼명 (기본: 'time')
    
    Returns:
        전처리된 가격 데이터프레임 (ret_1d 컬럼 추가됨)
    """
```

**기능:**
1. 날짜 컬럼 정리 (`time` → `date`)
2. 날짜 기준 정렬 (중요!)
3. `ret_1d` 자동 계산 (없는 경우에만)
4. 첫 번째 행의 NaN 처리 (0으로 채움)

## 📁 파일명 변경

### Before
- 입력: `corn_future_price_processed.csv` (ret_1d 포함)

### After  
- 입력: `corn_future_price.csv` (원본 데이터)
- `ret_1d`는 자동 계산됨

## 🔧 사용 방법

### 1. 학습 시 (train.py)
```python
from preprocessing import preprocess_price_data

# 데이터 로드
price_df = pd.read_csv('corn_future_price.csv')

# 전처리 (ret_1d 자동 계산)
price_df = preprocess_price_data(price_df, time_column='time')

# 이제 ret_1d 사용 가능!
print(price_df[['date', 'close', 'ret_1d']].head())
```

### 2. 추론 시 (inference.py)
```python
# prepare_inference_features 내부에서 자동 처리
# 사용자는 신경 쓸 필요 없음!

result = predict_next_day(news_data, price_history)
```

## 📋 입력 데이터 요구사항

### corn_future_price.csv
| 컬럼 | 타입 | 필수 여부 | 설명 |
|------|------|-----------|------|
| `time` | datetime | 필수 | 거래 날짜 및 시간 |
| `close` | float | 필수 | 종가 |
| `open` | float | 선택 | 시가 |
| `high` | float | 선택 | 고가 |
| `low` | float | 선택 | 저가 |
| `volume` | int | 선택 | 거래량 |

**주의:** `ret_1d`는 없어도 됩니다. 자동으로 계산됩니다!

## 🎯 변경된 파일 목록

### 1. preprocessing.py
- ✅ `preprocess_price_data()` 함수 추가
- ✅ `prepare_inference_features()`에서 자동 호출

### 2. train.py
- ✅ `preprocess_price_data` import 추가
- ✅ 가격 데이터 전처리 부분 수정
- ✅ 기본 파일명 변경: `corn_future_price.csv`

### 3. inference.py
- ✅ 예시 코드의 파일명 변경

### 4. examples_inference.py
- ✅ 모든 예시의 파일명 변경

### 5. README.md
- ✅ 데이터 요구사항 업데이트
- ✅ `ret_1d` 자동 계산 설명 추가

## ⚠️ 마이그레이션 가이드

### 기존 코드를 사용하는 경우

#### Before (수동 계산)
```python
import pandas as pd
import numpy as np

price_df = pd.read_csv('corn_future_price.csv')
price_df['time'] = pd.to_datetime(price_df['time'])
price_df['date'] = price_df['time'].dt.date
price_df = price_df.sort_values('date')

# 수동으로 ret_1d 계산
price_df['ret_1d'] = np.log(price_df['close'] / price_df['close'].shift(1))
price_df['ret_1d'] = price_df['ret_1d'].fillna(0)
```

#### After (자동 계산)
```python
from preprocessing import preprocess_price_data

price_df = pd.read_csv('corn_future_price.csv')
price_df = preprocess_price_data(price_df)  # 한 줄로 끝!
```

### 데이터 파일 변경
```bash
# 기존 파일명
corn_future_price_processed.csv

# 새 파일명 (원본 그대로 사용)
corn_future_price.csv
```

**중요:** `corn_future_price.csv`에 `ret_1d` 컬럼이 있어도 상관없습니다.
있으면 그대로 사용하고, 없으면 자동으로 계산합니다.

## 🧪 테스트 방법

### 1. 수익률 계산 확인
```python
from preprocessing import preprocess_price_data
import pandas as pd

# 테스트 데이터
df = pd.DataFrame({
    'time': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'close': [100, 105, 103]
})

# 전처리
df = preprocess_price_data(df)

# 확인
print(df[['date', 'close', 'ret_1d']])

# 예상 결과:
#         date  close    ret_1d
# 0 2024-01-01    100  0.000000  (첫날은 0)
# 1 2024-01-02    105  0.048790  (log(105/100))
# 2 2024-01-03    103 -0.019418  (log(103/105))
```

### 2. 학습 파이프라인 테스트
```bash
# 1. 감성 분석
python run_sentiment_analysis.py

# 2. 학습 (ret_1d 자동 계산됨)
python train.py

# 3. 추론
python -c "
from inference import predict_next_day
import pandas as pd

news = pd.read_csv('corn_all_news_with_sentiment.csv').tail(100)
price = pd.read_csv('corn_future_price.csv').tail(10)

result = predict_next_day(news, price)
print(result)
"
```

## 💡 FAQ

### Q1: 기존 `corn_future_price_processed.csv` 파일은 어떻게 하나요?
**A:** 사용하지 않아도 됩니다. 원본 `corn_future_price.csv`만 있으면 됩니다.

### Q2: ret_1d가 이미 있는 데이터를 사용하면?
**A:** 문제없습니다. `preprocess_price_data()`는 ret_1d가 이미 있으면 재계산하지 않습니다.

### Q3: 수익률 계산식이 다르면?
**A:** `preprocess_price_data()` 함수를 수정하여 원하는 계산식을 사용할 수 있습니다.

### Q4: 추론 시 ret_1d가 없는 데이터를 입력하면?
**A:** `prepare_inference_features()` 내부에서 자동으로 계산되므로 문제없습니다.

### Q5: 날짜 순서가 뒤죽박죽이면?
**A:** `preprocess_price_data()`가 자동으로 날짜 기준 정렬을 수행합니다.

## ✅ 체크리스트

마이그레이션 전 확인사항:
- [ ] `corn_future_price.csv` 파일이 있는가?
- [ ] `close` 컬럼이 있는가?
- [ ] `time` 또는 `date` 컬럼이 있는가?

코드 업데이트:
- [ ] `preprocessing.py` 업데이트 완료
- [ ] `train.py` 업데이트 완료
- [ ] `inference.py` 업데이트 완료
- [ ] 기존 코드에서 수동 ret_1d 계산 제거

테스트:
- [ ] 수익률 계산 확인
- [ ] 학습 파이프라인 정상 작동
- [ ] 추론 파이프라인 정상 작동
