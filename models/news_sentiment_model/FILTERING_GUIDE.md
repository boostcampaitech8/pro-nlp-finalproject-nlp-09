# 뉴스 데이터 필터링 가이드

## 📌 기본 설정

### 입력 파일
- **파일명**: `news_articles_resources.csv`
- **설명**: 전체 원자재 뉴스 데이터

### 기본 필터링 조건
```python
filter_status == 'T'
key_word == 'corn and (price or demand or supply or inventory)'
```

### 출력 파일
- **파일명**: `corn_all_news_with_sentiment.csv`
- **설명**: 옥수수 관련 뉴스 + 감성 분석 결과

## 🚀 사용 방법

### 1. 기본 실행 (권장)
```bash
python run_sentiment_analysis.py
```
자동으로 다음 조건을 적용합니다:
- ✅ `filter_status == 'T'`
- ✅ `key_word == 'corn and (price or demand or supply or inventory)'`

### 2. 커스텀 키워드
```bash
# 가격 관련만
python run_sentiment_analysis.py --keyword "corn and price"

# 공급 관련만
python run_sentiment_analysis.py --keyword "corn and supply"

# 수요 관련만
python run_sentiment_analysis.py --keyword "corn and demand"
```

### 3. filter_status 변경
```bash
# 'F' 상태만 분석
python run_sentiment_analysis.py --filter-status "F"

# 모든 filter_status 포함
python run_sentiment_analysis.py --filter-status "all"
```

### 4. 필터링 없이 전체 분석
```bash
python run_sentiment_analysis.py --no-filter
```

## 📊 데이터 구조

### 입력 데이터 (news_articles_resources.csv)
| 컬럼 | 설명 | 예시 |
|------|------|------|
| `title` | 뉴스 제목 | "Corn prices surge..." |
| `description` | 뉴스 설명 | "Strong demand..." |
| `all_text` | 전체 내용 | "..." |
| `publish_date` | 발행일 | "2024-01-27" |
| `filter_status` | 필터 상태 | "T" 또는 "F" |
| `key_word` | 키워드 | "corn and (price or demand...)" |

### 출력 데이터 (corn_all_news_with_sentiment.csv)
입력 데이터의 모든 컬럼 + 다음 컬럼들이 추가됩니다:

| 컬럼 | 설명 | 범위 |
|------|------|------|
| `sentiment` | 감성 | positive/negative/neutral |
| `sentiment_confidence` | 감성 신뢰도 | 0~1 |
| `positive_score` | 긍정 점수 | 0~1 |
| `negative_score` | 부정 점수 | 0~1 |
| `neutral_score` | 중립 점수 | 0~1 |
| `price_impact_score` | 가격 영향 점수 | -1~1 |
| `article_embedding` | 임베딩 벡터 | 512차원 |

## 🔍 필터링 로직

### filter_status란?
뉴스가 옥수수 가격과 실제로 관련이 있는지 판단한 플래그입니다.
- **'T' (True)**: 옥수수 가격과 관련이 높음 → 학습에 사용
- **'F' (False)**: 관련성이 낮음 → 학습에서 제외

### key_word란?
뉴스를 수집할 때 사용한 검색 키워드입니다.

#### 기본 키워드 분석
```
corn and (price or demand or supply or inventory)
```
이 키워드는 다음 조건을 만족하는 뉴스를 찾습니다:
- ✅ "corn"이 포함되어야 함 (필수)
- ✅ "price", "demand", "supply", "inventory" 중 최소 1개 포함

#### 키워드 조합 예시
| 키워드 | 설명 | 예상 기사 수 |
|--------|------|--------------|
| `corn and price` | 가격 관련만 | 가장 많음 |
| `corn and demand` | 수요 관련만 | 중간 |
| `corn and supply` | 공급 관련만 | 중간 |
| `corn and inventory` | 재고 관련만 | 가장 적음 |
| `corn and (price or demand)` | 가격 + 수요 | 많음 |

## 📈 필터링 효과

### Before (전체 데이터)
```
news_articles_resources.csv: 10,000 기사
```

### After (필터링 적용)
```
filter_status == 'T': 3,000 기사
+ key_word 조건: 1,500 기사
→ corn_all_news_with_sentiment.csv: 1,500 기사
```

## 💡 추천 사용 시나리오

### 시나리오 1: 최초 학습 (권장)
```bash
# 기본 설정으로 고품질 데이터만 사용
python run_sentiment_analysis.py
python train.py
```

### 시나리오 2: 데이터 부족 시
```bash
# filter_status 무시, 키워드만 적용
python run_sentiment_analysis.py --filter-status "all"
python train.py
```

### 시나리오 3: 특정 주제 분석
```bash
# 가격 관련만 집중 분석
python run_sentiment_analysis.py --keyword "corn and price"
python train.py
```

### 시나리오 4: 전체 데이터 탐색
```bash
# 모든 필터 제거
python run_sentiment_analysis.py --no-filter
# 결과 확인 후 적절한 필터 선택
```

## ⚠️ 주의사항

### 1. 데이터 품질 vs 데이터 양
- **필터링 多**: 고품질, 적은 양 → 정확도 ↑, 일반화 ↓
- **필터링 少**: 저품질, 많은 양 → 정확도 ↓, 일반화 ↑

### 2. 권장 사항
- 최초 학습: 기본 필터링 적용 (고품질 데이터)
- 성능 부족 시: 필터 완화 (데이터 양 증가)
- 과적합 발생 시: 필터 강화 (데이터 품질 향상)

### 3. key_word 컬럼이 없는 경우
```bash
# 자동으로 키워드 필터링을 건너뜁니다
python run_sentiment_analysis.py
# ⚠️ 'key_word' 컬럼이 없어 키워드 필터링을 건너뜁니다.
```

## 🔧 고급 사용법

### Python 코드로 직접 필터링
```python
import pandas as pd
from finbert import analyze_news_sentiment, prepare_text_for_analysis

# 데이터 로드
df = pd.read_csv('news_articles_resources.csv')

# 커스텀 필터링
df_filtered = df[
    (df['filter_status'] == 'T') &
    (df['key_word'].str.contains('corn', case=False)) &
    (df['key_word'].str.contains('price|demand', case=False))
].copy()

# 추가 조건: 최근 6개월만
df_filtered['publish_date'] = pd.to_datetime(df_filtered['publish_date'])
recent_date = df_filtered['publish_date'].max() - pd.Timedelta(days=180)
df_filtered = df_filtered[df_filtered['publish_date'] >= recent_date]

# 감성 분석
df_filtered = prepare_text_for_analysis(df_filtered)
df_result = analyze_news_sentiment(df_filtered)

# 저장
df_result.to_csv('corn_all_news_with_sentiment.csv', index=False)
```

## 📝 체크리스트

학습 전 확인사항:
- [ ] `news_articles_resources.csv` 파일이 존재하는가?
- [ ] `filter_status` 컬럼이 있는가?
- [ ] `key_word` 컬럼이 있는가?
- [ ] 필터링 후 최소 100개 이상의 뉴스가 있는가?
- [ ] GPU 환경에서 실행하는가? (권장)

필터링 조건 결정:
- [ ] 데이터 품질이 중요한가? → 기본 필터링 사용
- [ ] 데이터 양이 중요한가? → `--filter-status "all"` 사용
- [ ] 특정 주제에 집중하는가? → `--keyword` 옵션 사용
- [ ] 탐색 단계인가? → `--no-filter` 사용 후 결과 확인
