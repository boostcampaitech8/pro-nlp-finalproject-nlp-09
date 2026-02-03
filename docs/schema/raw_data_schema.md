# 데이터 스키마 및 상세 사양 문서 (Data Specification)

## 시장 가격 관련 데이터 (3개 파일)

* `3개의 파일`은 `시장 가격과 관련된 데이터`임
  해당 데이터는 각각의 작물에 대한 선물 가격이며, 휴장일에 대해서는 데이터가 존재하지 않음. (파일 첨부함)

아래 데이터들에 대해서는 모두 동일한 컬럼을 가지고 있음.

| 컬럼명    | 컬럼에 대한 설명       |
| ------ | --------------- |
| time   | 거래 날짜           |
| open   | 해당 날짜의 최초 거래 가격 |
| high   | 해당 날짜의 거래 최고 가격 |
| low    | 해달 날짜의 거래 최저 가격 |
| close  | 해당 날짜의 종가       |
| EMA    | 지수이동평균          |
| Volume | 해당 날짜의 거래량      |

* corn_future_price.csv - 2010.09.22 ~ 2025.11.26까지 하루 주기로, 수집된 총 3,824개의 corn 선물 가격 데이터
* soybean_future_price.csv - 2015.04.22 ~ 2025.11.26까지 하루 주기로, 수집된 총 2,670개의 soybean 선물 가격 데이터
* wheat_future_price.csv - 2015.01.13 ~ 2025.11.26까지 하루 주기로, 수집된 총 2,738개의 wheat 선물 가격 데이터

---

## 뉴스 기사 관련 데이터 (3개 파일)

* `3개의 파일`은 `뉴스와 관련된 데이터`임. (용량이 커서 첨부 못함)

### news_articles_resources.csv

* 2017.11.07 ~ 2025.11.14까지 수집된, 총 296,514개의 뉴스 기사

| 컬럼명               | 컬럼에 대한 설명                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| id                | Primary Key                                                                                                               |
| title             | 기사 제목                                                                                                                     |
| doc_url           | 해당 기사의 URL                                                                                                                |
| all_text          | 특정 길이로 자른 article content                                                                                                 |
| authors           | 기사 저자                                                                                                                     |
| publish_date      | 기사 발행일                                                                                                                    |
| meta_site_name    | 해당 기사를 보도한 매체                                                                                                             |
| description       | 기사 description                                                                                                            |
| key_word          | 기사 검색 키워드                                                                                                                 |
| filter_status     | 해당 기사가 선물 시장과 관련된 기사가 맞는지 판단한 결과. T는 관련된 기사, F는 관련되어 있지 않다고 판단한 기사.  `title 이나 description이 없어 처리 하지 못한 항목에 대해서는 E로 기록됨.` |
| named_entities    | 해당 기사의 엔티티를 추출한 리스트. `filter_status == T`인 경우에만 존재                                                                        |
| triples           | 해당 기사의 triple을 추출한 리스트. `filter_status == T`인 경우에만 존재                                                                     |
| article_embedding | 해당 기사를 embedding한 vector(512차원). `filter_status == T`인 경우에만 존재                                                            |

### 샘플 예시

| id    | title                                                                              | filter_status | key_word                                           | named_entity                                                     | triples                                                                                                                                                                                                                                                                                                    |
| ----- | ---------------------------------------------------------------------------------- | ------------- | -------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 45939 | India’s Food Ministry favours lifting ban on wheat products export, moots 1 mt cap | T             | wheat and (price or demand or supply or inventory) | ["India", "Food Ministry", "wheat products", "1 million tonnes"] | [["Food Ministry", "located in", "India"], ["Food Ministry", "supports", "lifting ban on wheat products export"], ["Food Ministry", "proposes", "1 million tonnes cap"], ["1 million tonnes cap", "applies to", "wheat products"], ["lifting ban on wheat products export", "intended to", "boost trade"]] |

1. 해당 sample은 예측에 유효하다고 판단되어 filter_status를 T로 매핑
2. 해당 기사 수집은 key_word열에 있는 wheat and (price or demand or supply or inventory)로 수집되었음
3. 해당 sample의 filter_status는 T로 매핑되었기에, named_entity, triples, article_embedding을 추출. F 또는 E라면 해당 처리를 하지 않음
4. named_entity는 해당 기사의 명사구를 추출한 것
5. triples는 주어, 술어, 목적어 순으로 이뤄져 있으며, 주어와 목적어 간의 관계를 표시함
6. article_embedding은 title과 description을 추출하여  `f"{article.title}\n\n{article.description}"` 형태로 string을 구성하고, amazone의 titan-embed-text-v2를 사용하여 (512)차원으로 임베딩

---

### news_articles_resources_entities.csv

* `news_articles_resources.csv`에서 추출한 entity와 entity의 임베딩 정보
* 총 104,573개의 entity가 존재함

| 컬럼명         | 컬럼에 대한 설명                      |
| ----------- | ------------------------------ |
| hash_id     | 명사구를 해시로 매핑한 값. 구체적인 설명은 하단 참고 |
| entity_text | 엔티티                            |
| embedding   | 엔티티의 임베딩 정보 (1024차원)           |

---

### news_articles_resources_triples.csv

* `news_articles_resources.csv`에서 추출한 tripe과 triple의 임베딩 정보
* 총 714,664개의 triple이 존재함

| 컬럼명         | 컬럼에 대한 설명                         |
| ----------- | --------------------------------- |
| hash_id     | triple을 해시로 매핑한 값. 구체적인 설명은 하단 참고 |
| triple_text | triple                            |
| embedding   | triple의 임베딩 정보 (1024차원)           |

---

## hash_id 생성 코드

```python
from hashlib import md5

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

## 엔티티 해쉬
entity = entity.strip()
compute_mdhash_id(entity, prefix="entity-")

## 트리플 해쉬
triple_text = str(triple).strip()
triple_hash = compute_mdhash_id(triple_text, prefix="triple-")
```

---

## 뉴스 기사 시각화 데이터 (1개 파일)

* `1개의 파일`은 해당 기사 데이터를 `클러스터링하여 시각화`한 파일임.

### resource_article_clustering.html

* 2024.01.01 ~ 2024.08.01에 수집된 기사에 대해서 article_embedding을 통해 클러스터링을 진행하고, 이를 시각화하여 HTML 파일로 저장한 시각화 샘플 파일
* 해당 파일을 브라우저를 통해 열람시, 값 확인이 가능

---