"""
키워드 분석 모듈
- 뉴스 기사의 triples(Subject-Verb-Object)를 분석하여 주요 키워드 추출
- PageRank 알고리즘을 활용한 엔티티 중요도 계산
- 임베딩 기반 클러스터링으로 유사 엔티티 통합
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union

import networkx as nx
import numpy as np

# NLTK 관련 import
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet

# NLTK 데이터 다운로드 (최초 1회만 필요)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# sklearn 임베딩 유사도
from sklearn.metrics.pairwise import cosine_similarity

# Sentence Transformers 임베딩 (선택적)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    EMBEDDING_AVAILABLE = False
    _embedding_model = None

from app.utils.bigquery_client import BigQueryClient


# ---------------------------
# 설정 상수
# ---------------------------
CLUSTERING_SIMILARITY_THRESHOLD = 0.65  # 클러스터링 유사도 임계값

# Verb 설정: 시장 이동성(stance) vs 토픽 관련성(action)
BE_VERB_WEIGHT = 0.05  # meaningless be(단독, non-passive) 엣지
DEFAULT_EDGE_WEIGHT = 1.0  # 초기 엣지 weight (non-be)
WEIGHT_MIN = 0.1   # contribution 정규화 후 clip 하한
WEIGHT_MAX = 1.2   # contribution 정규화 후 clip 상한
STANCE_MIN_WEIGHT = 1   # stance 임베딩 유사도 통과 시 최소 weight
STANCE_KEYWORDS = {
    "doubt", "question", "downgrade", "cut", "default", "warn",
    "agreement", "deal", "contract", "commitment", "promise",
    "ban", "sanction", "embargo", "restriction",
    "regulate", "implement", "enforce",
    "profit", "loss", "gain", "revenue",
    "invest", "divest", "acquire", "merge", "settle",
    "risk", "alert", "caution", "concern", "fail", "breach", "litigate", "dispute"
}
STANCE_SIMILARITY_THRESHOLD = 0.65

PRICE_DEMAND_WEIGHT = 0.8
PRICE_DEMAND_VERBS = {
    "prices rise", "prices fall", "value increased", "value decreased",
    "market arrivals", "oversupply", "shortage", "has reduced", "imported by",
}
PRICE_DEMAND_SIMILARITY_THRESHOLD = 0.65

# 후처리에서 제외할 노이즈 엔티티
NOISE_ENTITIES = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "data", "week", "month", "year", "midday",
    "last year", "month off", "time period",
    "cents", "percent", "digit losses", "name brands", "private labels",
    "midday", "daily", "quarter", "half", "name", "brands"
}

BE_FORMS = {"is", "are", "was", "were", "be", "been", "being"}
HAVE_FORMS = {"has", "have", "had"}  # 의미 약한 have 동사 → 낮은 weight
MEANINGLESS_VERB_FORMS = BE_FORMS | HAVE_FORMS

# Lemmatizer 인스턴스
_lemmatizer = WordNetLemmatizer()


def _get_wordnet_pos(tag: str):
    """NLTK POS 태그를 WordNet POS 태그로 변환"""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def _is_meaningless_be(verb: str) -> bool:
    """
    verb가 '의미 없는' 단독 be/have 동사인지 판별.
    - 단일 be/has 형태 → True (의미 없음, 낮은 weight)
    - be + VBN(ed형) 수동태 → False (의미 있음)
    """
    if not verb or not isinstance(verb, str):
        return False
    verb_tokens = verb.lower().strip().split()
    if len(verb_tokens) == 1 and verb_tokens[0] in MEANINGLESS_VERB_FORMS:
        return True
    if len(verb_tokens) >= 2 and verb_tokens[0] in BE_FORMS and verb_tokens[1].endswith("ed"):
        return False
    # "has CAGR of", "have been" 등: 첫 토큰이 has/have/had면 의미 약함
    if len(verb_tokens) >= 1 and verb_tokens[0] in HAVE_FORMS:
        return True
    return False


def _is_valid_entity(entity: str) -> bool:
    """유효한 엔티티인지 확인"""
    if not entity or not isinstance(entity, str):
        return False
    entity = entity.strip()
    if len(entity) < 2:
        return False
    if entity.replace('.', '').replace('-', '').isdigit():
        return False
    if not any(c.isalnum() for c in entity):
        return False
    return True


def _normalize_entity(entity: str, aggressive: bool = True) -> str:
    """
    NLTK를 활용한 엔티티 정규화
    - 2~4글자 대문자 약어는 Lemmatization 건너뜀
    """
    if not entity or not isinstance(entity, str):
        return entity
    
    original = entity.strip()
    
    # 2~4글자 대문자 약어: Lemmatization 건너뜀
    if original.isupper() and 2 <= len(original) <= 4:
        return original
    
    normalized = original.lower()
    
    if not aggressive:
        return normalized
    
    try:
        tokens = word_tokenize(normalized)
        if not tokens:
            return normalized
        
        tagged = pos_tag(tokens)
        
        if len(tokens) > 1 or "'" in normalized:
            lemmatized_tokens = []
            for word, tag in tagged:
                pos_wn = wordnet.NOUN if tag.startswith('NN') else _get_wordnet_pos(tag)
                lemmatized_tokens.append(_lemmatizer.lemmatize(word, pos_wn))
            return ' '.join(lemmatized_tokens)
        
        if len(tagged) > 0:
            word, tag = tagged[0]
            pos_wn = wordnet.NOUN if tag.startswith('NN') else _get_wordnet_pos(tag)
            return _lemmatizer.lemmatize(word, pos_wn)
        
    except Exception:
        pass
    
    return normalized


def _cluster_similar_words(
    words: List[str],
    similarity_threshold: float = CLUSTERING_SIMILARITY_THRESHOLD,
    embeddings: np.ndarray = None
) -> Dict[str, str]:
    """
    임베딩 기반으로 유사한 단어들을 클러스터링하고, 엔티티 → 대표어 매핑을 반환.
    """
    entity_to_representative = {w: w for w in words} if words else {}
    if not EMBEDDING_AVAILABLE or not words:
        return entity_to_representative
    if len(words) <= 1:
        return entity_to_representative
    
    try:
        if embeddings is None:
            embeddings = _embedding_model.encode(words)
        else:
            embeddings = np.asarray(embeddings)
        
        similarity_matrix = cosine_similarity(embeddings)
        
        used = set()
        
        for i, word in enumerate(words):
            if i in used:
                continue
            
            cluster = [word]
            used.add(i)
            
            for j in range(i + 1, len(words)):
                if j in used:
                    continue
                if similarity_matrix[i][j] >= similarity_threshold:
                    cluster.append(words[j])
                    used.add(j)
            
            representative = min(cluster, key=len)
            for w in cluster:
                entity_to_representative[w] = representative
        
        return entity_to_representative
    except Exception as e:
        print(f"⚠️ 클러스터링 오류: {e}")
        return entity_to_representative


def _contains_number(entity: str) -> bool:
    """엔티티에 숫자가 포함되어 있는지 확인"""
    return bool(re.search(r'\d', entity))


class KeywordAnalyzer:
    """뉴스 기사의 triples를 분석하여 주요 키워드를 추출하는 분석기"""
    
    def __init__(self):
        self.client = BigQueryClient()
    
    def analyze_keywords(
        self,
        target_date: str,
        days: int = 3,
        keyword_filter: str = "corn and (price or demand or supply or inventory) OR united states department of agriculture",
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        지정된 날짜 기준으로 뉴스 기사의 키워드를 분석합니다.
        
        Args:
            target_date: 분석 기준 날짜 (형식: "YYYY-MM-DD")
            days: 분석할 일수 (기본 3일)
            keyword_filter: BigQuery key_word 필터. " OR "로 구분하면 여러 값 OR 검색 (기본: corn 관련 + united states department of agriculture)
            top_k: 반환할 상위 키워드 수
        
        Returns:
            Dict containing:
                - top_entities: 상위 엔티티와 PageRank 점수
                - top_triples: 핵심 엔티티(top_entities)가 포함된 triple 중 중요도(엣지 실제 weight × entity PageRank) 상위 10개, 각 항목 {"triple": [s,v,o], "importance": 점수}
                - top_verbs: 상위 verb와 contribution 점수
                - graph_stats: 그래프 통계 (노드 수, 엣지 수)
                - triples_count: 분석된 triples 수
        """
        # key_word 조건: " OR "로 구분하면 여러 값 OR 검색
        parts = [p.strip() for p in keyword_filter.split(" OR ") if p.strip()]
        if not parts:
            parts = [keyword_filter]
        key_conditions = " OR ".join(
            f"key_word = '{k.replace(chr(39), chr(39) + chr(39))}'" for k in parts
        )
        where_clause = f"filter_status = 'T' AND ({key_conditions})"
        # 1. BigQuery에서 데이터 가져오기
        news_data = self.client.get_timeseries_data(
            table_id="news_article",
            value_column=["triples"],
            date_column="publish_date",
            base_date=target_date,
            days=days,
            where_clause=where_clause
        )
        
        if not news_data:
            return {
                "top_entities": [],
                "top_verbs": [],
                "graph_stats": {"nodes": 0, "edges": 0},
                "triples_count": 0,
                "error": "데이터를 찾을 수 없습니다."
            }
        
        # 2. Triples 추출
        triples = self._extract_triples(news_data)
        
        if not triples:
            return {
                "top_entities": [],
                "top_verbs": [],
                "graph_stats": {"nodes": 0, "edges": 0},
                "triples_count": 0,
                "error": "유효한 triples가 없습니다."
            }
        
        # 3. 정규화 및 필터링 (entity_triples용 원본은 raw_triples_kept에 보관)
        normalized_triples, normalized_entities, raw_triples_kept = self._normalize_triples(triples)

        # 4. 클러스터링 (선택적)
        final_triples, final_entities = self._cluster_entities(
            normalized_triples, normalized_entities
        )
        
        # 5. 그래프 생성 및 PageRank 계산
        pagerank_scores, verb_scores, graph = self._compute_pagerank(
            final_triples, final_entities
        )
        
        # 6. 결과 필터링 및 정렬
        filtered_pagerank = {
            e: score for e, score in pagerank_scores.items()
            if not _contains_number(e) and e.lower() not in NOISE_ENTITIES
        }
        
        top_entities = sorted(
            filtered_pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        top_verbs = sorted(
            verb_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 핵심 엔티티(top_entities)가 포함된 triple만 대상으로, 중요도(엣지 실제 weight × entity PageRank) 상위 10개만 원본 [s,v,o]로 수집
        # graph 엣지 weight = verb 중요도(contribution·stance·price_demand 반영)가 반영된 실제 weight
        top_entity_names = {e for e, _ in top_entities}
        candidates = []
        for i in range(len(final_triples)):
            s_f, v_f, o_f = final_triples[i][0], final_triples[i][1], final_triples[i][2]
            if s_f not in top_entity_names and o_f not in top_entity_names:
                continue
            raw = raw_triples_kept[i]
            edge_weight = graph[s_f][o_f].get("weight", 0.0) if graph.has_edge(s_f, o_f) else 0.0
            entity_score = pagerank_scores.get(s_f, 0.0) + pagerank_scores.get(o_f, 0.0)
            importance = edge_weight * entity_score
            candidates.append((importance, raw))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_triples = [
            {"triple": raw, "importance": round(imp, 4)}
            for imp, raw in candidates[:10]
        ]
        
        return {
            "top_entities": [
                {"entity": e, "score": round(s, 4)} for e, s in top_entities
            ],
            "top_triples": top_triples,
            "top_verbs": [
                {"verb": v, "score": round(s, 4)} for v, s in top_verbs
            ],
            "graph_stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges()
            },
            "triples_count": len(final_triples),
            "analysis_date": target_date,
            "analysis_days": days
        }
    
    def _extract_triples(self, news_data: List[Dict]) -> List[List[str]]:
        """뉴스 데이터에서 triples 추출"""
        triples = []
        for article in news_data:
            triples_raw = article.get('triples')
            if triples_raw:
                try:
                    if isinstance(triples_raw, str):
                        triples_list = json.loads(triples_raw)
                    else:
                        triples_list = triples_raw
                    if isinstance(triples_list, list):
                        for triple in triples_list:
                            if isinstance(triple, list) and len(triple) >= 3:
                                triples.append([triple[0], triple[1], triple[2]])
                except Exception:
                    continue
        return triples
    
    def _normalize_triples(
        self, triples: List[List[str]]
    ) -> Tuple[List[List[str]], set, List[List[str]]]:
        """Triples 정규화 및 필터링. 유지된 raw triple 목록도 동일 순서로 반환."""
        normalized_entities_set = set()
        normalized_triples = []
        raw_triples_kept: List[List[str]] = []

        for s, v, o in triples:
            if not _is_valid_entity(s) or not _is_valid_entity(o):
                continue

            raw_triples_kept.append([s, v, o])

            s_normalized = _normalize_entity(s, aggressive=True)
            o_normalized = _normalize_entity(o, aggressive=True)

            if s_normalized:
                normalized_entities_set.add(s_normalized)
            if o_normalized:
                normalized_entities_set.add(o_normalized)

            # 대문자 고유명사 우선
            if s.isupper() and len(s) > 1:
                normalized_entities_set.add(s)
                s_normalized = s
            if o.isupper() and len(o) > 1:
                normalized_entities_set.add(o)
                o_normalized = o

            normalized_triples.append([s_normalized, v, o_normalized])

        # 복수형을 단수형으로 통합
        plural_to_singular_map = {}
        entities_to_remove = set()

        for entity in normalized_entities_set:
            if ' ' not in entity and "'" not in entity and entity.endswith('s') and len(entity) > 3:
                singular_candidate = entity[:-1]
                if singular_candidate in normalized_entities_set:
                    plural_to_singular_map[entity] = singular_candidate
                    entities_to_remove.add(entity)

        normalized_entities_set -= entities_to_remove

        for i, (s, v, o) in enumerate(normalized_triples):
            if s in plural_to_singular_map:
                normalized_triples[i][0] = plural_to_singular_map[s]
            if o in plural_to_singular_map:
                normalized_triples[i][2] = plural_to_singular_map[o]

        return normalized_triples, normalized_entities_set, raw_triples_kept
    
    def _cluster_entities(
        self,
        normalized_triples: List[List[str]],
        normalized_entities: set
    ) -> Tuple[List[List[str]], set]:
        """임베딩 기반 엔티티 클러스터링"""
        if not EMBEDDING_AVAILABLE or not normalized_entities:
            return normalized_triples, normalized_entities
        
        all_entities = list(normalized_entities)
        
        try:
            entity_embeddings_arr = _embedding_model.encode(all_entities)
        except Exception:
            entity_embeddings_arr = None
        
        entity_cluster_map = _cluster_similar_words(
            all_entities,
            similarity_threshold=CLUSTERING_SIMILARITY_THRESHOLD,
            embeddings=entity_embeddings_arr
        )
        
        clustered_triples = []
        for s, v, o in normalized_triples:
            s_clustered = entity_cluster_map.get(s, s)
            o_clustered = entity_cluster_map.get(o, o)
            
            if s_clustered not in normalized_entities:
                s_clustered = s
            if o_clustered not in normalized_entities:
                o_clustered = o
            
            clustered_triples.append([s_clustered, v, o_clustered])
        
        final_entities = set(entity_cluster_map.values())
        for entity in normalized_entities:
            if entity not in entity_cluster_map:
                final_entities.add(entity)
        
        return clustered_triples, final_entities
    
    def _compute_pagerank(
        self,
        triples: List[List[str]],
        entities: set
    ) -> Tuple[Dict[str, float], Dict[str, float], nx.DiGraph]:
        """NetworkX 그래프 생성 및 PageRank 계산"""
        G = nx.DiGraph()
        
        # 노드 추가
        for e in entities:
            G.add_node(e)
        
        # 엣지 추가
        for s, v, o in triples:
            if s in entities and o in entities:
                weight = BE_VERB_WEIGHT if _is_meaningless_be(v) else DEFAULT_EDGE_WEIGHT
                G.add_edge(s, o, verb=v, weight=weight)
        
        if G.number_of_edges() == 0:
            return {}, {}, G
        
        # 1단계: 초기 PageRank 계산
        pagerank_initial = nx.pagerank(G, weight="weight")
        
        # Verb Average Contribution 계산
        verb_scores = defaultdict(list)
        for s, o, data in G.edges(data=True):
            verb = data["verb"]
            contribution = pagerank_initial[s] + pagerank_initial[o]
            verb_scores[verb].append(contribution)
        
        verb_avg = {
            verb: sum(vals) / len(vals)
            for verb, vals in verb_scores.items()
        }
        
        # 2단계: Verb Contribution 기반 weight 재조정
        if verb_avg:
            unique_verbs = list({data["verb"] for _, _, data in G.edges(data=True)})
            
            # STANCE / PRICE_DEMAND 임베딩 유사도 판별
            stance_verb_set = set()
            price_demand_verb_set = set()
            
            if EMBEDDING_AVAILABLE and unique_verbs:
                try:
                    stance_list = list(STANCE_KEYWORDS)
                    price_demand_list = list(PRICE_DEMAND_VERBS)
                    verb_embeddings = _embedding_model.encode(unique_verbs)
                    stance_embeddings = _embedding_model.encode(stance_list)
                    price_demand_embeddings = _embedding_model.encode(price_demand_list)
                    sim_stance = cosine_similarity(verb_embeddings, stance_embeddings)
                    sim_price_demand = cosine_similarity(verb_embeddings, price_demand_embeddings)
                    for i, verb in enumerate(unique_verbs):
                        if sim_stance[i].max() >= STANCE_SIMILARITY_THRESHOLD:
                            stance_verb_set.add(verb)
                        if sim_price_demand[i].max() >= PRICE_DEMAND_SIMILARITY_THRESHOLD:
                            price_demand_verb_set.add(verb)
                except Exception:
                    v_lower = lambda v: (v or "").lower()
                    stance_verb_set = {v for v in unique_verbs if any(kw in v_lower(v) for kw in STANCE_KEYWORDS)}
                    price_demand_verb_set = {v for v in unique_verbs if any(phrase in v_lower(v) for phrase in PRICE_DEMAND_VERBS)}
            else:
                v_lower = lambda v: (v or "").lower()
                stance_verb_set = {v for v in unique_verbs if any(kw in v_lower(v) for kw in STANCE_KEYWORDS)}
                price_demand_verb_set = {v for v in unique_verbs if any(phrase in v_lower(v) for phrase in PRICE_DEMAND_VERBS)}
            
            max_contrib = max(verb_avg.values()) if verb_avg else 1.0
            log_max_contrib = np.log1p(max_contrib) if max_contrib > 0 else 1.0
            
            for s, o, data in G.edges(data=True):
                verb = data["verb"]
                if _is_meaningless_be(verb):
                    w = BE_VERB_WEIGHT
                else:
                    contrib = verb_avg.get(verb, 0.0)
                    if log_max_contrib <= 0:
                        w = WEIGHT_MIN
                    else:
                        ratio = np.log1p(contrib) / log_max_contrib
                        w = WEIGHT_MIN + ratio * (WEIGHT_MAX - WEIGHT_MIN)
                        w = float(min(max(w, WEIGHT_MIN), WEIGHT_MAX))
                
                stance_prior = STANCE_MIN_WEIGHT if verb in stance_verb_set else 0.0
                price_prior = PRICE_DEMAND_WEIGHT if verb in price_demand_verb_set else 0.0
                w = max(w, stance_prior, price_prior)
                
                G[s][o]['weight'] = w
        
        # 3단계: 최종 PageRank 계산
        pagerank = nx.pagerank(G, weight="weight")
        
        # 최종 Verb Contribution 계산
        verb_scores_final = defaultdict(list)
        for s, o, data in G.edges(data=True):
            verb = data["verb"]
            contribution = pagerank[s] + pagerank[o]
            verb_scores_final[verb].append(contribution)
        
        verb_avg_final = {
            verb: sum(vals) / len(vals)
            for verb, vals in verb_scores_final.items()
        }
        
        return pagerank, verb_avg_final, G


def analyze_keywords(target_date: str, days: int = 3, top_k: int = 20) -> str:
    """
    키워드 분석 결과를 JSON 문자열로 반환하는 함수 (LLM Tool용)
    
    Args:
        target_date: 분석 기준 날짜 (형식: "YYYY-MM-DD")
        days: 분석할 일수 (기본 3일)
        top_k: 반환할 상위 키워드 수
    
    Returns:
        JSON 형식의 분석 결과 문자열
    """
    analyzer = KeywordAnalyzer()
    result = analyzer.analyze_keywords(
        target_date=target_date,
        days=days,
        top_k=top_k
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
