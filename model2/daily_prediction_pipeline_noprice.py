"""
옥수수 가격 예측 일일 파이프라인
- 매일 실행되어 해당 날짜의 뉴스를 분석하고 다음날 가격 예측
- 앙상블 모델을 사용하여 신뢰도 높은 예측 수행
- 결과를 JSON 파일로 저장하여 팀원들과 공유

사용법:
    # 특정 날짜 예측
    python daily_prediction_pipeline.py --date 2024-02-03
    
    # 오늘 날짜 자동 예측
    python daily_prediction_pipeline.py
    
    # 여러 날짜 배치 예측
    python daily_prediction_pipeline.py --start_date 2024-02-01 --end_date 2024-02-10
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ensemble_predictor import EnsemblePredictor


# ============================================================================
# 데이터 전처리 (train_models.py와 동일)
# ============================================================================

class CornDataPreprocessor:
    """옥수수 뉴스 및 가격 데이터 전처리"""
    
    def __init__(self, news_path, price_path):
        self.news_df = pd.read_csv(news_path)
        self.price_df = pd.read_csv(price_path)
        
        # 날짜 변환
        self.news_df['publish_date'] = pd.to_datetime(self.news_df['publish_date'])
        self.price_df['time'] = pd.to_datetime(self.price_df['time'])
    
    def prepare_single_day_data(self, target_date, lookback_days=7):
        """
        특정 날짜의 예측 데이터 준비
        
        Args:
            target_date: 예측 대상 날짜 (str or datetime)
            lookback_days: 과거 며칠의 뉴스를 볼 것인가
        
        Returns:
            dict or None: 전처리된 데이터 (뉴스가 없으면 None)
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # 해당 날짜의 가격 정보
        price_row = self.price_df[self.price_df['time'] == target_date]
        
        if len(price_row) == 0:
            print(f"경고: {target_date.date()}의 가격 데이터가 없습니다.")
            return None
        
        price_row = price_row.iloc[0]
        
        # 해당 날짜 이전 lookback_days 동안의 뉴스 수집
        start_date = target_date - timedelta(days=lookback_days)
        relevant_news = self.news_df[
            (self.news_df['publish_date'] >= start_date) & 
            (self.news_df['publish_date'] < target_date) &
            (self.news_df['filter_status'] == 'T')
        ].copy()
        
        if len(relevant_news) == 0:
            print(f"경고: {target_date.date()}의 관련 뉴스가 없습니다.")
            return None
        
        # NaN 처리
        relevant_news['positive_score'] = relevant_news['positive_score'].fillna(0)
        relevant_news['negative_score'] = relevant_news['negative_score'].fillna(0)
        relevant_news['neutral_score'] = relevant_news['neutral_score'].fillna(0)
        
        # 뉴스 임베딩 파싱
        article_embeddings = []
        entity_embeddings = []
        triple_embeddings = []
        
        for _, news in relevant_news.iterrows():
            try:
                art_emb = self._parse_embedding(news['article_embedding'])
                if art_emb is not None:
                    article_embeddings.append(art_emb)
                
                if pd.notna(news.get('entity_embedding')):
                    ent_emb = self._parse_embedding(news['entity_embedding'])
                    if ent_emb is not None:
                        entity_embeddings.append(ent_emb)
                
                if pd.notna(news.get('triple_embedding')):
                    tri_emb = self._parse_embedding(news['triple_embedding'])
                    if tri_emb is not None:
                        triple_embeddings.append(tri_emb)
            except:
                continue
        
        if len(article_embeddings) == 0:
            print(f"경고: {target_date.date()}의 유효한 임베딩이 없습니다.")
            return None
        
        # 평균 임베딩 계산
        avg_article_emb = np.mean(article_embeddings, axis=0)
        avg_entity_emb = np.mean(entity_embeddings, axis=0) if entity_embeddings else np.zeros(1024)
        avg_triple_emb = np.mean(triple_embeddings, axis=0) if triple_embeddings else np.zeros(1024)
        
        # 감성 점수 집계
        sentiment_features = {
            'avg_price_impact': relevant_news['price_impact_score'].mean(),
            'avg_positive': relevant_news['positive_score'].mean(),
            'avg_negative': relevant_news['negative_score'].mean(),
            'avg_neutral': relevant_news['neutral_score'].mean(),
            'sentiment_std': relevant_news['price_impact_score'].std() if len(relevant_news) > 1 else 0,
            'news_count': len(relevant_news),
            'positive_count': int((relevant_news['sentiment'] == 'positive').sum()),
            'negative_count': int((relevant_news['sentiment'] == 'negative').sum()),
            'neutral_count': int((relevant_news['sentiment'] == 'neutral').sum()),
        }
        
        # 가격 특성
        price_features = {
            'open': float(price_row['open']),
            'high': float(price_row['high']),
            'low': float(price_row['low']),
            'close': float(price_row['close']),
            'volume': int(price_row['Volume']),
            'ema': float(price_row['EMA']),
            'volatility': float((price_row['high'] - price_row['low']) / price_row['close'])
        }
        
        return {
            'date': target_date,
            'article_embedding': avg_article_emb,
            'entity_embedding': avg_entity_emb,
            'triple_embedding': avg_triple_emb,
            'sentiment_features': sentiment_features,
            'price_features': price_features,
            'news_articles': relevant_news[[
                'id', 'title', 'description', 'sentiment', 'price_impact_score',
                'positive_score', 'negative_score', 'neutral_score', 'named_entities'
            ]].to_dict('records')
        }
    
    def _parse_embedding(self, emb_str):
        """문자열로 저장된 임베딩을 numpy array로 변환"""
        if pd.isna(emb_str):
            return None
        
        try:
            if isinstance(emb_str, str):
                emb = json.loads(emb_str)
            else:
                emb = emb_str
            return np.array(emb, dtype=np.float32)
        except:
            return None


# ============================================================================
# 예측 결과 생성기
# ============================================================================

class PredictionReportGenerator:
    """예측 결과를 사용자 친화적인 보고서로 변환"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def generate_report(self, processed_data, ensemble_result):
        """
        종합 예측 보고서 생성
        
        Args:
            processed_data: 전처리된 데이터
            ensemble_result: 앙상블 예측 결과
        
        Returns:
            dict: 최종 보고서
        """
        # 예측 날짜 계산 (다음날 예측)
        prediction_date = processed_data['date'] + timedelta(days=1)
        if isinstance(prediction_date, pd.Timestamp):
            prediction_date_str = prediction_date.strftime('%Y-%m-%d')
            base_date_str = processed_data['date'].strftime('%Y-%m-%d')
        else:
            prediction_date_str = str(prediction_date)[:10]
            base_date_str = str(processed_data['date'])[:10]
        
        # 주요 뉴스 추출 (price_impact_score 기준)
        news_articles = processed_data['news_articles']
        sorted_articles = sorted(
            news_articles,
            key=lambda x: abs(x.get('price_impact_score', 0)),
            reverse=True
        )
        
        # 예측 방향에 따른 증거/반대 증거 분류
        direction = ensemble_result['direction']
        evidence = []
        counter = []
        
        for article in sorted_articles[:15]:  # 상위 15개만 확인
            impact = article.get('price_impact_score', 0)
            
            # 엔티티 파싱
            entities_str = article.get('named_entities', '{}')
            try:
                if isinstance(entities_str, str):
                    entities = json.loads(entities_str)
                else:
                    entities = entities_str
                key_entities = list(entities.keys())[:5] if entities else []
            except:
                key_entities = []
            
            article_info = {
                'article_id': int(article['id']),
                'title': article['title'],
                'description': article.get('description', ''),
                'sentiment': article['sentiment'],
                'impact_score': round(float(impact), 3),
                'positive_score': round(float(article.get('positive_score', 0)), 3),
                'negative_score': round(float(article.get('negative_score', 0)), 3),
                'neutral_score': round(float(article.get('neutral_score', 0)), 3),
                'key_entities': key_entities
            }
            
            # 예측 방향과 일치하는지 판단
            if direction == "상승" and impact > 0.1:
                evidence.append(article_info)
            elif direction == "하락" and impact < -0.1:
                evidence.append(article_info)
            elif direction == "유지" and abs(impact) < 0.1:
                evidence.append(article_info)
            else:
                counter.append(article_info)
            
            # 각각 최대 5개까지만
            if len(evidence) >= 5 and len(counter) >= 3:
                break
        
        # 감성 요약
        sentiment_summary = {
            'total_news_count': processed_data['sentiment_features']['news_count'],
            'avg_price_impact': round(processed_data['sentiment_features']['avg_price_impact'], 3),
            'avg_positive_score': round(processed_data['sentiment_features']['avg_positive'], 3),
            'avg_negative_score': round(processed_data['sentiment_features']['avg_negative'], 3),
            'avg_neutral_score': round(processed_data['sentiment_features']['avg_neutral'], 3),
            'positive_ratio': round(
                processed_data['sentiment_features']['positive_count'] / 
                processed_data['sentiment_features']['news_count'], 3
            ),
            'negative_ratio': round(
                processed_data['sentiment_features']['negative_count'] / 
                processed_data['sentiment_features']['news_count'], 3
            ),
            'neutral_ratio': round(
                processed_data['sentiment_features']['neutral_count'] / 
                processed_data['sentiment_features']['news_count'], 3
            )
        }
        
        # 상세한 reasoning 생성
        detailed_reasoning = self._generate_detailed_reasoning(
            direction,
            ensemble_result,
            sentiment_summary,
            evidence
        )
        
        # 최종 보고서
        report = {
            'metadata': {
                'base_date': base_date_str,
                'prediction_date': prediction_date_str,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'lookback_days': 7
            },
            'prediction': {
                'direction': direction,
                'confidence': round(ensemble_result['confidence'], 3),
                'agreement_level': ensemble_result['agreement_level'],
                'probabilities': {
                    k: round(v, 3) for k, v in ensemble_result['probabilities'].items()
                }
            },
            'model_consensus': {
                'ensemble_reasoning': ensemble_result['reasoning'],
                'detailed_reasoning': detailed_reasoning,
                'model_details': ensemble_result['model_details']
            },
            'evidence': {
                'supporting_news': evidence[:5],
                'opposing_news': counter[:3]
            },
            'market_analysis': {
                'sentiment_summary': sentiment_summary,
                'price_info': {
                    'current_close': processed_data['price_features']['close'],
                    'current_volume': processed_data['price_features']['volume'],
                    'volatility': round(processed_data['price_features']['volatility'], 3),
                    'ema': processed_data['price_features']['ema']
                }
            }
        }
        
        return report
    
    def _generate_detailed_reasoning(self, direction, ensemble_result, sentiment_summary, evidence):
        """상세한 예측 근거 생성"""
        reasons = []
        
        # 감성 분석 기반
        if direction == "상승":
            if sentiment_summary['positive_ratio'] > 0.5:
                reasons.append(
                    f"긍정 기사 비율 {sentiment_summary['positive_ratio']*100:.1f}%로 시장 낙관론 우세"
                )
            if sentiment_summary['avg_price_impact'] > 0.1:
                reasons.append(
                    f"평균 가격 영향도 {sentiment_summary['avg_price_impact']:.2f}로 긍정적"
                )
        elif direction == "하락":
            if sentiment_summary['negative_ratio'] > 0.5:
                reasons.append(
                    f"부정 기사 비율 {sentiment_summary['negative_ratio']*100:.1f}%로 시장 비관론 확산"
                )
            if sentiment_summary['avg_price_impact'] < -0.1:
                reasons.append(
                    f"평균 가격 영향도 {sentiment_summary['avg_price_impact']:.2f}로 부정적"
                )
        else:  # 유지
            if sentiment_summary['neutral_ratio'] > 0.4:
                reasons.append(
                    f"중립 기사 비율 {sentiment_summary['neutral_ratio']*100:.1f}%로 시장 관망세"
                )
        
        # 주요 기사 기반
        if evidence:
            top_article = evidence[0]
            if abs(top_article['impact_score']) > 0.2:
                reasons.append(
                    f"주요 이슈 (영향도 {top_article['impact_score']:.2f}): {top_article['title'][:60]}..."
                )
        
        # 모델 합의 기반
        if ensemble_result['agreement_level'] == 'high':
            reasons.append("3개 앙상블 모델 강한 합의")
        elif ensemble_result['agreement_level'] == 'medium':
            reasons.append("모델 간 부분 합의, 중간 수준 신뢰도")
        else:
            reasons.append("모델 간 의견 차이 존재, 신중한 해석 필요")
        
        if not reasons:
            reasons.append("중립적 뉴스 흐름, 제한적 가격 변동 예상")
        
        return " | ".join(reasons)


# ============================================================================
# 메인 파이프라인
# ============================================================================

def predict_single_day(predictor, preprocessor, target_date, output_dir='outputs'):
    """
    특정 날짜의 예측 수행
    
    Args:
        predictor: EnsemblePredictor 인스턴스
        preprocessor: CornDataPreprocessor 인스턴스
        target_date: 예측 대상 날짜 (str or datetime)
        output_dir: 출력 디렉토리
    
    Returns:
        dict: 예측 보고서 (성공 시) or None (실패 시)
    """
    print(f"\n{'='*80}")
    print(f"예측 시작: {target_date}")
    print(f"{'='*80}")
    
    # 데이터 준비
    print("[1/3] 데이터 전처리 중...")
    processed_data = preprocessor.prepare_single_day_data(target_date, lookback_days=7)
    
    if processed_data is None:
        print(f"실패: {target_date}의 데이터를 준비할 수 없습니다.")
        return None
    
    print(f"  ✓ {processed_data['sentiment_features']['news_count']}개 뉴스 수집")
    print(f"  ✓ 평균 감성 영향도: {processed_data['sentiment_features']['avg_price_impact']:.3f}")
    
    # 특성 벡터 생성
    features = np.concatenate([
        processed_data['article_embedding'],  # 512
        processed_data['entity_embedding'],   # 1024
        processed_data['triple_embedding'],   # 1024
        np.array([
            processed_data['sentiment_features']['avg_price_impact'],
            processed_data['sentiment_features']['avg_positive'],
            processed_data['sentiment_features']['avg_negative'],
            processed_data['sentiment_features']['avg_neutral'],
            processed_data['sentiment_features']['sentiment_std'] if not np.isnan(processed_data['sentiment_features']['sentiment_std']) else 0,
            processed_data['sentiment_features']['news_count'],
            processed_data['sentiment_features']['positive_count'],
            processed_data['sentiment_features']['negative_count'],
            processed_data['sentiment_features']['neutral_count'],
        ]),  # 9
        # np.array([
        #     processed_data['price_features']['open'],
        #     processed_data['price_features']['high'],
        #     processed_data['price_features']['low'],
        #     processed_data['price_features']['close'],
        #     processed_data['price_features']['volume'],
        #     processed_data['price_features']['ema'],
        #     processed_data['price_features']['volatility']
        # ])  # 7
    ])  # Total: 2569 (price 제외)
    
    # 앙상블 예측
    print("\n[2/3] 앙상블 예측 중...")
    ensemble_result = predictor.predict_single(features)
    
    print(f"  ✓ 예측 방향: {ensemble_result['direction']}")
    print(f"  ✓ 신뢰도: {ensemble_result['confidence']:.2%}")
    print(f"  ✓ 합의 수준: {ensemble_result['agreement_level']}")
    
    # 보고서 생성
    print("\n[3/3] 보고서 생성 중...")
    report_generator = PredictionReportGenerator(predictor)
    report = report_generator.generate_report(processed_data, ensemble_result)
    
    # 파일 저장
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(target_date, str):
        date_str = target_date
    elif isinstance(target_date, pd.Timestamp):
        date_str = target_date.strftime('%Y-%m-%d')
    else:
        date_str = str(target_date)[:10]
    
    output_path = os.path.join(output_dir, f'news_prediction_{date_str}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 보고서 저장: {output_path}")
    print(f"\n{'='*80}")
    print("예측 완료!")
    print(f"{'='*80}\n")
    
    return report


def predict_date_range(predictor, preprocessor, start_date, end_date, output_dir='outputs'):
    """
    날짜 범위에 대한 배치 예측
    
    Args:
        predictor: EnsemblePredictor 인스턴스
        preprocessor: CornDataPreprocessor 인스턴스
        start_date: 시작 날짜 (str or datetime)
        end_date: 종료 날짜 (str or datetime)
        output_dir: 출력 디렉토리
    
    Returns:
        list: 예측 보고서 리스트
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    print(f"\n{'='*80}")
    print(f"배치 예측: {start_date.date()} ~ {end_date.date()}")
    print(f"{'='*80}\n")
    
    # 날짜 범위 생성
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    results = []
    success_count = 0
    fail_count = 0
    
    for current_date in date_range:
        try:
            report = predict_single_day(predictor, preprocessor, current_date, output_dir)
            if report is not None:
                results.append(report)
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"오류 발생 ({current_date.date()}): {e}")
            fail_count += 1
    
    print(f"\n{'='*80}")
    print(f"배치 예측 완료: 성공 {success_count}건, 실패 {fail_count}건")
    print(f"{'='*80}\n")
    
    return results


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='옥수수 가격 일일 예측 파이프라인')
    
    # 데이터 경로
    parser.add_argument('--news_path', type=str, default='corn_all_news_with_sentiment.csv',
                       help='뉴스 데이터 CSV 파일 경로')
    parser.add_argument('--price_path', type=str, default='corn_future_price.csv',
                       help='가격 데이터 CSV 파일 경로')
    
    # 모델 경로
    parser.add_argument('--model_dir', type=str, default='trained_models',
                       help='학습된 모델 디렉토리')
    
    # 예측 날짜
    parser.add_argument('--date', type=str, default=None,
                       help='예측 날짜 (YYYY-MM-DD), 없으면 오늘 날짜')
    parser.add_argument('--start_date', type=str, default=None,
                       help='배치 예측 시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='배치 예측 종료 날짜 (YYYY-MM-DD)')
    
    # 출력 경로
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='예측 결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("옥수수 가격 예측 파이프라인")
    print("="*80)
    
    # 1. 데이터 로더 초기화
    print("\n[1/3] 데이터 로더 초기화 중...")
    preprocessor = CornDataPreprocessor(
        news_path=args.news_path,
        price_path=args.price_path
    )
    print(f"  ✓ 뉴스 데이터: {len(preprocessor.news_df)}개 기사")
    print(f"  ✓ 가격 데이터: {len(preprocessor.price_df)}개 레코드")
    
    # 2. 앙상블 예측기 로드
    print(f"\n[2/3] 앙상블 예측기 로드 중... (from {args.model_dir})")
    predictor = EnsemblePredictor(model_dir=args.model_dir)
    
    # 3. 예측 수행
    print(f"\n[3/3] 예측 수행 중...")
    
    if args.start_date and args.end_date:
        # 배치 예측
        results = predict_date_range(
            predictor, 
            preprocessor, 
            args.start_date, 
            args.end_date,
            args.output_dir
        )
        
        # 요약 출력
        if results:
            print("\n예측 결과 요약:")
            for result in results[-5:]:  # 마지막 5개만 출력
                print(f"  - {result['metadata']['prediction_date']}: "
                      f"{result['prediction']['direction']} "
                      f"(신뢰도 {result['prediction']['confidence']:.2%})")
    
    else:
        # 단일 예측
        if args.date:
            target_date = args.date
        else:
            # 오늘 날짜 (또는 데이터에서 가장 최근 날짜)
            price_date = preprocessor.price_df['time'].max()
            news_date = preprocessor.news_df['publish_date'].max()
            target_date = min(price_date, news_date)
            print(f"  ℹ️  날짜 미지정, 최근 데이터 사용: {target_date.date()}")
        
        result = predict_single_day(
            predictor,
            preprocessor,
            target_date,
            args.output_dir
        )
        
        # 결과 출력
        if result:
            print("\n📊 예측 결과:")
            print(f"  예측 날짜: {result['metadata']['prediction_date']}")
            print(f"  예측 방향: {result['prediction']['direction']}")
            print(f"  신뢰도: {result['prediction']['confidence']:.2%}")
            print(f"  합의 수준: {result['prediction']['agreement_level']}")
            print(f"\n  확률 분포:")
            for k, v in result['prediction']['probabilities'].items():
                print(f"    {k}: {v:.2%}")
            print(f"\n  예측 근거:")
            print(f"    {result['model_consensus']['detailed_reasoning']}")
            
            print(f"\n  뉴스 분석:")
            print(f"    총 뉴스 수: {result['market_analysis']['sentiment_summary']['total_news_count']}")
            print(f"    긍정 비율: {result['market_analysis']['sentiment_summary']['positive_ratio']:.2%}")
            print(f"    부정 비율: {result['market_analysis']['sentiment_summary']['negative_ratio']:.2%}")
            print(f"    중립 비율: {result['market_analysis']['sentiment_summary']['neutral_ratio']:.2%}")
    
    print("\n" + "="*80)
    print("파이프라인 실행 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
