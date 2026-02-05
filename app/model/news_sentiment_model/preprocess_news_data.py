"""
옥수수 뉴스 데이터 전처리 스크립트
원본 데이터(news_articles_resources.csv)를 필터링하고 감성 분석을 수행하여
학습 가능한 데이터(corn_all_news_with_sentiment.csv)를 생성합니다.

사용법:
    python preprocess_news_data.py \
        --input news_articles_resources.csv \
        --output corn_all_news_with_sentiment.csv
"""

import pandas as pd
import argparse
import re
from datetime import datetime
from finbert import (
    CommoditySentimentAnalyzer,
    prepare_text_for_analysis,
    get_sentiment_summary
)


def filter_corn_news(df, keyword_pattern=None):
    """
    옥수수 관련 뉴스 필터링
    
    필터링 조건:
    1. filter_status == 'T'
    2. key_word에 corn AND (price OR demand OR supply OR inventory) 포함
    
    Args:
        df: 원본 데이터프레임
        keyword_pattern: 커스텀 키워드 패턴 (없으면 기본 패턴 사용)
    
    Returns:
        DataFrame: 필터링된 데이터프레임
    """
    print("\n[1/5] 뉴스 데이터 필터링 중...")
    print(f"원본 데이터: {len(df)}개 기사")
    
    # 1. filter_status == 'T' 필터링
    if 'filter_status' in df.columns:
        df_filtered = df[df['filter_status'] == 'T'].copy()
        print(f"  ✓ filter_status='T' 필터링: {len(df_filtered)}개 기사")
    else:
        print("  ⚠️  'filter_status' 컬럼이 없습니다. 전체 데이터 사용")
        df_filtered = df.copy()
    
    # 2. key_word 필터링
    # corn AND (price OR demand OR supply OR inventory)
    if 'key_word' not in df_filtered.columns:
        print("  ⚠️  'key_word' 컬럼이 없습니다. 키워드 필터링 생략")
        return df_filtered
    
    if keyword_pattern is None:
        # 기본 패턴: corn이 있고, price/demand/supply/inventory 중 하나 이상 포함
        def matches_keyword(keyword_str):
            if pd.isna(keyword_str):
                return False
            
            keyword_lower = str(keyword_str).lower()
            
            # corn이 있는지 확인
            has_corn = 'corn' in keyword_lower
            
            # price, demand, supply, inventory 중 하나라도 있는지 확인
            has_market_terms = any(term in keyword_lower for term in 
                                  ['price', 'demand', 'supply', 'inventory'])
            
            return has_corn and has_market_terms
    else:
        # 커스텀 패턴 사용
        def matches_keyword(keyword_str):
            if pd.isna(keyword_str):
                return False
            return bool(re.search(keyword_pattern, str(keyword_str), re.IGNORECASE))
    
    # 필터 적용
    mask = df_filtered['key_word'].apply(matches_keyword)
    df_filtered = df_filtered[mask].copy()
    
    print(f"  ✓ 키워드 필터링 완료: {len(df_filtered)}개 기사")
    print(f"    조건: corn AND (price OR demand OR supply OR inventory)")
    
    return df_filtered


def validate_required_columns(df):
    """
    필수 컬럼 검증
    
    Args:
        df: 검증할 데이터프레임
    
    Returns:
        bool: 모든 필수 컬럼이 있으면 True
    """
    required_cols = ['title', 'publish_date']
    optional_cols = ['description', 'all_text', 'article_embedding', 
                     'entity_embedding', 'triple_embedding', 'named_entities']
    
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        print(f"\n❌ 오류: 필수 컬럼이 없습니다: {missing_required}")
        return False
    
    missing_optional = [col for col in optional_cols if col not in df.columns]
    if missing_optional:
        print(f"\n⚠️  경고: 일부 선택적 컬럼이 없습니다: {missing_optional}")
        print("   학습 성능이 저하될 수 있습니다.")
    
    return True


def add_missing_columns(df):
    """
    학습에 필요하지만 없는 컬럼을 기본값으로 추가
    
    Args:
        df: 데이터프레임
    
    Returns:
        DataFrame: 컬럼이 추가된 데이터프레임
    """
    df_result = df.copy()
    
    # 임베딩 컬럼이 없으면 빈 리스트로 초기화
    embedding_cols = ['article_embedding', 'entity_embedding', 'triple_embedding']
    for col in embedding_cols:
        if col not in df_result.columns:
            print(f"  ⚠️  '{col}' 컬럼이 없습니다. 빈 값으로 초기화합니다.")
            df_result[col] = None
    
    # named_entities 컬럼이 없으면 빈 딕셔너리로 초기화
    if 'named_entities' not in df_result.columns:
        print(f"  ⚠️  'named_entities' 컬럼이 없습니다. 빈 값으로 초기화합니다.")
        df_result['named_entities'] = '{}'
    
    # description이 없으면 빈 문자열
    if 'description' not in df_result.columns:
        print(f"  ⚠️  'description' 컬럼이 없습니다. 빈 값으로 초기화합니다.")
        df_result['description'] = ''
    
    return df_result


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='옥수수 뉴스 데이터 전처리 (필터링 + 감성 분석)'
    )
    
    # 입출력 파일
    parser.add_argument('--input', type=str, default='news_articles_resources.csv',
                       help='원본 뉴스 데이터 CSV 파일 (기본값: news_articles_resources.csv)')
    parser.add_argument('--output', type=str, default='corn_all_news_with_sentiment.csv',
                       help='출력 파일명 (기본값: corn_all_news_with_sentiment.csv)')
    
    # 필터링 옵션
    parser.add_argument('--keyword_pattern', type=str, default=None,
                       help='커스텀 키워드 정규표현식 패턴 (기본값: None, 자동 패턴 사용)')
    parser.add_argument('--skip_filter', action='store_true',
                       help='필터링을 건너뛰고 전체 데이터 사용')
    
    # 감성 분석 옵션
    parser.add_argument('--model_name', type=str, default='ProsusAI/finbert',
                       help='감성 분석 모델명 (기본값: ProsusAI/finbert)')
    parser.add_argument('--skip_sentiment', action='store_true',
                       help='감성 분석 건너뛰기 (이미 분석된 데이터인 경우)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("옥수수 뉴스 데이터 전처리 파이프라인")
    print("="*80)
    print(f"\n입력 파일: {args.input}")
    print(f"출력 파일: {args.output}")
    print(f"감성 분석 모델: {args.model_name}")
    
    # ========================================================================
    # STEP 1: 데이터 로드
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 1: 데이터 로드")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(args.input)
        print(f"✓ 데이터 로드 완료: {len(df)}개 기사")
        print(f"  컬럼: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {args.input}")
        return
    except Exception as e:
        print(f"❌ 오류: 파일 로드 실패: {e}")
        return
    
    # 필수 컬럼 검증
    if not validate_required_columns(df):
        return
    
    # ========================================================================
    # STEP 2: 필터링
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: 뉴스 필터링")
    print(f"{'='*80}")
    
    if args.skip_filter:
        print("⏭️  필터링을 건너뜁니다 (--skip_filter 옵션)")
        df_filtered = df.copy()
    else:
        df_filtered = filter_corn_news(df, keyword_pattern=args.keyword_pattern)
        
        if len(df_filtered) == 0:
            print("\n❌ 오류: 필터링 결과가 비어있습니다. 조건을 확인하세요.")
            return
    
    # ========================================================================
    # STEP 3: 누락된 컬럼 추가
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: 데이터 검증 및 보완")
    print(f"{'='*80}")
    
    df_filtered = add_missing_columns(df_filtered)
    
    # ========================================================================
    # STEP 4: 텍스트 준비
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: 감성 분석용 텍스트 준비")
    print(f"{'='*80}")
    
    print("\n[2/5] 텍스트 결합 중...")
    df_prepared = prepare_text_for_analysis(
        df_filtered,
        title_col='title',
        description_col='description',
        all_text_col='all_text' if 'all_text' in df_filtered.columns else None,
        output_col='combined_text'
    )
    print(f"  ✓ combined_text 컬럼 생성 완료")
    
    # ========================================================================
    # STEP 5: 감성 분석
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: 감성 분석")
    print(f"{'='*80}")
    
    if args.skip_sentiment:
        print("⏭️  감성 분석을 건너뜁니다 (--skip_sentiment 옵션)")
        df_result = df_prepared.copy()
    else:
        print("\n[3/5] FinBERT 모델 로드 중...")
        analyzer = CommoditySentimentAnalyzer(model_name=args.model_name)
        
        print("\n[4/5] 감성 분석 수행 중...")
        df_result = analyzer.analyze_dataframe(
            df_prepared,
            text_column='combined_text',
            show_progress=True
        )
        
        # 감성 분석 요약
        print("\n[5/5] 감성 분석 결과 요약:")
        summary = get_sentiment_summary(df_result)
        
        print(f"\n📊 감성 분포:")
        for sentiment, info in summary['sentiment_distribution'].items():
            print(f"  {sentiment:10s}: {info['count']:5d}개 ({info['percentage']:5.1f}%)")
        
        print(f"\n📈 통계:")
        print(f"  평균 신뢰도:        {summary['avg_confidence']:.3f}")
        print(f"  평균 가격 영향도:  {summary['avg_price_impact']:.3f}")
        print(f"  총 기사 수:        {summary['total_count']:,}개")
    
    # ========================================================================
    # STEP 6: 저장
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 6: 결과 저장")
    print(f"{'='*80}")
    
    # 날짜 형식 확인 및 정렬
    if 'publish_date' in df_result.columns:
        df_result['publish_date'] = pd.to_datetime(df_result['publish_date'])
        df_result = df_result.sort_values('publish_date')
    
    # 저장
    df_result.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n✓ 결과 저장 완료: {args.output}")
    print(f"  최종 기사 수: {len(df_result):,}개")
    print(f"  컬럼 수: {len(df_result.columns)}개")
    
    # 샘플 데이터 출력
    print(f"\n📋 샘플 데이터 (최신 3개 기사):")
    sample_cols = ['publish_date', 'title', 'sentiment', 'price_impact_score']
    available_cols = [col for col in sample_cols if col in df_result.columns]
    print(df_result[available_cols].tail(3).to_string(index=False))
    
    # ========================================================================
    # 완료
    # ========================================================================
    print(f"\n{'='*80}")
    print("✅ 전처리 완료!")
    print(f"{'='*80}")
    print(f"\n다음 단계:")
    print(f"  1. 전처리된 데이터 확인: {args.output}")
    print(f"  2. 모델 학습 실행:")
    print(f"     python train_models.py --news_path {args.output}")
    print()


if __name__ == "__main__":
    main()
