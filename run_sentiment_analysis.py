"""
뉴스 감성 분석 실행 스크립트
CSV 파일을 읽어 감성 분석을 수행하고 결과를 저장합니다.

사용법:
    python run_sentiment_analysis.py
    
    또는 커스텀 경로 지정:
    python run_sentiment_analysis.py --input my_news.csv --output result.csv
"""

import pandas as pd
import argparse
import os
from finbert import (
    CommoditySentimentAnalyzer,
    prepare_text_for_analysis,
    get_sentiment_summary,
    get_daily_sentiment_trend
)


def parse_arguments():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='뉴스 감성 분석 실행')
    
    parser.add_argument(
        '--input',
        type=str,
        default='news_articles_resources.csv',
        help='입력 CSV 파일 경로 (기본: news_articles_resources.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='corn_all_news_with_sentiment.csv',
        help='출력 CSV 파일 경로 (기본: corn_all_news_with_sentiment.csv)'
    )
    
    parser.add_argument(
        '--filter-status',
        type=str,
        default='T',
        help='filter_status 값 (기본: T, 전체 분석하려면 "all" 입력)'
    )
    
    parser.add_argument(
        '--keyword',
        type=str,
        default='corn and (price or demand or supply or inventory)',
        help='필터링할 키워드 (기본: corn and (price or demand or supply or inventory))'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='모든 필터링 무시하고 전체 데이터 분석'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='ProsusAI/finbert',
        help='사용할 모델명 (기본: ProsusAI/finbert)'
    )
    
    parser.add_argument(
        '--text-column',
        type=str,
        default='combined_text',
        help='분석할 텍스트 컬럼명 (기본: combined_text)'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='진행상황 표시 안함'
    )
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    
    # 1. 인자 파싱
    args = parse_arguments()
    
    print("=" * 80)
    print("뉴스 감성 분석 시작")
    print("=" * 80)
    print(f"입력 파일: {args.input}")
    print(f"출력 파일: {args.output}")
    print(f"모델: {args.model}")
    
    if args.no_filter:
        print(f"필터링: 미적용 (전체 데이터 분석)")
    else:
        print(f"필터링: filter_status='{args.filter_status}', keyword='{args.keyword}'")
    
    print("=" * 80)
    
    # 2. 데이터 로드
    print("\n[Step 1] 데이터 로드 중...")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input}")
    
    df = pd.read_csv(args.input)
    print(f"✓ 전체 기사 수: {len(df)}")
    
    # 3. 필터링
    if args.no_filter:
        print("\n[Step 2] 필터링 없음 - 전체 기사 분석")
        df_to_analyze = df.copy()
    else:
        print("\n[Step 2] 데이터 필터링 중...")
        df_to_analyze = df.copy()
        
        # filter_status 필터링
        if 'filter_status' in df.columns and args.filter_status.lower() != 'all':
            before_count = len(df_to_analyze)
            df_to_analyze = df_to_analyze[df_to_analyze['filter_status'] == args.filter_status].copy()
            print(f"✓ filter_status='{args.filter_status}' 필터링: {before_count} → {len(df_to_analyze)}")
        
        # keyword 필터링
        if 'key_word' in df.columns and args.keyword:
            before_count = len(df_to_analyze)
            df_to_analyze = df_to_analyze[df_to_analyze['key_word'] == args.keyword].copy()
            print(f"✓ keyword 필터링: {before_count} → {len(df_to_analyze)}")
        elif 'key_word' not in df.columns and args.keyword:
            print(f"⚠️ 'key_word' 컬럼이 없어 키워드 필터링을 건너뜁니다.")
    
    print(f"\n✓ 최종 분석 대상: {len(df_to_analyze)} 기사")
    
    if len(df_to_analyze) == 0:
        print("⚠️ 분석할 데이터가 없습니다.")
        return
    
    # 4. 텍스트 준비
    print("\n[Step 3] 분석용 텍스트 준비 중...")
    df_prepared = prepare_text_for_analysis(df_to_analyze)
    print(f"✓ 텍스트 결합 완료 (컬럼: {args.text_column})")
    
    # 5. 감성 분석 실행
    print("\n[Step 4] 감성 분석 실행 중...")
    print("※ GPU 환경에서 실행하는 것을 권장합니다.")
    print("=" * 80)
    
    analyzer = CommoditySentimentAnalyzer(model_name=args.model)
    df_with_sentiment = analyzer.analyze_dataframe(
        df_prepared,
        text_column=args.text_column,
        show_progress=not args.no_progress
    )
    
    print("=" * 80)
    print("✓ 감성 분석 완료!")
    
    # 6. 결과 요약
    print("\n[Step 5] 분석 결과 요약")
    print("=" * 80)
    
    summary = get_sentiment_summary(df_with_sentiment)
    
    print("\n감성 분포:")
    for sentiment, stats in summary['sentiment_distribution'].items():
        print(f"  {sentiment.capitalize():8s}: {stats['count']:4d} ({stats['percentage']:5.1f}%)")
    
    print(f"\n평균 신뢰도: {summary['avg_confidence']:.3f}")
    print(f"평균 가격 영향 점수: {summary['avg_price_impact']:.3f}")
    print(f"  (positive_score - negative_score의 평균)")
    
    # 7. 일별 트렌드 (publish_date가 있는 경우)
    if 'publish_date' in df_with_sentiment.columns:
        print("\n[Step 6] 일별 감성 트렌드 분석")
        print("=" * 80)
        
        daily_trend = get_daily_sentiment_trend(df_with_sentiment)
        print(f"\n최근 10일 트렌드:")
        print(daily_trend.tail(10).to_string(index=False))
    
    # 8. 결과 저장
    print("\n[Step 7] 결과 저장 중...")
    df_with_sentiment.to_csv(args.output, index=False)
    print(f"✓ 결과 저장 완료: {args.output}")
    
    # 9. 샘플 출력
    print("\n[Step 8] 샘플 결과 (상위 5개)")
    print("=" * 80)
    sample_cols = ['title', 'sentiment', 'sentiment_confidence', 'price_impact_score']
    available_cols = [col for col in sample_cols if col in df_with_sentiment.columns]
    
    if available_cols:
        print(df_with_sentiment[available_cols].head().to_string(index=False))
    
    # 10. 완료
    print("\n" + "=" * 80)
    print("감성 분석 완료!")
    print("=" * 80)
    print(f"총 분석 기사: {len(df_with_sentiment)}")
    print(f"출력 파일: {args.output}")
    print("\n다음 단계:")
    print("  1. train.py로 모델 학습")
    print("  2. inference.py로 가격 예측")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n중단됨: 사용자가 실행을 중단했습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
