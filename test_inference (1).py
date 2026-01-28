"""
Inference 테스트 스크립트 (기본 버전)
test_news_data.csv와 test_price_data.csv를 사용하여 예측 수행

사용법:
    python test_inference.py
"""

import pandas as pd
import json
from inference import predict_next_day


def main():
    print("=" * 80)
    print("Inference 테스트 시작")
    print("=" * 80)
    
    # 1. 테스트 데이터 로드
    print("\n[Step 1] 테스트 데이터 로드...")
    
    try:
        test_news = pd.read_csv('test_news_data.csv')
        test_price = pd.read_csv('test_price_data.csv')
    except FileNotFoundError as e:
        print(f"\n오류: {e}")
        print("\n먼저 다음 명령을 실행하세요:")
        print("  python prepare_inference_test_data.py")
        print("또는:")
        print("  python extract_test_data_simple.py")
        return
    
    print(f"✓ 뉴스 데이터: {len(test_news)} 건")
    print(f"✓ 가격 데이터: {len(test_price)} 건")
    
    # 날짜 범위 확인
    test_news['publish_date'] = pd.to_datetime(test_news['publish_date'])
    test_price['time'] = pd.to_datetime(test_price['time'])
    
    print(f"✓ 뉴스 기간: {test_news['publish_date'].min().date()} ~ {test_news['publish_date'].max().date()}")
    print(f"✓ 가격 기간: {test_price['time'].min().date()} ~ {test_price['time'].max().date()}")
    
    # 뉴스 감성 분포 확인
    if 'sentiment' in test_news.columns:
        print(f"\n뉴스 감성 분포:")
        sentiment_dist = test_news['sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} 건")
    
    # 2. 예측 수행
    print("\n[Step 2] 예측 수행...")
    print("=" * 80)
    
    target_date = '2025-11-15'
    
    try:
        # 기본 예측 수행
        result = predict_next_day(
            news_data=test_news,
            price_history=test_price,
            model_dir='models'
        )
        
        # 3. 결과 출력
        print("\n" + "=" * 80)
        print("예측 결과")
        print("=" * 80)
        
        print(f"\n타겟 날짜: {target_date}")
        print(f"예측: {'상승 (1)' if result['prediction'] == 1 else '하락 (0)'}")
        print(f"상승 확률: {result['probability']:.2%}")
        
        print(f"\n피처 요약:")
        fs = result['features_summary']
        print(f"  - 뉴스 기사 수: {fs['latest_news_count']} 개")
        print(f"  - 평균 감성 점수: {fs['avg_sentiment']:.3f}")
        print(f"  - 평균 가격 영향: {fs['avg_price_impact']:.3f}")
        print(f"  - 최근 가격: ${fs['latest_price']:.2f}")
        print(f"  - 사용된 데이터: {fs['data_points_used']} 일")
        
        # 4. 결과 저장
        print("\n" + "=" * 80)
        print("결과 저장 중...")
        
        with open('test_inference_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("✓ 저장 완료: test_inference_result.json")
        
        # 5. 실제 결과와 비교 (있다면)
        print("\n[Step 3] 실제 결과와 비교...")
        print("=" * 80)
        
        try:
            price_df = pd.read_csv('corn_future_price.csv')
            price_df['time'] = pd.to_datetime(price_df['time'])
            
            actual_data = price_df[price_df['time'].dt.date == pd.to_datetime(target_date).date()]
            
            if len(actual_data) > 0:
                actual_close = actual_data.iloc[0]['close']
                prev_close = test_price.iloc[-1]['close']
                actual_change = ((actual_close - prev_close) / prev_close) * 100
                
                print(f"\n실제 2025-11-15 데이터:")
                print(f"  종가: ${actual_close:.2f}")
                print(f"  전일 종가: ${prev_close:.2f}")
                print(f"  변화율: {actual_change:+.2f}%")
                
                # 실제 라벨 (0.5% 임계값 기준)
                if actual_change > 0.5:
                    actual_label = 1  # 상승
                    actual_text = "상승 (0.5% 초과)"
                else:
                    actual_label = 0  # 하락/횡보
                    actual_text = "하락/횡보 (0.5% 이하)"
                
                predicted_label = result['prediction']
                
                print(f"\n예측 vs 실제:")
                print(f"  예측: {result['prediction']} ({'상승' if predicted_label == 1 else '하락'})")
                print(f"  실제: {actual_label} ({actual_text})")
                
                if predicted_label == actual_label:
                    print(f"  결과: ✓ 정답!")
                else:
                    print(f"  결과: ✗ 오답")
                
                # 상세 분석
                print(f"\n상세 분석:")
                print(f"  - 모델 신뢰도: {result['probability']:.2%}")
                print(f"  - 실제 변화율: {actual_change:+.2f}%")
                
                if abs(actual_change) < 0.5:
                    print(f"  - 참고: 실제 변화가 작아 예측이 어려운 케이스입니다.")
                
            else:
                print("\n⚠️ 2025-11-15 실제 데이터가 없습니다.")
                print("   예측만 수행되었으며 정답 확인은 불가능합니다.")
        
        except FileNotFoundError:
            print("\n⚠️ corn_future_price.csv를 찾을 수 없습니다.")
            print("   실제 데이터 비교를 건너뜁니다.")
        
        print("\n" + "=" * 80)
        print("테스트 완료!")
        print("=" * 80)
        
        # 요약
        print("\n요약:")
        print(f"  - 타겟 날짜: {target_date}")
        print(f"  - 예측: {'상승' if result['prediction'] == 1 else '하락'}")
        print(f"  - 신뢰도: {result['probability']:.2%}")
        print(f"  - 뉴스 기사: {fs['latest_news_count']}개")
        print(f"  - 가격 데이터: {fs['data_points_used']}일")
        
    except FileNotFoundError as e:
        print(f"\n오류: 모델 파일을 찾을 수 없습니다.")
        print(f"상세: {e}")
        print("\n다음을 확인하세요:")
        print("  1. models/ 폴더가 있는가?")
        print("  2. models/xgb_model.json 파일이 있는가?")
        print("  3. models/pca_transformer.pkl 파일이 있는가?")
        print("  4. models/feature_columns.json 파일이 있는가?")
        print("\n모델이 없다면 먼저 학습을 수행하세요:")
        print("  python train.py")
        
    except ValueError as e:
        print(f"\n오류: 데이터 검증 실패")
        print(f"상세: {e}")
        print("\n다음을 확인하세요:")
        print("  1. test_news_data.csv에 필수 컬럼이 있는가?")
        print("     - publish_date, article_embedding, price_impact_score")
        print("     - sentiment_confidence, positive_score, negative_score")
        print("  2. test_price_data.csv에 필수 컬럼이 있는가?")
        print("     - time (또는 date), close")
        print("     - ret_1d는 자동 계산되므로 없어도 됨")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\n문제 해결 팁:")
        print("  1. 뉴스 데이터에 article_embedding 컬럼이 있는지 확인")
        print("  2. 가격 데이터에 close 컬럼이 있는지 확인")
        print("  3. 모델이 제대로 학습되었는지 확인")
        print("  4. 데이터 날짜 형식이 올바른지 확인")


if __name__ == "__main__":
    main()
