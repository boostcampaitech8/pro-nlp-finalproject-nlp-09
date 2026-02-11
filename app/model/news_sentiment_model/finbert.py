"""
원자재(옥수수) 뉴스 감성 분석 모듈
FinBERT를 사용하여 긍정/부정/중립 감성 분석 수행

이 모듈은 순수 함수/클래스만 제공하며, 실행 스크립트는 별도로 존재합니다.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np


class CommoditySentimentAnalyzer:
    """
    원자재(옥수수) 뉴스 감성 분석기
    FinBERT를 사용하여 긍정/부정/중립 감성 분석 수행
    """
    
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        감성 분석기 초기화
        
        Args:
            model_name: 사용할 모델 (기본값: FinBERT)
        """
        print(f"Loading model: {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # FinBERT 레이블 매핑
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        print(f"Model loaded successfully on {self.device}")
    
    def analyze_text(self, text, max_length=512):
        """
        단일 텍스트에 대한 감성 분석
        
        Args:
            text: 분석할 텍스트
            max_length: 최대 토큰 길이
            
        Returns:
            dict: {
                'sentiment': str,           # positive/negative/neutral
                'confidence': float,        # 예측 신뢰도
                'positive_score': float,    # 긍정 점수
                'negative_score': float,    # 부정 점수
                'neutral_score': float      # 중립 점수
            }
        """
        if pd.isna(text) or text.strip() == '':
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 0.0
            }
        
        # 토큰화 및 인코딩
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 결과 파싱
        scores = predictions[0].cpu().numpy()
        predicted_class = np.argmax(scores)
        
        result = {
            'sentiment': self.label_map[predicted_class],
            'confidence': float(scores[predicted_class]),
            'positive_score': float(scores[0]),
            'negative_score': float(scores[1]),
            'neutral_score': float(scores[2])
        }
        
        return result
    
    def analyze_dataframe(self, df, text_column='combined_text', batch_size=16, show_progress=True):
        """
        데이터프레임 전체에 대한 감성 분석
        
        Args:
            df: 분석할 데이터프레임
            text_column: 분석할 텍스트 컬럼명
            batch_size: 배치 크기 (메모리에 따라 조정, 현재는 미사용)
            show_progress: 진행상황 표시 여부
            
        Returns:
            DataFrame: 감성 분석 결과가 추가된 데이터프레임
                추가 컬럼:
                - sentiment: positive/negative/neutral
                - sentiment_confidence: 예측 신뢰도
                - positive_score: 긍정 점수
                - negative_score: 부정 점수
                - neutral_score: 중립 점수
                - price_impact_score: positive_score - negative_score
        """
        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' 컬럼이 데이터프레임에 없습니다.")
        
        results = []
        
        if show_progress:
            print(f"Analyzing {len(df)} articles...")
            iterator = tqdm(df[text_column], desc="Sentiment Analysis")
        else:
            iterator = df[text_column]
        
        for text in iterator:
            result = self.analyze_text(text)
            results.append(result)
        
        # 결과를 데이터프레임에 추가
        df_result = df.copy()
        df_result['sentiment'] = [r['sentiment'] for r in results]
        df_result['sentiment_confidence'] = [r['confidence'] for r in results]
        df_result['positive_score'] = [r['positive_score'] for r in results]
        df_result['negative_score'] = [r['negative_score'] for r in results]
        df_result['neutral_score'] = [r['neutral_score'] for r in results]
        
        # 가격 영향 점수 계산 (positive - negative)
        df_result['price_impact_score'] = (
            df_result['positive_score'] - df_result['negative_score']
        )
        
        return df_result


def prepare_text_for_analysis(df, title_col='title', description_col='description', 
                               all_text_col='all_text', output_col='combined_text'):
    """
    분석을 위한 텍스트 결합 함수
    title, description, all_text를 적절히 결합
    
    Args:
        df: 원본 데이터프레임
        title_col: 제목 컬럼명 (기본: 'title')
        description_col: 설명 컬럼명 (기본: 'description')
        all_text_col: 전체 텍스트 컬럼명 (기본: 'all_text')
        output_col: 출력 컬럼명 (기본: 'combined_text')
        
    Returns:
        DataFrame: combined_text 컬럼이 추가된 데이터프레임
    """
    df_prepared = df.copy()
    
    # title + description 결합 (all_text는 너무 길 수 있으므로 선택적 사용)
    if title_col in df.columns and description_col in df.columns:
        df_prepared[output_col] = (
            df_prepared[title_col].fillna('') + ' ' + 
            df_prepared[description_col].fillna('')
        ).str.strip()
    elif title_col in df.columns:
        df_prepared[output_col] = df_prepared[title_col].fillna('').str.strip()
    elif description_col in df.columns:
        df_prepared[output_col] = df_prepared[description_col].fillna('').str.strip()
    else:
        raise ValueError(f"'{title_col}' 또는 '{description_col}' 컬럼이 필요합니다.")
    
    # 텍스트가 너무 짧으면 all_text의 앞부분 추가
    if all_text_col in df.columns:
        for idx, row in df_prepared.iterrows():
            if len(row[output_col]) < 50 and pd.notna(row.get(all_text_col)):
                df_prepared.at[idx, output_col] = (
                    row[output_col] + ' ' + str(row[all_text_col])[:500]
                ).strip()
    
    return df_prepared


def get_sentiment_summary(df):
    """
    감성 분석 결과 요약 통계 반환
    
    Args:
        df: 감성 분석이 완료된 데이터프레임
        
    Returns:
        dict: 요약 통계
            - sentiment_distribution: 감성별 분포
            - avg_confidence: 평균 신뢰도
            - avg_price_impact: 평균 가격 영향 점수
            - total_count: 전체 기사 수
    """
    if 'sentiment' not in df.columns:
        raise ValueError("감성 분석이 완료된 데이터프레임이 필요합니다.")
    
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    total = len(df)
    
    summary = {
        'sentiment_distribution': {
            sentiment: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for sentiment, count in sentiment_counts.items()
        },
        'avg_confidence': float(df['sentiment_confidence'].mean()),
        'avg_price_impact': float(df['price_impact_score'].mean()),
        'total_count': total
    }
    
    return summary


def get_daily_sentiment_trend(df, date_column='publish_date'):
    """
    일별 감성 트렌드 분석
    
    Args:
        df: 감성 분석이 완료된 데이터프레임
        date_column: 날짜 컬럼명
        
    Returns:
        DataFrame: 일별 감성 트렌드
            - date: 날짜
            - price_impact_score: 평균 가격 영향 점수
            - sentiment_confidence: 평균 신뢰도
            - article_count: 기사 수
    """
    if date_column not in df.columns:
        raise ValueError(f"'{date_column}' 컬럼이 데이터프레임에 없습니다.")
    
    df_trend = df.copy()
    df_trend[date_column] = pd.to_datetime(df_trend[date_column])
    df_trend['date'] = df_trend[date_column].dt.date
    
    daily_sentiment = df_trend.groupby('date').agg({
        'price_impact_score': 'mean',
        'sentiment_confidence': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'article_count'}).reset_index()
    
    return daily_sentiment


# ============================================
# 편의 함수
# ============================================

def analyze_news_sentiment(df, text_column='combined_text', model_name="ProsusAI/finbert", 
                           show_progress=True):
    """
    뉴스 데이터프레임에 대한 감성 분석 수행 (원스텝 함수)
    
    Args:
        df: 분석할 데이터프레임
        text_column: 분석할 텍스트 컬럼명
        model_name: 사용할 모델명
        show_progress: 진행상황 표시 여부
        
    Returns:
        DataFrame: 감성 분석 결과가 추가된 데이터프레임
    """
    analyzer = CommoditySentimentAnalyzer(model_name=model_name)
    df_result = analyzer.analyze_dataframe(df, text_column=text_column, show_progress=show_progress)
    return df_result
