from typing import List, Dict, Any


class SentimentAnalyzer:
    """FinBERT 감성분석 (테스트용 틀)"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """텍스트 감성분석 (한국어/영어 지원)"""
        # 테스트용 더미 구현
        text_lower = text.lower()
        
        # 긍정 키워드 (한국어 + 영어)
        positive_keywords = [
            # 일반 긍정
            "상승", "호실적", "긍정", "긍정적", "증가", "성장", "개선", "회복",
            # 금융/시장 긍정
            "rise", "increase", "positive", "growth", "gain", "gains", "up", 
            "surge", "boost", "profit", "success", "improve", "better",
            # 가격 상승 관련
            "higher", "rising", "climb", "advance", "rally", "soar",
            # 시장 전망 긍정
            "projected", "expected", "forecast", "outlook", "cagr", "expand",
            "set to grow", "will grow", "is growing", "growing",
            # 정책/행동 긍정
            "approved", "ending", "end", "support", "anticipate", "ease",
            "reduction", "reductions", "cut", "cuts", "lower prices",
            # 농업/에너지 긍정
            "harvest", "yield", "production", "demand", "strong demand"
        ]
        
        # 부정 키워드 (한국어 + 영어)
        negative_keywords = [
            # 일반 부정
            "하락", "침체", "부정", "부정적", "감소", "위기", "위험", "불안",
            # 금융/시장 부정
            "fall", "decrease", "negative", "decline", "drop", "down", "lower",
            "loss", "losses", "crisis", "fail", "failure", "recession", "worse",
            # 가격 하락 관련
            "plunge", "crash", "collapse", "sink", "tumble", "slump",
            # 시장 부정
            "struggle", "stalled", "squeeze", "stress", "pressure", "dangerous",
            "threat", "risk", "risks", "uncertainty", "volatility",
            # 공급 과잉/부족
            "glut", "surplus", "oversupply", "shortage", "deficit",
            # 정책/상황 부정
            "shutdown", "shut down", "disaster", "reckoning", "populism",
            # 농업/에너지 부정
            "spreading", "outbreak", "killing", "decline in production",
            "sharp decline", "sharp rise in costs"
        ]
        
        if any(word in text_lower for word in positive_keywords):
            sentiment = "positive"
            scores = {"positive": 0.7, "negative": 0.2, "neutral": 0.1}
        elif any(word in text_lower for word in negative_keywords):
            sentiment = "negative"
            scores = {"positive": 0.2, "negative": 0.7, "neutral": 0.1}
        else:
            sentiment = "neutral"
            scores = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        
        return {
            "text": text,
            "sentiment": sentiment,
            "scores": scores
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """배치 감성분석"""
        return [self.analyze(text) for text in texts]
3