"""
옥수수 가격 예측 앙상블 모듈
- negative_model, positive_model, tri_model을 통합하여 최종 예측 수행
- 모델 간 합의를 통한 신뢰도 높은 예측
- 상세한 예측 근거 및 증거 제공

사용법:
    from ensemble_predictor import EnsemblePredictor
    
    predictor = EnsemblePredictor(model_dir='trained_models')
    result = predictor.predict(preprocessed_data)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 모델 아키텍처 (train_models.py와 동일)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention 메커니즘"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(attention_output)
        
        return output, attention_weights


class CornPricePredictor(nn.Module):
    """Transformer + Attention 기반 가격 예측 모델"""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        
        attn_output, attention_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        x = x.squeeze(1)
        logits = self.classifier(x)
        
        return logits, attention_weights


# ============================================================================
# 앙상블 전략 함수들
# ============================================================================

def voting_strategy(negative_result, positive_result, tri_result):
    """
    3개 모델의 예측을 통합하는 투표 전략
    
    Args:
        negative_result: dict {prediction: int (0 or 1), confidence: float, probs: array}
        positive_result: dict {prediction: int (0 or 1), confidence: float, probs: array}
        tri_result: dict {prediction: int (0, 1, or 2), confidence: float, probs: array}
    
    Returns:
        dict: {
            direction: str (하락/상승/유지),
            confidence: float,
            probabilities: dict,
            agreement_level: str (high/medium/low),
            reasoning: str
        }
    """
    
    # Negative Model 해석
    # class 0 = 하락 or 유지 (not_up)
    # class 1 = 상승 (up)
    neg_pred = negative_result['prediction']
    neg_conf = negative_result['confidence']
    neg_says_up = (neg_pred == 1)
    
    # Positive Model 해석 (주의: 반대로 해석!)
    # class 0 = 상승 or 유지 (not_down)
    # class 1 = 하락 (down)
    pos_pred = positive_result['prediction']
    pos_conf = positive_result['confidence']
    pos_says_down = (pos_pred == 1)
    
    # Tri Model 해석
    # class 0 = 하락, class 1 = 상승, class 2 = 유지
    tri_pred = tri_result['prediction']
    tri_conf = tri_result['confidence']
    tri_probs = tri_result['probs']
    
    tri_says_down = (tri_pred == 0)
    tri_says_up = (tri_pred == 1)
    tri_says_stable = (tri_pred == 2)
    
    # ========== 투표 로직 ==========
    
    # Case 1: 강한 합의 (모든 모델이 일치)
    if neg_says_up and (not pos_says_down) and tri_says_up:
        # Negative: 상승, Positive: not_down (상승 or 유지), Tri: 상승
        direction = "상승"
        confidence = np.mean([neg_conf, pos_conf, tri_conf])
        agreement_level = "high"
        reasoning = "3개 모델 모두 상승 예측 (강한 합의)"
        
    elif (not neg_says_up) and pos_says_down and tri_says_down:
        # Negative: not_up (하락 or 유지), Positive: 하락, Tri: 하락
        direction = "하락"
        confidence = np.mean([neg_conf, pos_conf, tri_conf])
        agreement_level = "high"
        reasoning = "3개 모델 모두 하락 예측 (강한 합의)"
        
    # Case 2: 중간 합의 (bi 모델들이 중립, tri가 결정)
    elif (not neg_says_up) and (not pos_says_down):
        # Negative: not_up, Positive: not_down → 중립 신호
        if tri_says_up:
            direction = "상승"
            confidence = tri_conf * 0.8  # 신뢰도 약간 낮춤
            agreement_level = "medium"
            reasoning = "Bi 모델들 중립, Tri 모델 상승 예측"
        elif tri_says_down:
            direction = "하락"
            confidence = tri_conf * 0.8
            agreement_level = "medium"
            reasoning = "Bi 모델들 중립, Tri 모델 하락 예측"
        else:  # tri_says_stable
            direction = "유지"
            confidence = tri_conf * 0.9
            agreement_level = "medium"
            reasoning = "모든 모델이 중립적 신호"
    
    # Case 3: Bi 모델 중 하나만 강한 신호, Tri와 일치
    elif neg_says_up and tri_says_up:
        # Negative: 상승, Tri: 상승 (Positive는 어떤 값이든)
        direction = "상승"
        if not pos_says_down:
            # Positive도 not_down이면 완전 합의
            confidence = np.mean([neg_conf, pos_conf, tri_conf])
            agreement_level = "high"
            reasoning = "Negative와 Tri 모델 상승 예측, Positive 중립"
        else:
            # Positive는 down 예측 (충돌)
            confidence = np.mean([neg_conf, tri_conf]) * 0.7
            agreement_level = "medium"
            reasoning = "Negative와 Tri 모델 상승 예측하나 Positive는 하락 예측 (부분 충돌)"
            
    elif pos_says_down and tri_says_down:
        # Positive: 하락, Tri: 하락 (Negative는 어떤 값이든)
        direction = "하락"
        if not neg_says_up:
            confidence = np.mean([neg_conf, pos_conf, tri_conf])
            agreement_level = "high"
            reasoning = "Positive와 Tri 모델 하락 예측, Negative 중립"
        else:
            confidence = np.mean([pos_conf, tri_conf]) * 0.7
            agreement_level = "medium"
            reasoning = "Positive와 Tri 모델 하락 예측하나 Negative는 상승 예측 (부분 충돌)"
    
    # Case 4: 강한 충돌 (bi 모델들이 정반대 신호)
    elif neg_says_up and pos_says_down:
        # Negative: 상승, Positive: 하락 → 충돌
        if tri_says_up:
            direction = "상승"
            confidence = np.mean([neg_conf, tri_conf]) * 0.6
            agreement_level = "low"
            reasoning = "Negative와 Tri는 상승, Positive는 하락 예측 (모델 간 충돌)"
        elif tri_says_down:
            direction = "하락"
            confidence = np.mean([pos_conf, tri_conf]) * 0.6
            agreement_level = "low"
            reasoning = "Positive와 Tri는 하락, Negative는 상승 예측 (모델 간 충돌)"
        else:  # tri_says_stable
            direction = "유지"
            confidence = 0.5  # 매우 낮은 신뢰도
            agreement_level = "low"
            reasoning = "모델 간 강한 충돌 (Negative 상승, Positive 하락, Tri 유지)"
    
    # Case 5: 기타 (Tri 모델 우선)
    else:
        if tri_says_up:
            direction = "상승"
        elif tri_says_down:
            direction = "하락"
        else:
            direction = "유지"
        
        confidence = tri_conf * 0.7
        agreement_level = "low"
        reasoning = "모델 간 불일치, Tri 모델 예측 우선 적용"
    
    # 확률 계산 (tri_model의 확률을 기반으로, 다른 모델 신호 반영)
    probabilities = {
        "up": float(tri_probs[1]),
        "down": float(tri_probs[0]),
        "stable": float(tri_probs[2])
    }
    
    # Bi 모델의 강한 신호 반영
    if neg_says_up and neg_conf > 0.8:
        probabilities["up"] = min(probabilities["up"] * 1.2, 0.95)
    if pos_says_down and pos_conf > 0.8:
        probabilities["down"] = min(probabilities["down"] * 1.2, 0.95)
    
    # 정규화
    total = sum(probabilities.values())
    probabilities = {k: v/total for k, v in probabilities.items()}
    
    return {
        'direction': direction,
        'confidence': float(confidence),
        'probabilities': probabilities,
        'agreement_level': agreement_level,
        'reasoning': reasoning
    }


def calculate_ensemble_confidence(model_results, agreement_level):
    """
    앙상블 신뢰도 계산 (추가 보정)
    
    Args:
        model_results: dict of model predictions
        agreement_level: str (high/medium/low)
    
    Returns:
        float: 보정된 신뢰도
    """
    confidences = [
        model_results['negative_model']['confidence'],
        model_results['positive_model']['confidence'],
        model_results['tri_model']['confidence']
    ]
    
    avg_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    
    # 합의 수준에 따라 가중치 적용
    if agreement_level == "high":
        weight = 1.0
    elif agreement_level == "medium":
        weight = 0.85
    else:  # low
        weight = 0.7
    
    # 표준편차가 크면 (모델 간 신뢰도 차이가 크면) 신뢰도 감소
    if std_conf > 0.2:
        weight *= 0.9
    
    return min(avg_conf * weight, 0.95)


# ============================================================================
# 앙상블 예측기 클래스
# ============================================================================

class EnsemblePredictor:
    """3개 모델을 통합한 앙상블 예측기"""
    
    def __init__(self, model_dir='trained_models', device=None):
        """
        앙상블 예측기 초기화
        
        Args:
            model_dir: 학습된 모델이 저장된 디렉토리
            device: 추론 디바이스 (None이면 자동 선택)
        """
        self.model_dir = model_dir
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"앙상블 예측기 초기화 중... (device={self.device})")
        
        # 메타데이터 로드
        metadata_path = os.path.join(model_dir, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"메타데이터 로드 완료: {metadata_path}")
        else:
            print("경고: training_metadata.json을 찾을 수 없습니다.")
            self.metadata = {}
        
        # StandardScaler 로드
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"StandardScaler 로드 완료")
        
        # 모델 로드
        self.models = {}
        self._load_models()
        
        print("앙상블 예측기 초기화 완료!\n")
    
    def _load_models(self):
        """3개 모델 로드"""
        model_configs = {
            'negative_model': {'num_classes': 2, 'name': 'Negative Model (하락 감지)'},
            'positive_model': {'num_classes': 2, 'name': 'Positive Model (상승 감지)'},
            'tri_model': {'num_classes': 3, 'name': 'Tri Model (3-way 분류)'}
        }
        
        # Input dimension (메타데이터에서 가져오거나 기본값 사용)
        if 'models' in self.metadata and 'tri_model' in self.metadata['models']:
            input_dim = self.metadata['models']['tri_model']['input_dim']
        else:
            input_dim = 2569  # 기본값: 512(article) + 1024(entity) + 1024(triple) + 9(sentiment) (price 제외)
        
        for model_key, config in model_configs.items():
            model_path = os.path.join(self.model_dir, f'{model_key}.pt')
            
            print(f"Loading {config['name']}...")
            model = CornPricePredictor(
                input_dim=input_dim,
                hidden_dim=256,
                num_classes=config['num_classes']
            )
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models[model_key] = model
            print(f"  ✓ {model_key}.pt 로드 완료")
    
    def predict_single(self, features):
        """
        단일 샘플에 대한 예측
        
        Args:
            features: numpy array (이미 전처리된 특성 벡터)
        
        Returns:
            dict: 앙상블 예측 결과
        """
        # Features를 tensor로 변환 및 정규화
        features = self.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # 각 모델 예측
        model_results = {}
        
        with torch.no_grad():
            # Negative Model
            logits, _ = self.models['negative_model'](features_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
            
            model_results['negative_model'] = {
                'prediction': pred,
                'confidence': conf,
                'probs': probs[0].cpu().numpy()
            }
            
            # Positive Model
            logits, _ = self.models['positive_model'](features_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
            
            model_results['positive_model'] = {
                'prediction': pred,
                'confidence': conf,
                'probs': probs[0].cpu().numpy()
            }
            
            # Tri Model
            logits, _ = self.models['tri_model'](features_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
            
            model_results['tri_model'] = {
                'prediction': pred,
                'confidence': conf,
                'probs': probs[0].cpu().numpy()
            }
        
        # 앙상블 전략 적용
        ensemble_result = voting_strategy(
            model_results['negative_model'],
            model_results['positive_model'],
            model_results['tri_model']
        )
        
        # 신뢰도 재계산
        final_confidence = calculate_ensemble_confidence(
            model_results,
            ensemble_result['agreement_level']
        )
        ensemble_result['confidence'] = final_confidence
        
        # 모델별 상세 결과 추가
        ensemble_result['model_details'] = {
            'negative_model': {
                'prediction': 'up' if model_results['negative_model']['prediction'] == 1 else 'not_up',
                'confidence': round(model_results['negative_model']['confidence'], 3),
                'probabilities': {
                    'not_up': round(float(model_results['negative_model']['probs'][0]), 3),
                    'up': round(float(model_results['negative_model']['probs'][1]), 3)
                }
            },
            'positive_model': {
                'prediction': 'down' if model_results['positive_model']['prediction'] == 1 else 'not_down',
                'confidence': round(model_results['positive_model']['confidence'], 3),
                'probabilities': {
                    'not_down': round(float(model_results['positive_model']['probs'][0]), 3),
                    'down': round(float(model_results['positive_model']['probs'][1]), 3)
                }
            },
            'tri_model': {
                'prediction': ['down', 'up', 'stable'][model_results['tri_model']['prediction']],
                'confidence': round(model_results['tri_model']['confidence'], 3),
                'probabilities': {
                    'down': round(float(model_results['tri_model']['probs'][0]), 3),
                    'up': round(float(model_results['tri_model']['probs'][1]), 3),
                    'stable': round(float(model_results['tri_model']['probs'][2]), 3)
                }
            }
        }
        
        return ensemble_result
    
    def predict_batch(self, features_list):
        """
        여러 샘플에 대한 배치 예측
        
        Args:
            features_list: list of numpy arrays
        
        Returns:
            list of dict: 예측 결과 리스트
        """
        results = []
        for features in features_list:
            result = self.predict_single(features)
            results.append(result)
        return results
    
    def generate_prediction_report(self, prediction_result, metadata):
        """
        예측 결과를 사람이 읽기 쉬운 보고서로 변환
        
        Args:
            prediction_result: predict_single() 결과
            metadata: 추가 메타데이터 (날짜, 뉴스 정보 등)
        
        Returns:
            dict: 상세 보고서
        """
        report = {
            'prediction': {
                'direction': prediction_result['direction'],
                'confidence': round(prediction_result['confidence'], 2),
                'agreement_level': prediction_result['agreement_level'],
                'probabilities': {
                    k: round(v, 2) for k, v in prediction_result['probabilities'].items()
                }
            },
            'model_consensus': {
                'reasoning': prediction_result['reasoning'],
                'details': prediction_result['model_details']
            },
            'metadata': metadata
        }
        
        return report


# ============================================================================
# 편의 함수
# ============================================================================

def load_ensemble_predictor(model_dir='trained_models', device=None):
    """
    앙상블 예측기 로드 (편의 함수)
    
    Args:
        model_dir: 모델 디렉토리
        device: 추론 디바이스
    
    Returns:
        EnsemblePredictor 인스턴스
    """
    return EnsemblePredictor(model_dir=model_dir, device=device)


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == "__main__":
    """
    테스트 실행 예시
    실제 사용 시에는 daily_prediction_pipeline.py에서 호출
    """
    print("앙상블 예측기 테스트\n")
    
    # 예측기 로드
    predictor = EnsemblePredictor(model_dir='trained_models')
    
    # 더미 데이터 생성 (실제로는 전처리된 데이터 사용)
    print("더미 데이터로 예측 테스트...")
    dummy_features = np.random.randn(2569)  # 512 + 1024 + 1024 + 9 (price 제외)
    
    # 예측
    result = predictor.predict_single(dummy_features)
    
    # 결과 출력
    print("\n" + "="*80)
    print("예측 결과:")
    print("="*80)
    print(f"방향: {result['direction']}")
    print(f"신뢰도: {result['confidence']:.2%}")
    print(f"합의 수준: {result['agreement_level']}")
    print(f"\n확률:")
    for k, v in result['probabilities'].items():
        print(f"  {k}: {v:.2%}")
    print(f"\n근거: {result['reasoning']}")
    
    print("\n" + "="*80)
    print("모델별 상세 결과:")
    print("="*80)
    for model_name, details in result['model_details'].items():
        print(f"\n{model_name}:")
        print(f"  예측: {details['prediction']}")
        print(f"  신뢰도: {details['confidence']:.2%}")
        print(f"  확률: {details['probabilities']}")
    
    print("\n테스트 완료!")
