"""
옥수수 가격 예측 모델 학습 스크립트
- negative_model: 하락 감지 전문 (threshold=0.005, 2-class)
- positive_model: 상승 감지 전문 (threshold=-0.005, 2-class)
- tri_model: 3-way 분류 (threshold=±0.005, 3-class)

사용법:
    python train_models.py --news_path corn_all_news_with_sentiment.csv \
                          --price_path corn_future_price.csv \
                          --output_dir trained_models/
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json
import os
import argparse
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 데이터 전처리 클래스 (공통)
# ============================================================================

class CornDataPreprocessor:
    """옥수수 뉴스 및 가격 데이터 전처리"""
    
    def __init__(self, news_path, price_path):
        self.news_df = pd.read_csv(news_path)
        self.price_df = pd.read_csv(price_path)
        
        # 날짜 변환
        self.news_df['publish_date'] = pd.to_datetime(self.news_df['publish_date'])
        self.price_df['time'] = pd.to_datetime(self.price_df['time'])
        
    def prepare_data(self, lookback_days=7, future_days=1, threshold=0.005, num_classes=3):
        """
        시계열 데이터 준비
        
        Args:
            lookback_days: 과거 며칠의 뉴스를 볼 것인가
            future_days: 며칠 후의 가격을 예측할 것인가
            threshold: 가격 변동 임계값 (0.005 = 0.5%)
            num_classes: 2 (binary) or 3 (tri-class)
        """
        processed_data = []
        
        # 가격 변동 계산
        self.price_df['price_change'] = self.price_df['close'].pct_change()
        self.price_df['target'] = self.price_df['close'].shift(-future_days)
        
        # 타겟 레이블 생성
        if num_classes == 2:
            # Binary: 0=하락 or 유지, 1=상승 (or 0=상승 or 유지, 1=하락 for negative model)
            self.price_df['target_direction'] = np.where(
                self.price_df['target'] > self.price_df['close'] * (1 + threshold), 1, 0
            )
        else:  # num_classes == 3
            # Tri-class: 0=하락, 1=상승, 2=유지
            self.price_df['target_direction'] = np.where(
                self.price_df['target'] > self.price_df['close'] * (1 + threshold), 1,
                np.where(self.price_df['target'] < self.price_df['close'] * (1 - threshold), 0, 2)
            )

        for idx, price_row in self.price_df.iterrows():
            if pd.isna(price_row['target']):
                continue
                
            date = price_row['time']
            
            # 해당 날짜 이전 lookback_days 동안의 뉴스 수집
            start_date = date - timedelta(days=lookback_days)
            relevant_news = self.news_df[
                (self.news_df['publish_date'] >= start_date) & 
                (self.news_df['publish_date'] < date) &
                (self.news_df['filter_status'] == 'T')
            ].copy()
            
            if len(relevant_news) == 0:
                continue
            
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
                continue
            
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
                'positive_count': (relevant_news['sentiment'] == 'positive').sum(),
                'negative_count': (relevant_news['sentiment'] == 'negative').sum(),
                'neutral_count': (relevant_news['sentiment'] == 'neutral').sum(),
            }
            
            # 가격 특성
            price_features = {
                'open': price_row['open'],
                'high': price_row['high'],
                'low': price_row['low'],
                'close': price_row['close'],
                'volume': price_row['Volume'],
                'ema': price_row['EMA'],
                'volatility': (price_row['high'] - price_row['low']) / price_row['close']
            }
            
            processed_data.append({
                'date': date,
                'article_embedding': avg_article_emb,
                'entity_embedding': avg_entity_emb,
                'triple_embedding': avg_triple_emb,
                'sentiment_features': sentiment_features,
                'price_features': price_features,
                'target': price_row['target'],
                'target_direction': price_row['target_direction'],
                'news_articles': relevant_news[[
                    'id', 'title', 'sentiment', 'price_impact_score',
                    'positive_score', 'negative_score', 'neutral_score', 'named_entities'
                ]].to_dict('records')
            })
        
        return processed_data
    
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
# PyTorch Dataset (공통)
# ============================================================================

class CornNewsDataset(Dataset):
    """뉴스 기반 가격 예측 데이터셋"""
    
    def __init__(self, processed_data, feature_scaler=None):
        self.data = processed_data
        self.feature_scaler = feature_scaler
        self.prepare_features()
        
    def prepare_features(self):
        """모든 특성을 하나의 벡터로 결합"""
        self.X = []
        self.y = []
        
        for item in self.data:
            # 임베딩 결합 (512 + 1024 + 1024 = 2560차원)
            embedding = np.concatenate([
                item['article_embedding'],
                item['entity_embedding'],
                item['triple_embedding']
            ])
            
            # 감성 특성 (9차원)
            sentiment_vec = np.array([
                item['sentiment_features']['avg_price_impact'],
                item['sentiment_features']['avg_positive'],
                item['sentiment_features']['avg_negative'],
                item['sentiment_features']['avg_neutral'],
                item['sentiment_features']['sentiment_std'] if not np.isnan(item['sentiment_features']['sentiment_std']) else 0,
                item['sentiment_features']['news_count'],
                item['sentiment_features']['positive_count'],
                item['sentiment_features']['negative_count'],
                item['sentiment_features']['neutral_count'],
            ])
            
            # 가격 특성 (7차원)
            price_vec = np.array([
                item['price_features']['open'],
                item['price_features']['high'],
                item['price_features']['low'],
                item['price_features']['close'],
                item['price_features']['volume'],
                item['price_features']['ema'],
                item['price_features']['volatility']
            ])
            
            # 전체 특성 결합
            features = np.concatenate([embedding, sentiment_vec, price_vec])
            
            self.X.append(features)
            self.y.append(item['target_direction'])
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # 표준화
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            self.X = self.feature_scaler.fit_transform(self.X)
        else:
            self.X = self.feature_scaler.transform(self.X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.X[idx]),
            'target': torch.LongTensor([self.y[idx]])[0],
            'metadata': self.data[idx]
        }


def custom_collate_fn(batch):
    """배치 데이터 정리"""
    features = torch.stack([item['features'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    
    return {
        'features': features,
        'targets': targets,
        'metadata': metadata
    }


# ============================================================================
# Transformer 모델 (공통)
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
# 학습 클래스
# ============================================================================

class ModelTrainer:
    """모델 학습 및 평가 관리"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, class_weights=None):
        self.model = model.to(device)
        self.device = device
        
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5, 
            factor=0.5
        )
        
    def train_epoch(self, dataloader):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def evaluate(self, dataloader):
        """모델 평가"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                logits, _ = self.model(features)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'predictions': all_predictions,
            'targets': all_targets
        }


# ============================================================================
# 메인 학습 함수
# ============================================================================

def train_single_model(model_name, preprocessor, num_classes, threshold, output_dir, 
                       num_epochs=50, device='cpu'):
    """
    단일 모델 학습
    
    Args:
        model_name: 모델 이름 (negative_model, positive_model, tri_model)
        preprocessor: CornDataPreprocessor 인스턴스
        num_classes: 2 or 3
        threshold: 가격 변동 임계값
        output_dir: 모델 저장 디렉토리
        num_epochs: 학습 에폭 수
        device: 학습 디바이스
    
    Returns:
        dict: 학습 결과 (best_accuracy, model_path)
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name} (num_classes={num_classes}, threshold={threshold})")
    print(f"{'='*80}")
    
    # 데이터 준비
    print("[1/5] 데이터 전처리 중...")
    processed_data = preprocessor.prepare_data(
        lookback_days=7, 
        future_days=1, 
        threshold=threshold,
        num_classes=num_classes
    )
    print(f"총 {len(processed_data)}개 샘플 생성")
    
    # 시간 순서로 정렬 및 분할
    print("[2/5] 학습/검증 데이터 분할 중...")
    processed_data = sorted(processed_data, key=lambda x: x['date'])
    split_idx = int(len(processed_data) * 0.8)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    train_dataset = CornNewsDataset(train_data)
    test_dataset = CornNewsDataset(test_data, feature_scaler=train_dataset.feature_scaler)
    
    # 클래스 분포 확인 및 가중치 계산
    unique, counts = np.unique(train_dataset.y, return_counts=True)
    print(f"\n학습 데이터 클래스 분포:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt}개")
    
    total = len(train_dataset.y)
    class_weights = [total / (len(unique) * count) for count in counts]
    print(f"계산된 클래스 가중치: {[f'{w:.2f}' for w in class_weights]}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    
    # 모델 초기화
    print(f"\n[3/5] 모델 초기화 중... (device={device})")
    input_dim = train_dataset.X.shape[1]
    model = CornPricePredictor(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
    trainer = ModelTrainer(model, device=device, class_weights=class_weights)
    
    # 모델 학습
    print(f"\n[4/5] 모델 학습 중... ({num_epochs} epochs)")
    best_acc = 0
    best_model_path = os.path.join(output_dir, f'{model_name}.pt')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        test_results = trainer.evaluate(test_loader)
        
        trainer.scheduler.step(test_results['loss'])
        
        if test_results['accuracy'] > best_acc:
            best_acc = test_results['accuracy']
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Test Loss: {test_results['loss']:.4f}, Test Acc: {test_results['accuracy']:.4f}")
    
    print(f"\n[5/5] 학습 완료! 최고 정확도: {best_acc:.4f}")
    print(f"모델 저장 위치: {best_model_path}")
    
    # Scaler 저장 (첫 번째 모델에서만)
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        with open(scaler_path, 'wb') as f:
            pickle.dump(train_dataset.feature_scaler, f)
        print(f"StandardScaler 저장: {scaler_path}")
    
    # 혼동 행렬 출력
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        
        model.load_state_dict(torch.load(best_model_path))
        final_results = trainer.evaluate(test_loader)
        
        cm = confusion_matrix(final_results['targets'], final_results['predictions'])
        print(f"\nConfusion Matrix:")
        print(cm)
        
        if num_classes == 2:
            target_names = ['class_0', 'class_1']
        else:
            target_names = ['down', 'up', 'stable']
        
        print(f"\nClassification Report:")
        print(classification_report(final_results['targets'], final_results['predictions'], 
                                   target_names=target_names))
    except Exception as e:
        print(f"혼동 행렬 생성 실패: {e}")
    
    return {
        'model_name': model_name,
        'best_accuracy': best_acc,
        'model_path': best_model_path,
        'num_classes': num_classes,
        'threshold': threshold,
        'input_dim': input_dim
    }


def main():
    """전체 학습 파이프라인 실행"""
    parser = argparse.ArgumentParser(description='옥수수 가격 예측 모델 학습')
    parser.add_argument('--news_path', type=str, default='corn_all_news_with_sentiment.csv',
                       help='뉴스 데이터 CSV 파일 경로')
    parser.add_argument('--price_path', type=str, default='corn_future_price.csv',
                       help='가격 데이터 CSV 파일 경로')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                       help='모델 저장 디렉토리')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='학습 에폭 수')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    
    # 데이터 로드
    print(f"\n데이터 로드 중...")
    print(f"  뉴스 데이터: {args.news_path}")
    print(f"  가격 데이터: {args.price_path}")
    preprocessor = CornDataPreprocessor(
        news_path=args.news_path,
        price_path=args.price_path
    )
    
    # 학습 결과 저장
    results = {}
    
    # 1. Negative Model (하락 감지)
    # threshold=0.005 → target > close * 1.005이면 class 1 (상승), 아니면 class 0 (하락 or 유지)
    results['negative_model'] = train_single_model(
        model_name='negative_model',
        preprocessor=preprocessor,
        num_classes=2,
        threshold=0.005,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # 2. Positive Model (상승 감지)
    # threshold=-0.005 → target < close * 0.995이면 class 1 (하락), 아니면 class 0 (상승 or 유지)
    # 주의: 이 모델은 "하락"을 감지하는 것이므로, 나중에 예측 시 반대로 해석 필요
    results['positive_model'] = train_single_model(
        model_name='positive_model',
        preprocessor=preprocessor,
        num_classes=2,
        threshold=-0.005,  # 음수 threshold
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # 3. Tri Model (3-way 분류)
    results['tri_model'] = train_single_model(
        model_name='tri_model',
        preprocessor=preprocessor,
        num_classes=3,
        threshold=0.005,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # 최종 결과 요약
    print(f"\n{'='*80}")
    print("학습 완료 요약")
    print(f"{'='*80}")
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  - 정확도: {result['best_accuracy']:.4f}")
        print(f"  - 클래스 수: {result['num_classes']}")
        print(f"  - Threshold: {result['threshold']}")
        print(f"  - 저장 경로: {result['model_path']}")
    
    # 메타데이터 저장
    metadata_path = os.path.join(args.output_dir, 'training_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'news_path': args.news_path,
            'price_path': args.price_path,
            'num_epochs': args.num_epochs,
            'device': str(device),
            'models': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n메타데이터 저장: {metadata_path}")
    print(f"\n모든 모델 학습 완료!")
    
    return results


if __name__ == "__main__":
    results = main()
