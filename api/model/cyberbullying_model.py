"""
Siber zorbalık tespiti için kullanılan model sınıfı.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import json
import re
from collections import Counter
import pickle

class SelfAttention(nn.Module):
    """Self-attention mekanizması."""
    def __init__(self, hidden_dim):
        super().__init__()
        # Eğitilmiş modele uygun tek katmanlı attention
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        # Attention skorlarını hesapla
        attention_weights = self.attention(x)  # [batch_size, seq_len, hidden_dim]
        attention_weights = torch.tanh(attention_weights)  # [batch_size, seq_len, hidden_dim]
        # Attention ağırlıklarını normalize et
        attention_weights = F.softmax(attention_weights.mean(dim=-1, keepdim=True), dim=1)  # [batch_size, seq_len, 1]
        # Context vektörünü hesapla
        attended = x * attention_weights  # [batch_size, seq_len, hidden_dim]
        return attended

class LSTMAttentionModel(nn.Module):
    """LSTM ve Attention tabanlı siber zorbalık tespit modeli."""
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, output_dim=6, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Model parametreleri
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Embedding katmanı
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bi-directional LSTM katmanı
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Self-attention mekanizması
        self.attention = SelfAttention(hidden_dim * 2)  # BiLSTM olduğu için hidden_dim * 2
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout ve normalizasyon
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        # Embedding katmanı
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM katmanı
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # Layer normalization
        lstm_out = self.layer_norm1(lstm_out)
        
        # Self-attention uygula
        attended = self.attention(lstm_out)  # [batch_size, seq_len, hidden_dim*2]
        
        # Global max pooling
        pooled = F.max_pool1d(
            attended.transpose(1, 2),  # [batch_size, hidden_dim*2, seq_len]
            kernel_size=attended.size(1)  # seq_len boyutunda pooling
        ).squeeze(2)  # [batch_size, hidden_dim*2]
        
        # Fully connected katmanlar
        dense1 = self.fc1(pooled)  # [batch_size, hidden_dim]
        dense1 = self.layer_norm2(dense1)
        dense1 = F.gelu(dense1)
        dense1 = self.dropout(dense1)
        
        # Son katman
        output = self.fc2(dense1)  # [batch_size, output_dim]
        
        return output

class CyberbullyingDetector:
    """
    Siber zorbalık tespiti için ana sınıf.
    """
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Cihaz: {self.device}")
        
        # Sınıf isimleri
        self.classes = ['not_cyberbullying', 'ethnicity', 'gender', 'religion', 'age', 'other_cyberbullying']
        
        # Varsayılan vocabulary
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 17691  # Varsayılan vocabulary boyutu
        
        # Vocabulary dosyalarını kontrol et
        if model_path:
            vocab_paths = [
                os.path.join(os.path.dirname(model_path), 'vocabulary.pkl'),
                os.path.join('saved_models', 'vocabulary.pkl'),
                os.path.join(os.path.dirname(os.path.dirname(model_path)), 'data', 'vocabulary.pkl')
            ]
            
            for vocab_path in vocab_paths:
                if os.path.exists(vocab_path):
                    try:
                        print(f"Vocabulary yükleniyor: {vocab_path}")
                        with open(vocab_path, 'rb') as f:
                            vocab_data = pickle.load(f)
                            if isinstance(vocab_data, dict):
                                self.word2idx = vocab_data
                                self.idx2word = {v: k for k, v in self.word2idx.items()}
                                self.vocab_size = len(self.word2idx)
                                print(f"Vocabulary başarıyla yüklendi. Kelime sayısı: {self.vocab_size}")
                                break
                            else:
                                print("Vocabulary formatı uygun değil, varsayılan kullanılacak.")
                    except Exception as e:
                        print(f"Vocabulary yükleme hatası: {str(e)}, varsayılan kullanılacak.")
            else:
                print("Vocabulary dosyası bulunamadı, varsayılan kullanılacak.")
        
        # Model parametreleri
        self.embedding_dim = 300
        self.hidden_dim = 256
        self.output_dim = 6
        
        # Özel kelime listeleri ve ağırlıkları
        self.category_keywords = {
            'ethnicity': {
                'high': ['nigga', 'negro', 'zenci', 'go back to your country', 'immigrant'],
                'medium': ['racist', 'race', 'ethnic', 'foreigner'],
                'low': ['black', 'white', 'asian', 'mexican', 'chinese', 'indian', 'african']
            },
            'gender': {
                'high': ['bitch', 'whore', 'slut', 'orospu', 'kaltak', 'dressed like a girl'],
                'medium': ['gay', 'lesbian', 'trans', 'sexist', 'like a girl', 'like a boy'],
                'low': ['girl', 'boy', 'woman', 'man', 'female', 'male', 'gender']
            },
            'religion': {
                'high': ['kafir', 'dinsiz', 'imansız', 'stupid religion', 'religion is stupid'],
                'medium': ['muslim', 'christian', 'jew', 'müslüman', 'hristiyan', 'yahudi'],
                'low': ['hindu', 'buddhist', 'allah', 'god', 'jesus', 'religious', 'faith', 'belief']
            },
            'age': {
                'high': ['moruk', 'yaşlı', 'boomer', 'too old to', 'too young to'],
                'medium': ['genç', 'çocuk', 'teenage', 'little girl', 'little boy'],
                'low': ['old', 'young', 'kid', 'child', 'adult', 'generation', 'elderly']
            },
            'other_cyberbullying': {
                'high': ['stupid', 'idiot', 'aptal', 'gerizekalı', 'salak', 'mal', 'ugly', 'nerd', 'weak'],
                'medium': ['loser', 'kaybeden', 'ezik', 'dangalak', 'nobody likes'],
                'low': ['dumb', 'fool', 'ahmak', 'beyinsiz', 'weird']
            }
        }
        
        # Kategori ağırlıkları
        self.keyword_weights = {
            'high': 0.95,    # Çok yüksek güven
            'medium': 0.75,  # Orta güven
            'low': 0.45      # Düşük güven
        }
        
        # Model oluştur
        self.model = LSTMAttentionModel(
            self.vocab_size, 
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim
        ).to(self.device)
        
        # Eğer model dosyası varsa yükle
        if model_path and os.path.exists(model_path):
            try:
                print(f"\nModel yükleniyor: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print("Eğitim checkpoint'i tespit edildi.")
                    else:
                        state_dict = checkpoint
                        print("State dict checkpoint'i tespit edildi.")
                else:
                    raise ValueError("Checkpoint formatı tanınamadı.")
                
                # Modeli yükle
                self.model.load_state_dict(state_dict, strict=True)
                print("\nModel başarıyla yüklendi!")
                
                # Model değerlendirme moduna al
                self.model.eval()
                print("Model değerlendirme moduna alındı.")
                
            except Exception as e:
                print(f"\nKritik hata: {str(e)}")
                raise e
        else:
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
    def tokenize(self, text: str) -> List[str]:
        """Metni kelimelere ayırır."""
        # Metin temizleme
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Kelimelere ayır
        words = text.split()
        
        # Bilinmeyen kelimeleri kontrol et
        tokens = []
        for word in words:
            if word in self.word2idx:
                tokens.append(word)
            else:
                tokens.append('<UNK>')
        
        return tokens
    
    def detect_category_by_keywords(self, text: str) -> Dict[str, float]:
        """Anahtar kelimelere göre kategori tespiti yapar ve tüm skorları döndürür."""
        text = text.lower()
        
        # Her kategori için skor hesapla
        category_scores = {category: 0.0 for category in self.classes}
        total_matches = 0
        
        for category, levels in self.category_keywords.items():
            score = 0
            matches = 0
            
            # Kelime ve ifade eşleşmelerini bul
            for level, keywords in levels.items():
                for keyword in keywords:
                    if ' ' in keyword:  # Çoklu kelime kontrolü
                        if keyword in text:
                            score += self.keyword_weights[level] * 1.5  # Çoklu kelime bonus
                            matches += 1
                    else:  # Tekli kelime kontrolü
                        if f" {keyword} " in f" {text} ":
                            score += self.keyword_weights[level]
                            matches += 1
            
            # Kategori skorunu hesapla
            if matches > 0:
                # Skoru normalize et (0-1 arasında)
                normalized_score = min(score / (matches * self.keyword_weights['high']), 1.0)
                category_scores[category] = normalized_score
                total_matches += matches

        # Not cyberbullying skorunu hesapla
        if total_matches == 0:
            category_scores['not_cyberbullying'] = 0.95
        else:
            # Zorbalık içeren kategorilerin maksimum skoru
            max_bully_score = max(score for cat, score in category_scores.items() if cat != 'not_cyberbullying')
            
            # Eğer herhangi bir zorbalık kategorisi belirli bir eşiği geçerse
            if max_bully_score > 0.3:
                category_scores['not_cyberbullying'] = 0.05
            else:
                category_scores['not_cyberbullying'] = max(0.05, 1.0 - max_bully_score)

        # Skorları normalize et
        total_score = sum(category_scores.values())
        if total_score > 0:
            # İlk normalizasyon
            for category in category_scores:
                category_scores[category] = round(category_scores[category] / total_score, 4)
            
            # Minimum değer kontrolü
            min_probability = 0.01
            for category in category_scores:
                if category_scores[category] < min_probability:
                    category_scores[category] = min_probability
            
            # Son normalizasyon
            total = sum(category_scores.values())
            category_scores = {k: round(v/total, 4) for k, v in category_scores.items()}

        return category_scores
    
    def predict(self, text: str) -> Dict:
        """Metin sınıflandırması yapar."""
        try:
            # Modelin değerlendirme modunda olduğundan emin ol
            self.model.eval()
            
            # Metni tensöre çevir
            input_tensor = self.text_to_tensor(text)
            print(f"\nGiriş metni: {text}")
            print(f"Tensor boyutu: {input_tensor.shape}")
            
            # Tahmin yap
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1).squeeze()
                print(f"Ham çıktı boyutu: {output.shape}")
                print(f"Olasılık boyutu: {probabilities.shape}")
            
            # Sonuçları sözlüğe çevir
            category_scores = {
                category: float(prob) 
                for category, prob in zip(self.classes, probabilities)
            }
            
            # Debug için olasılıkları göster
            print("\nKategori olasılıkları:")
            for category, prob in category_scores.items():
                print(f"{category}: {prob:.4f}")
            
            # En yüksek olasılıklı sınıfı bul
            predicted_class = max(category_scores.items(), key=lambda x: x[1])[0]
            confidence = category_scores[predicted_class]
            
            result = {
                'predicted_class': predicted_class,
                'confidence': round(float(confidence), 4),
                'probabilities': {k: round(float(v), 4) for k, v in category_scores.items()}
            }
            
            print(f"\nTahmin sonucu: {result}")
            return result
            
        except Exception as e:
            print(f"\nTahmin sırasında hata: {str(e)}")
            raise e
    
    def text_to_tensor(self, text: str, max_len: int = 100) -> torch.Tensor:
        """Metni tensöre çevirir."""
        # Tokenize
        tokens = self.tokenize(text)
        print(f"Tokenize edilmiş metin: {tokens}")
        
        # Padding/truncating
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + ['<PAD>'] * (max_len - len(tokens))
        
        # Token'ları indekslere çevir
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # Tensöre çevir
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        print(f"Oluşturulan tensor boyutu: {tensor.shape}")
        return tensor
    
    def load_model(self, path: str) -> None:
        """Modeli yükler."""
        if os.path.exists(path):
            try:
                print(f"Model yükleniyor: {path}")
                checkpoint = torch.load(path, map_location=self.device)
                
                # Model state kontrolü
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print("Eğitim checkpoint'i tespit edildi.")
                    else:
                        state_dict = checkpoint
                        print("State dict checkpoint'i tespit edildi.")
                else:
                    raise ValueError("Checkpoint formatı tanınamadı.")
                
                # Modeli yükle (strict=False ile eksik parametreleri görmezden gel)
                self.model.load_state_dict(state_dict, strict=False)
                print("\nModel başarıyla yüklendi!")
                print(f"Toplam parametre sayısı: {sum(p.numel() for p in self.model.parameters())}")
                
            except Exception as e:
                print(f"\nKritik hata: {str(e)}")
                print("Kural tabanlı sistem kullanılacak.")
        else:
            print(f"Model dosyası bulunamadı: {path}")
            print("Kural tabanlı sistem kullanılacak.") 