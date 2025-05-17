"""
Metin ön işleme işlemleri için kullanılan modül.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import os
from collections import Counter
from typing import List, Dict, Set

class TextPreprocessor:
    def __init__(self):
        """
        TextPreprocessor sınıfının başlatıcı metodu.
        NLTK kaynaklarını indirme işlemini gerçekleştirir.
        """
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('turkish'))
        
        # Vocabulary
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 2
        
        # Vocabulary dosyasını kontrol et ve yükle
        vocab_path = os.path.join('saved_models', 'vocabulary.pkl')
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'rb') as f:
                    self.word2idx = pickle.load(f)
                    self.vocab_size = len(self.word2idx)
                print(f"Vocabulary yüklendi. Kelime sayısı: {self.vocab_size}")
            except Exception as e:
                print(f"Vocabulary yüklenemedi: {str(e)}")

    def preprocess_text(self, text):
        """
        Metni ön işleme tabi tutar.
        
        Args:
            text (str): İşlenecek ham metin
            
        Returns:
            str: İşlenmiş metin
        """
        if not isinstance(text, str):
            return ""
            
        # Küçük harfe çevirme
        text = text.lower()
        
        # Noktalama işaretlerini kaldırma
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Sayıları kaldırma
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Stopwords temizliği
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Boşlukları temizleme
        text = ' '.join(tokens)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_batch(self, texts):
        """
        Birden fazla metni toplu olarak işler.
        
        Args:
            texts (list): İşlenecek metinlerin listesi
            
        Returns:
            list: İşlenmiş metinlerin listesi
        """
        return [self.preprocess_text(text) for text in texts]

    def tokenize(self, text: str) -> List[str]:
        """Metni token'lara ayırır."""
        # Metin ön işleme
        text = self.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Stop words'leri kaldır
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens

    def build_vocabulary(self, texts: List[str], min_freq: int = 2, max_vocab_size: int = 20000) -> None:
        """Metinlerden vocabulary oluşturur."""
        # Tüm token'ları topla
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        # Token frekanslarını hesapla
        token_counts = Counter(all_tokens)
        
        # Sık kullanılan token'ları seç
        common_tokens = [
            token for token, count in token_counts.most_common(max_vocab_size)
            if count >= min_freq
        ]
        
        # Vocabulary'yi güncelle
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for token in common_tokens:
            if token not in self.word2idx:
                self.word2idx[token] = len(self.word2idx)
        
        self.vocab_size = len(self.word2idx)
        
        # Vocabulary'yi kaydet
        os.makedirs('saved_models', exist_ok=True)
        vocab_path = os.path.join('saved_models', 'vocabulary.pkl')
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.word2idx, f)
        
        print(f"Vocabulary oluşturuldu ve kaydedildi. Kelime sayısı: {self.vocab_size}")

    def text_to_sequence(self, text: str, max_len: int = 100) -> List[int]:
        """Metni indeks dizisine çevirir."""
        tokens = self.tokenize(text)
        
        # Maksimum uzunluğa göre kes veya doldur
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + ['<PAD>'] * (max_len - len(tokens))
        
        # Token'ları indekslere çevir
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        return sequence 