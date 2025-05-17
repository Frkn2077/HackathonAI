"""
Veri ön işleme işlemleri için modül.
"""
import re
import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from . import config

# NLTK kaynaklarını indir
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    print("NLTK kaynakları zaten indirilmiş.")

def preprocess_label(label):
    """Etiketleri standart formata dönüştür."""
    # Küçük harfe çevir
    label = str(label).lower()
    # Boşlukları alt çizgi ile değiştir
    label = label.replace(' ', '_')
    return label

class Vocabulary:
    """Metin verisi için kelime dağarcığı sınıfı."""
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        """Kelime dağarcığını oluştur."""
        frequencies = Counter()
        idx = 2
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def tokenize(self, text):
        """Metni kelimelere ayır."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def numericalize(self, text):
        """Metni sayısal forma dönüştür."""
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class CyberbullyingDataset(Dataset):
    """PyTorch veri seti sınıfı."""
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        # Metni sayısallaştır ve padding uygula
        numericalized_text = self.vocab.numericalize(text)
        if len(numericalized_text) > self.max_length:
            numericalized_text = numericalized_text[:self.max_length]
        else:
            numericalized_text.extend([0] * (self.max_length - len(numericalized_text)))
            
        # Label'ı int64'e dönüştür
        try:
            label_tensor = torch.tensor(int(label), dtype=torch.long)
        except (ValueError, TypeError) as e:
            print(f"Hata: Label dönüştürülemiyor. Label değeri: {label}, Tip: {type(label)}")
            raise
            
        return {
            'text': torch.tensor(numericalized_text),
            'label': label_tensor
        }

def clean_text(text):
    """Metin temizleme işlemleri."""
    if not isinstance(text, str):
        text = str(text)
    
    # Küçük harfe çevir
    text = text.lower()
    
    # URL'leri temizle
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Kullanıcı adlarını temizle
    text = re.sub(r'@\w+', '', text)
    
    # Hashtag'leri temizle ama kelimeyi tut
    text = re.sub(r'#', '', text)
    
    # Sayıları temizle
    text = re.sub(r'\d+', '', text)
    
    # Noktalama işaretlerini temizle
    text = re.sub(r'[^\w\s]', '', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def lemmatize_text(text):
    """Kelimeleri lemmatize et."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenize
    words = word_tokenize(text)
    
    # Stopword'leri kaldır ve lemmatize et
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def compute_class_weights(labels):
    """Sınıf ağırlıklarını hesapla."""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)

def augment_text(text):
    """Basit metin augmentation teknikleri."""
    augmented_texts = []
    words = text.split()
    
    if len(words) > 3:
        # Kelime silme
        remove_idx = np.random.randint(0, len(words))
        aug_text = ' '.join(words[:remove_idx] + words[remove_idx+1:])
        augmented_texts.append(aug_text)
        
        # Kelime sırası değiştirme
        shuffled_words = words.copy()
        np.random.shuffle(shuffled_words)
        aug_text = ' '.join(shuffled_words)
        augmented_texts.append(aug_text)
    
    return augmented_texts

def prepare_data(data_path):
    """Veri setini hazırla ve böl."""
    try:
        # Veriyi oku
        df = pd.read_csv(data_path)
        
        # Sütun isimlerini kontrol et
        required_columns = ['tweet_text', 'cyberbullying_type']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Veri setinde gerekli sütunlar bulunamadı. Gerekli sütunlar: {required_columns}")
        
        # Metin temizleme ve normalizasyon
        df['tweet_text'] = df['tweet_text'].apply(clean_text)
        df['tweet_text'] = df['tweet_text'].apply(lemmatize_text)
        
        # Veri artırma
        augmented_data = []
        for idx, row in df.iterrows():
            aug_texts = augment_text(row['tweet_text'])
            for aug_text in aug_texts:
                augmented_data.append({
                    'tweet_text': aug_text,
                    'cyberbullying_type': row['cyberbullying_type']
                })
        
        # Artırılmış veriyi ekle
        aug_df = pd.DataFrame(augmented_data)
        df = pd.concat([df, aug_df], ignore_index=True)
        
        # Etiketleri ön işle
        df['cyberbullying_type'] = df['cyberbullying_type'].apply(preprocess_label)
        
        # Etiketleri sayısallaştır
        label_map = {label: idx for idx, label in config.CLASS_LABELS.items()}
        df['label'] = df['cyberbullying_type'].map(label_map)
        
        # NaN değerleri kontrol et
        if df['label'].isna().any():
            invalid_labels = df[df['label'].isna()]['cyberbullying_type'].unique()
            raise ValueError(f"Geçersiz etiketler bulundu: {invalid_labels}")
        
        # Sınıf ağırlıklarını hesapla
        class_weights = compute_class_weights(df['label'].values)
        
        # Veriyi böl
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df['tweet_text'].values,
            df['label'].values,
            train_size=config.TRAIN_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=df['label'].values
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=config.RANDOM_SEED,
            stratify=temp_labels
        )
        
        # Vocabulary oluştur
        vocab = Vocabulary()
        vocab.build_vocabulary(train_texts)
        
        # Dataset'leri oluştur
        train_dataset = CyberbullyingDataset(train_texts, train_labels, vocab, config.MAX_LENGTH)
        val_dataset = CyberbullyingDataset(val_texts, val_labels, vocab, config.MAX_LENGTH)
        test_dataset = CyberbullyingDataset(test_texts, test_labels, vocab, config.MAX_LENGTH)
        
        # DataLoader'ları oluştur
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=0)
        
        print(f"Veri seti yüklendi:")
        print(f"Eğitim seti boyutu: {len(train_texts)}")
        print(f"Doğrulama seti boyutu: {len(val_texts)}")
        print(f"Test seti boyutu: {len(test_texts)}")
        print(f"Kelime dağarcığı boyutu: {len(vocab)}")
        print(f"Sınıf dağılımı:")
        for label, count in df['cyberbullying_type'].value_counts().items():
            print(f"  {label}: {count}")
        
        return train_loader, val_loader, test_loader, vocab, class_weights
        
    except Exception as e:
        print(f"Veri hazırlama hatası: {str(e)}")
        raise 