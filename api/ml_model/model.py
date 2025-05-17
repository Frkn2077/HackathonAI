"""
Özel NLP modeli için PyTorch implementasyonu.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class SelfAttention(nn.Module):
    """Özel self-attention mekanizması."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        device = x.device
        
        query = self.attention(x)
        key = self.attention(x)
        value = x
        
        scale = torch.sqrt(torch.FloatTensor([self.hidden_dim])).to(device)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale
        attention_weights = F.softmax(scores, dim=-1)
        
        attended = torch.matmul(attention_weights, value)
        return attended

class LSTMAttentionModel(nn.Module):
    """LSTM ve Attention tabanlı siber zorbalık tespit modeli."""
    def __init__(self, vocab_size):
        super().__init__()
        
        # Embedding katmanı
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        
        # Bi-directional LSTM katmanı
        self.lstm = nn.LSTM(
            config.EMBEDDING_DIM,
            config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            bidirectional=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            batch_first=True
        )
        
        # Self-attention mekanizması
        self.attention = SelfAttention(config.HIDDEN_DIM * 2)
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.HIDDEN_DIM * 2)
        self.layer_norm2 = nn.LayerNorm(config.HIDDEN_DIM)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        # Embedding katmanı
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM katmanı
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # Layer normalization
        lstm_out = self.layer_norm1(lstm_out)
        
        # Self-attention uygula
        attended = self.attention(lstm_out)
        
        # Global max pooling
        pooled = F.max_pool1d(
            attended.transpose(1, 2),
            attended.size(1)
        ).squeeze(2)
        
        # Fully connected katmanlar
        dense1 = self.fc1(pooled)
        dense1 = self.layer_norm2(dense1)
        dense1 = F.gelu(dense1)
        dense1 = self.dropout(dense1)
        
        # Son katman
        output = self.fc2(dense1)
        
        return output

def create_model(vocab_size):
    """Model oluştur ve cihaza taşı."""
    model = LSTMAttentionModel(vocab_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model

def load_model(model_path, vocab_size, device=None):
    """Modeli yükle ve cihaza taşı."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMAttentionModel(vocab_size)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Eğer state_dict doğrudan kaydedilmişse
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model 
    return model 