# Model eğitimi için fonksiyonlar.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import pickle
import sys
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Proje kök dizinini ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = os.path.dirname(current_dir)
sys.path.append(api_dir)

from ml_model import config
from ml_model.model import create_model, load_model
from ml_model.evaluate import evaluate_model, save_confusion_matrix, calculate_metrics, print_detailed_metrics
from ml_model.data_preprocessing import prepare_data
from ml_model.loss import CombinedLoss

def ensure_directories():
    """Gerekli klasörlerin varlığını kontrol et ve oluştur."""
    directories = [
        os.path.join(api_dir, 'metrics'),
        os.path.join(api_dir, 'saved_models')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Klasör kontrol edildi/oluşturuldu: {directory}")

def get_absolute_path(relative_path):
    """Göreceli yolu mutlak yola çevir."""
    return os.path.join(api_dir, relative_path)

def print_device_info():
    """Kullanılan cihaz bilgisini yazdır."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nCihaz Bilgisi:")
    print(f"Kullanılan Cihaz: {device}")
    if torch.cuda.is_available():
        print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
        print(f"Kullanılabilir GPU Sayısı: {torch.cuda.device_count()}")
        print(f"Mevcut GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # CUDA bellek durumunu yazdır
        print(f"Kullanılan GPU Belleği: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Bellek Önbelleği: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print("-" * 50)

def move_batch_to_device(batch, device):
    """Batch içindeki tensörleri belirtilen cihaza taşı."""
    return {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}

def save_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    """Confusion matrix'i kaydet ve görüntüle."""
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    # Matplotlib figure oluştur
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    # Başlık ve etiketler
    plt.title(title)
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    
    # Dosyaya kaydet
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nConfusion matrix kaydedildi: {save_path}")
    
    # Ekranda göster
    plt.show()
    plt.close()

def train_model(train_loader, val_loader, vocab, class_weights, model_path=None):
    """Modeli eğit ve kaydet."""
    # Gerekli klasörlerin varlığını kontrol et
    ensure_directories()
    
    # GPU kullanılabilirliğini kontrol et ve belleği temizle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print_device_info()
    
    # Model oluştur veya yükle
    if model_path and os.path.exists(model_path):
        print(f"Önceki model yükleniyor: {model_path}")
        try:
            model = load_model(model_path, len(vocab), device)
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
            print("Yeni model oluşturuluyor...")
            model = create_model(len(vocab))
    else:
        print("Yeni model oluşturuluyor...")
        model = create_model(len(vocab))
    
    model = model.to(device)
    
    # Optimizer, scheduler ve loss function oluştur
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Combined Loss kullan
    criterion = CombinedLoss(
        alpha=class_weights.to(device),
        gamma=2.0,  # Focal Loss gamma parametresi
        ce_weight=0.3  # CrossEntropy ağırlığı (0.3 CE, 0.7 Focal)
    ).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Wandb ile deney takibi başlat
    wandb.init(project="cyberbullying-detection", config={
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "model_type": "LSTM_Attention",
        "embedding_dim": config.EMBEDDING_DIM,
        "hidden_dim": config.HIDDEN_DIM,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "loss_type": "Combined(CE+Focal)",
        "focal_gamma": 2.0,
        "ce_weight": 0.3
    })
    
    try:
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.NUM_EPOCHS}')
            for batch in progress_bar:
                # Batch'i GPU'ya taşı
                batch = move_batch_to_device(batch, device)
                texts = batch['text']
                labels = batch['label']
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                # Gradient clipping ekle
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Backward pass ve optimize et
                loss.backward()
                optimizer.step()
                
                # Metrikleri hesapla
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Progress bar güncelle
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated() / 1024**3:.2f}GB' if torch.cuda.is_available() else 'N/A'
                })
            
            # Epoch metriklerini hesapla
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            val_loss, val_accuracy, val_predictions, val_labels, val_probabilities = evaluate_model(
                model, val_loader, criterion, device
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Sınıf bazlı metrikleri hesapla
            class_metrics, overall_metrics = calculate_metrics(val_predictions, val_labels, val_probabilities)
            
            # Wandb'ye metrikleri logla
            wandb.log({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "macro_f1": overall_metrics["macro_avg"]["f1-score"]
            })
            
            # Confusion matrix kaydet
            confusion_matrix_path = get_absolute_path(
                config.CONFUSION_MATRIX_PATH.format(f'val_epoch_{epoch + 1}')
            )
            save_confusion_matrix(
                val_labels,
                val_predictions,
                confusion_matrix_path,
                title=f'Validation Results - Epoch {epoch + 1}'
            )
            
            # Learning rate scheduler'ı güncelle
            scheduler.step(val_loss)
            
            print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}:')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            print(f'Macro F1: {overall_metrics["macro_avg"]["f1-score"]:.4f}')
            
            # Detaylı metrikleri yazdır
            print_detailed_metrics(class_metrics, overall_metrics)
            
            # Early stopping kontrolü
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = model.state_dict()
                
                # En iyi modeli kaydet
                model_save_path = get_absolute_path(config.MODEL_SAVE_PATH)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'class_weights': class_weights,
                    'vocab_size': len(vocab)
                }, model_save_path)
                print(f"En iyi model kaydedildi: {model_save_path}")
            else:
                patience_counter += 1
                print(f"Early stopping sayacı: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                    break
            
            # GPU bellek durumunu yazdır
            if torch.cuda.is_available():
                print(f"GPU Bellek Kullanımı: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    
    except Exception as e:
        print(f"\nEğitim sırasında hata oluştu: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        # Wandb oturumunu kapat
        wandb.finish()
    
    # Vocabulary'yi kaydet
    vocab_save_path = get_absolute_path(config.VOCAB_SAVE_PATH)
    with open(vocab_save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary kaydedildi: {vocab_save_path}")
    
    # Metrikleri kaydet
    metrics_save_path = get_absolute_path(config.METRICS_SAVE_PATH)
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'final_class_metrics': class_metrics,
        'final_overall_metrics': overall_metrics
    }
    
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrikler kaydedildi: {metrics_save_path}")
    
    return model, metrics

if __name__ == '__main__':
    try:
        # Gerekli klasörlerin varlığını kontrol et
        ensure_directories()
        
        data_path = os.path.join(api_dir, 'data', 'cyberbullying_dataset.csv')
        
        if not os.path.exists(data_path):
            print(f"HATA: Veri seti bulunamadı: {data_path}")
            print("\nLütfen aşağıdaki adımları takip edin:")
            print("1. 'api/data' klasörünün varlığından emin olun")
            print("2. 'cyberbullying_dataset.csv' dosyasını bu klasöre yerleştirin")
            print("3. Dosya adının doğru yazıldığından emin olun")
            sys.exit(1)
            
        print(f"Veri seti yükleniyor: {data_path}")
        train_loader, val_loader, test_loader, vocab, class_weights = prepare_data(data_path)
        print("Veri seti yüklendi.")
        
        model, metrics = train_model(train_loader, val_loader, vocab, class_weights)
        print("Model eğitimi başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"HATA: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)
