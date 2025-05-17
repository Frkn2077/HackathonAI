"""
Model değerlendirme ve metrik hesaplama fonksiyonları.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from . import config
import os
import datetime

def evaluate_model(model, data_loader, criterion, device):
    """
    Modeli değerlendir ve metrikleri hesapla.
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Batch verilerini GPU'ya taşı
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            # Olasılıkları hesapla
            probabilities = torch.softmax(outputs, dim=1)
            
            # Metrikleri hesapla
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Tahminleri ve etiketleri sakla
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Ortalama kayıp ve doğruluk hesapla
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy, all_predictions, all_labels, all_probabilities

def save_confusion_matrix(predictions, labels, save_path, title='Confusion Matrix'):
    """Confusion matrix oluştur ve kaydet."""
    try:
        # Confusion matrix hesapla
        cm = confusion_matrix(labels, predictions)
        
        # Normalize edilmiş confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot oluştur
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ham confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(config.CLASS_LABELS.values()),
                   yticklabels=list(config.CLASS_LABELS.values()),
                   ax=ax1)
        ax1.set_title('Ham Confusion Matrix')
        ax1.set_xlabel('Tahmin Edilen')
        ax1.set_ylabel('Gerçek')
        
        # Normalize edilmiş confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlBu_r',
                   xticklabels=list(config.CLASS_LABELS.values()),
                   yticklabels=list(config.CLASS_LABELS.values()),
                   ax=ax2)
        ax2.set_title('Normalize Edilmiş Confusion Matrix')
        ax2.set_xlabel('Tahmin Edilen')
        ax2.set_ylabel('Gerçek')
        
        plt.suptitle(f'{title} - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        plt.tight_layout()
        
        # Kayıt yolunu mutlak yola çevir
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # Grafiği kaydet
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix kaydedildi: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Confusion matrix kaydedilirken hata oluştu: {str(e)}")
        plt.close()

def calculate_metrics(predictions, labels, probabilities=None):
    """Detaylı metrikleri hesapla."""
    # Classification report oluştur
    report = classification_report(labels, predictions,
                                 target_names=list(config.CLASS_LABELS.values()),
                                 output_dict=True)
    
    # Sınıf bazlı metrikler
    class_metrics = {}
    for class_name in config.CLASS_LABELS.values():
        class_metrics[class_name] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1-score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
    
    # Genel metrikler
    overall_metrics = {
        'accuracy': report['accuracy'],
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg']
    }
    
    # Hassasiyet analizi
    if probabilities is not None:
        probabilities = np.array(probabilities)
        for i, class_name in config.CLASS_LABELS.items():
            # Her sınıf için precision-recall eğrisi
            precision, recall, _ = precision_recall_curve(
                [1 if l == i else 0 for l in labels],
                [p[i] for p in probabilities]
            )
            class_metrics[class_name]['precision_curve'] = precision.tolist()
            class_metrics[class_name]['recall_curve'] = recall.tolist()
    
    return class_metrics, overall_metrics

def print_detailed_metrics(class_metrics, overall_metrics):
    """Detaylı metrikleri yazdır."""
    print("\n=== Detaylı Metrikler ===")
    print("\nSınıf Bazlı Metrikler:")
    for class_name, metrics in class_metrics.items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    print("\nGenel Metrikler:")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print("\nMacro Average:")
    print(f"  Precision: {overall_metrics['macro_avg']['precision']:.4f}")
    print(f"  Recall: {overall_metrics['macro_avg']['recall']:.4f}")
    print(f"  F1-Score: {overall_metrics['macro_avg']['f1-score']:.4f}")
    
    print("\nWeighted Average:")
    print(f"  Precision: {overall_metrics['weighted_avg']['precision']:.4f}")
    print(f"  Recall: {overall_metrics['weighted_avg']['recall']:.4f}")
    print(f"  F1-Score: {overall_metrics['weighted_avg']['f1-score']:.4f}")

def evaluate_and_save_metrics(model, test_loader, criterion, save_prefix='test'):
    """Test setinde değerlendir ve sonuçları kaydet."""
    device = next(model.parameters()).device
    
    # Model performansını değerlendir
    test_loss, test_accuracy, predictions, labels, probabilities = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Confusion matrix kaydet
    save_confusion_matrix(
        predictions, 
        labels,
        config.CONFUSION_MATRIX_PATH.format(f'{save_prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'),
        title=f'Test Sonuçları - {save_prefix}'
    )
    
    # Detaylı metrikleri hesapla
    class_metrics, overall_metrics = calculate_metrics(predictions, labels, probabilities)
    
    # Sonuçları yazdır
    print_detailed_metrics(class_metrics, overall_metrics)
    
    return test_loss, test_accuracy, class_metrics, overall_metrics

def test_single_text(model, text, vocab, device=None):
    """Tek bir metin için tahmin yap ve sonuçları göster."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        # Metni sayısallaştır
        numericalized_text = vocab.numericalize(text)
        if len(numericalized_text) > config.MAX_LENGTH:
            numericalized_text = numericalized_text[:config.MAX_LENGTH]
        else:
            numericalized_text.extend([0] * (config.MAX_LENGTH - len(numericalized_text)))
        
        # Tensor'a çevir ve boyut ekle
        text_tensor = torch.tensor(numericalized_text).unsqueeze(0).to(device)
        
        # Tahmin yap
        outputs = model(text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        
    # Sonuçları yazdır
    print("\n=== Tahmin Sonuçları ===")
    print(f"Girdi Metin: {text}")
    print(f"\nTahmin Edilen Sınıf: {config.CLASS_LABELS[prediction]}")
    print("\nSınıf Olasılıkları:")
    for i, prob in enumerate(probabilities[0]):
        print(f"{config.CLASS_LABELS[i]}: {prob.item():.4f}")
    
    return prediction, probabilities[0].cpu().numpy() 