"""
Flask API for cyberbullying detection
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from data_processing.data_processor import TextPreprocessor
from model.cyberbullying_model import CyberbullyingDetector
import os
import logging
import glob

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Model ve işlemci nesnelerini oluştur
MODEL_PATH = os.path.join('saved_models', 'cyberbullying_model.pth')
text_processor = TextPreprocessor()
model = CyberbullyingDetector(model_path=MODEL_PATH)

def process_text(text: str) -> dict:
    """
    Tek bir metin için analiz yapar.
    """
    # Metin ön işleme
    processed_text = text_processor.preprocess_text(text)
    logger.info(f"İşlenen metin: {processed_text}")
    
    # Model tahmini
    predictions = model.predict(processed_text)
    logger.info(f"Tahmin sonuçları: {predictions}")
    
    return predictions

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Tek bir metin için siber zorbalık analizi yapar."""
    try:
        logger.info("Metin analizi başlatılıyor...")
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Metin bulunamadı', 'success': False}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Boş metin analiz edilemez', 'success': False}), 400
        
        # Metin analizi
        results = process_text(text)
        
        # En son oluşturulan confusion matrix'i bul
        confusion_matrix_path = get_latest_confusion_matrix()
        
        return jsonify({
            'success': True,
            'text': text,
            'results': results,
            'confusion_matrix_path': f'/confusion-matrix?path={os.path.basename(confusion_matrix_path)}' if confusion_matrix_path else None
        })
        
    except Exception as e:
        logger.error(f"Metin analizi hatası: {str(e)}")
        return jsonify({
            'error': 'Analiz sırasında bir hata oluştu',
            'details': str(e),
            'success': False
        }), 500

def get_latest_confusion_matrix():
    """En son oluşturulan confusion matrix dosyasını bulur."""
    try:
        confusion_matrix_files = glob.glob(os.path.join('metrics', 'confusion_matrix_*.png'))
        if not confusion_matrix_files:
            return None
        return max(confusion_matrix_files, key=os.path.getctime)
    except Exception as e:
        logger.error(f"Confusion matrix dosyası bulunamadı: {str(e)}")
        return None

@app.route('/confusion-matrix')
def get_confusion_matrix():
    """Confusion matrix görüntüsünü döndürür."""
    try:
        path = request.args.get('path')
        if not path:
            return jsonify({'error': 'Dosya yolu belirtilmedi', 'success': False}), 400
            
        file_path = os.path.join('metrics', path)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Dosya bulunamadı', 'success': False}), 404
            
        return send_file(file_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Confusion matrix görüntüsü alınırken hata: {str(e)}")
        return jsonify({
            'error': 'Görüntü alınamadı',
            'details': str(e),
            'success': False
        }), 500

@app.route('/analyze-csv', methods=['POST'])
def analyze_csv():
    """CSV dosyasındaki metinler için toplu analiz yapar."""
    try:
        logger.info("CSV analizi başlatılıyor...")
        
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya yüklenmedi', 'success': False}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi', 'success': False}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Sadece CSV dosyaları kabul edilir', 'success': False}), 400
        
        # Geçici dosya oluştur ve kaydet
        temp_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # CSV dosyasını oku
        try:
            df = pd.read_csv(temp_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(temp_path, encoding='latin1')
            except Exception as e:
                os.remove(temp_path)
                return jsonify({'error': f'CSV dosyası okunamadı: {str(e)}', 'success': False}), 400
        finally:
            # Geçici dosyayı sil
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Sütun kontrolü
        text_columns = ['text', 'metin', 'yorum', 'comment']
        found_column = None
        for col in text_columns:
            if col in df.columns:
                found_column = col
                break
                
        if not found_column:
            return jsonify({
                'error': f'CSV dosyasında metin sütunu bulunamadı. Geçerli sütun isimleri: {", ".join(text_columns)}',
                'success': False
            }), 400
        
        # Metinleri işle
        results = []
        total = len(df)
        for idx, text in enumerate(df[found_column].fillna('').astype(str)):
            if text.strip():
                try:
                    # İlerleme durumunu logla
                    if (idx + 1) % 10 == 0:
                        logger.info(f"İşlenen: {idx + 1}/{total} ({((idx + 1)/total)*100:.1f}%)")
                    
                    # Metin analizi
                    result = process_text(text)
                    results.append({
                        'text': text,
                        'results': result
                    })
                except Exception as e:
                    logger.error(f"Metin işleme hatası: {text[:100]}... - Hata: {str(e)}")
                    continue
        
        if not results:
            return jsonify({
                'error': 'Hiçbir metin başarıyla analiz edilemedi',
                'success': False
            }), 400
            
        logger.info(f"CSV analizi tamamlandı. {len(results)} metin işlendi.")
        
        return jsonify({
            'success': True,
            'total_processed': len(results),
            'total_records': total,
            'data': results
        })
        
    except Exception as e:
        logger.error(f"CSV analizi hatası: {str(e)}")
        return jsonify({
            'error': 'CSV analizi sırasında bir hata oluştu',
            'details': str(e),
            'success': False
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Model hakkında bilgi verir."""
    return jsonify({
        'success': True,
        'classes': model.classes,
        'device': str(model.device),
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim
    })

if __name__ == '__main__':
    logger.info("API başlatılıyor...")
    app.run(debug=True, host='0.0.0.0', port=5000) 