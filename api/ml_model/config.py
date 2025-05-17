"""
Model ve eğitim konfigürasyonları için yapılandırma dosyası.
"""

# Veri seti yapılandırması
RANDOM_SEED = 42
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
MAX_LENGTH = 128  # Maximum tweet uzunluğu

# Model yapılandırması
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_CLASSES = 6

# Eğitim yapılandırması
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4

# Sınıf etiketleri
CLASS_LABELS = {
    0: "age",
    1: "ethnicity",
    2: "gender",
    3: "religion",
    4: "other_cyberbullying",
    5: "not_cyberbullying"
}

# Model kaydetme yolları
MODEL_SAVE_PATH = "saved_models/cyberbullying_model.pth"
VOCAB_SAVE_PATH = "saved_models/vocab.pkl"
CONFUSION_MATRIX_PATH = "metrics/confusion_matrix_{}.png"  # train/val/test
METRICS_SAVE_PATH = "metrics/metrics.json"

# Veri ön işleme yapılandırması
MIN_WORD_FREQ = 3  # Minimum kelime frekansı
MAX_VOCAB_SIZE = 50000  # Maximum vocabulary boyutu 