"""Configuration centrale du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "leapGestRecog"      # dossier du dataset Kaggle dézippé
FEATURES_CSV = DATA_DIR / "features.csv"       # landmarks du dataset Kaggle
OWN_CSV = DATA_DIR / "own_features.csv"        # landmarks de VOS propres images (webcam)
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "gesture_model.joblib"

# Prétraitement
TARGET_WIDTH = 640        # largeur après redimensionnement (ratio conservé)
USE_GRAY = False          # niveaux de gris (ré-converti en 3 canaux pour MediaPipe)
USE_CLAHE = False         # égalisation locale du contraste (robustesse lumière)

NUM_LANDMARKS = 21
FEATURE_COLS = [f"{a}{i}" for i in range(NUM_LANDMARKS) for a in ("x", "y", "z")]

# Gestes du dataset leapGestRecog
GESTURE_NAMES = {
    "01_palm": "palm", "02_l": "l", "03_fist": "fist", "04_fist_moved": "fist_moved",
    "05_thumb": "thumb", "06_index": "index", "07_ok": "ok",
    "08_palm_moved": "palm_moved", "09_c": "c", "10_down": "down",
}
# Fusion optionnelle (--merge) : les variantes "moved" sont le même geste
MERGE_MAP = {"fist_moved": "fist", "palm_moved": "palm"}
