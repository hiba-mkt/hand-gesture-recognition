"""Prédire le geste d'une image fixe : python src/predict_image.py chemin/image.png"""
import sys
import cv2
import joblib

import config
from preprocessing import preprocess
from features import HandExtractor, normalize_landmarks

img = cv2.imread(sys.argv[1])
if img is None:
    raise SystemExit("Image introuvable.")
bundle = joblib.load(config.MODEL_PATH)
ext = HandExtractor(static_image_mode=True)
pts, _ = ext.extract(preprocess(img))
if pts is None:
    raise SystemExit("Aucune main détectée.")
x = normalize_landmarks(pts).reshape(1, -1)
proba = bundle["model"].predict_proba(x)[0]
i = proba.argmax()
print(f"Geste : {bundle['model'].classes_[i]}  (confiance {proba[i]*100:.1f}%)")
