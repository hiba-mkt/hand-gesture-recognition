"""Étape 4 : reconnaissance en temps réel (webcam, CPU).
Usage : python src/realtime.py [--camera 0] [--gray] [--no-mirror]
Touche q ou ESC pour quitter.
"""
import argparse
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
import mediapipe as mp

import config
from preprocessing import preprocess
from features import HandExtractor, normalize_landmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.6, help="confiance minimale")
    ap.add_argument("--smooth", type=int, default=7, help="fenêtre de vote majoritaire")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    print(f"Modèle chargé : {bundle['name']} | classes : {bundle['classes']}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra.")
    extractor = HandExtractor(static_image_mode=False)
    drawer, hands_mod = mp.solutions.drawing_utils, mp.solutions.hands
    history = deque(maxlen=args.smooth)
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        frame = preprocess(frame, width=args.width, gray=args.gray, clahe=args.clahe)

        pts, lm = extractor.extract(frame)
        text, color = "Aucune main", (0, 0, 255)
        if pts is not None:
            drawer.draw_landmarks(frame, lm, hands_mod.HAND_CONNECTIONS)
            feats = normalize_landmarks(pts).reshape(1, -1)
            proba = model.predict_proba(feats)[0]
            idx = int(np.argmax(proba))
            history.append(model.classes_[idx] if proba[idx] >= args.threshold else "?")
            label = Counter(history).most_common(1)[0][0]
            text = f"{label} ({proba[idx]*100:.0f}%)"
            color = (0, 200, 0) if label != "?" else (0, 165, 255)
        else:
            history.clear()

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
