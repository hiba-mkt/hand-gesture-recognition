"""Étape 2 : extraction des 21 repères avec MediaPipe Hands + normalisation."""
import cv2
import numpy as np
import mediapipe as mp


class HandExtractor:
    def __init__(self, static_image_mode=True, max_hands=1,
                 det_conf=0.5, track_conf=0.5, model_complexity=1):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def extract(self, frame_bgr):
        """Retourne (points (21,3) en pixels, objet landmarks pour le dessin)
        ou (None, None) si aucune main détectée."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        lm = res.multi_hand_landmarks[0]
        # x,y sont normalisés par largeur/hauteur -> on repasse en pixels pour
        # éviter la déformation des proportions sur les images non carrées.
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm.landmark], dtype=np.float32)
        return pts, lm

    def close(self):
        self.hands.close()


def normalize_landmarks(pts):
    """Invariance à la position et à la taille de la main :
    1) poignet (point 0) ramené à l'origine
    2) division par la distance maximale poignet-point (échelle)
    Retourne un vecteur de 63 valeurs."""
    pts = np.asarray(pts, dtype=np.float32).copy()
    pts -= pts[0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()
