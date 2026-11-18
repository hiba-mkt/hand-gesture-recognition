"""Étape 1 : prétraitement des frames."""
import cv2
from config import TARGET_WIDTH, USE_GRAY, USE_CLAHE


def preprocess(frame_bgr, width=TARGET_WIDTH, gray=USE_GRAY, clahe=USE_CLAHE):
    """Redimensionne (ratio conservé), option niveaux de gris + CLAHE.
    Retourne toujours une image BGR 3 canaux (exigée par MediaPipe)."""
    h, w = frame_bgr.shape[:2]
    if width and w != width:
        scale = width / w
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        frame_bgr = cv2.resize(frame_bgr, (width, int(h * scale)), interpolation=interp)

    if gray or clahe:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if clahe:
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        frame_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame_bgr
