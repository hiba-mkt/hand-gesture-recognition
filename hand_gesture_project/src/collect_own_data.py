"""OPTIONNEL mais recommandé : enregistrer VOS propres gestes avec la webcam.
Le dataset Kaggle est en infrarouge (Leap Motion) : un modèle entraîné uniquement
dessus généralise parfois mal à une webcam classique.

Usage : python src/collect_own_data.py --label fist --n 300
        python src/collect_own_data.py --label peace --n 300   (nouveau geste: victoire)
Appuyez sur ESPACE pour démarrer/pause la capture, q pour quitter.
Puis relancez : python src/train.py --use-own
"""
import argparse
import cv2
import pandas as pd

import config
from preprocessing import preprocess
from features import HandExtractor, normalize_landmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="nom du geste (ex. fist, palm, peace)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    extractor = HandExtractor(static_image_mode=False)
    rows, recording = [], False

    while len(rows) < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        frame = preprocess(cv2.flip(frame, 1))
        pts, _ = extractor.extract(frame)
        if recording and pts is not None:
            rows.append(["own", args.label, *normalize_landmarks(pts)])
        cv2.putText(frame, f"{args.label}: {len(rows)}/{args.n} {'REC' if recording else 'PAUSE'}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if recording else (200, 200, 200), 2)
        cv2.imshow("Collecte", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            recording = not recording
        elif k in (27, ord("q")):
            break
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()

    if rows:
        df = pd.DataFrame(rows, columns=["subject", "label", *config.FEATURE_COLS])
        header = not config.OWN_CSV.exists()
        df.to_csv(config.OWN_CSV, mode="a", header=header, index=False)
        print(f"{len(df)} échantillons '{args.label}' ajoutés à {config.OWN_CSV}")


if __name__ == "__main__":
    main()
