"""Construit data/features.csv à partir des images du dataset Kaggle leapGestRecog.
Usage : python src/extract_dataset.py [--step 2] [--gray] [--clahe]
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

import config
from preprocessing import preprocess
from features import HandExtractor, normalize_landmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--step", type=int, default=1, help="1 image sur N (accélère)")
    ap.add_argument("--gray", action="store_true", help="niveaux de gris")
    ap.add_argument("--clahe", action="store_true", help="CLAHE (contraste)")
    ap.add_argument("--out", default=str(config.FEATURES_CSV))
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.png"))[:: args.step]
    if not files:
        raise SystemExit(f"Aucune image trouvée dans {args.data_dir}. Voir README (étape 3).")
    print(f"{len(files)} images à traiter")

    extractor = HandExtractor(static_image_mode=True)
    rows, total, detected = [], {}, {}
    for path in tqdm(files):
        gesture_dir = path.parent.name           # ex. 03_fist
        subject = path.parent.parent.name        # ex. 00
        label = config.GESTURE_NAMES.get(gesture_dir)
        if label is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = preprocess(img, gray=args.gray, clahe=args.clahe)
        pts, _ = extractor.extract(img)
        total[label] = total.get(label, 0) + 1
        if pts is None:
            continue
        detected[label] = detected.get(label, 0) + 1
        rows.append([subject, label, *normalize_landmarks(pts)])
    extractor.close()

    df = pd.DataFrame(rows, columns=["subject", "label", *config.FEATURE_COLS])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\nTaux de détection MediaPipe par geste :")
    for k in sorted(total):
        d = detected.get(k, 0)
        print(f"  {k:12s} {d:5d}/{total[k]:5d}  ({100*d/total[k]:.1f}%)")
    print(f"\n{len(df)} échantillons sauvegardés -> {args.out}")


if __name__ == "__main__":
    main()
