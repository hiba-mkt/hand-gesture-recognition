"""Étape 3 : entraînement SVM / Random Forest sur les landmarks normalisés.
Usage : python src/train.py [--merge] [--augment-flip] [--use-own]
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def flip_x(X):
    """Miroir horizontal : x -> -x (colonnes 0,3,6,...) pour simuler l'autre main."""
    Xf = X.copy()
    Xf[:, 0::3] *= -1
    return Xf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.FEATURES_CSV))
    ap.add_argument("--merge", action="store_true", help="fusionner fist_moved->fist, palm_moved->palm")
    ap.add_argument("--augment-flip", action="store_true", help="ajoute les mains en miroir au train")
    ap.add_argument("--use-own", action="store_true", help="ajoute data/own_features.csv au train")
    ap.add_argument("--test-subjects", nargs="+", default=["08", "09"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"subject": str})
    if args.merge:
        df["label"] = df["label"].replace(config.MERGE_MAP)

    test_mask = df["subject"].isin(args.test_subjects)
    train_df, test_df = df[~test_mask], df[test_mask]
    if len(test_df) == 0:
        raise SystemExit("Aucun échantillon de test : vérifiez --test-subjects.")
    if args.use_own and config.OWN_CSV.exists():
        own = pd.read_csv(config.OWN_CSV, dtype={"subject": str})
        if args.merge:
            own["label"] = own["label"].replace(config.MERGE_MAP)
        train_df = pd.concat([train_df, own], ignore_index=True)
        print(f"+ {len(own)} échantillons personnels ajoutés au train")

    X_train = train_df[config.FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["label"].values
    groups = train_df["subject"].values
    X_test = test_df[config.FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["label"].values

    if args.augment_flip:
        X_train = np.vstack([X_train, flip_x(X_train)])
        y_train = np.concatenate([y_train, y_train])
        groups = np.concatenate([groups, groups])

    print(f"Train: {len(X_train)} | Test: {len(X_test)} (sujets {args.test_subjects})")
    print("Classes:", sorted(set(y_train)))

    cv = GroupKFold(n_splits=min(3, len(set(groups))))
    candidates = {
        "SVM": GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True)),
            {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01]},
            cv=cv, scoring="f1_macro", n_jobs=-1),
        "RandomForest": GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 20]},
            cv=cv, scoring="f1_macro", n_jobs=1),
    }

    results = {}
    for name, gs in candidates.items():
        print(f"\n=== {name} ===")
        gs.fit(X_train, y_train, groups=groups)
        pred = gs.predict(X_test)
        acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
        print("Meilleurs paramètres:", gs.best_params_)
        print(f"Accuracy test: {acc:.4f} | F1 macro: {f1:.4f}")
        print(classification_report(y_test, pred, zero_division=0))
        results[name] = (f1, gs.best_estimator_, pred)

        config.OUTPUTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"{name} - matrice de confusion (acc={acc:.3f})")
        fig.tight_layout()
        fig.savefig(config.OUTPUTS_DIR / f"confusion_{name}.png", dpi=120)
        plt.close(fig)

    best = max(results, key=lambda k: results[k][0])
    print(f"\n>>> Meilleur modèle : {best} (F1={results[best][0]:.4f})")
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": results[best][1], "name": best,
                 "classes": list(results[best][1].classes_)}, config.MODEL_PATH)
    print("Modèle sauvegardé ->", config.MODEL_PATH)


if __name__ == "__main__":
    main()
