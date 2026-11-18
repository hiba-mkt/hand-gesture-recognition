"""Entraînement avec évaluation sur VOS données webcam.
Usage : python src/train_own.py [--with-dataset] [--test-frac 0.2]
"""
import argparse
import warnings
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
from train import flip_x

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-dataset", action="store_true",
                    help="ajoute les images Kaggle au train")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    own = pd.read_csv(config.OWN_CSV, dtype={"subject": str})
    own["label"] = own["label"].replace(config.MERGE_MAP)
    # Découpage par blocs : la fin de chaque geste sert de test
    # (les images consécutives sont très proches, un tirage aléatoire serait trop optimiste)
    own["pos"] = own.groupby("label").cumcount()
    n = own.groupby("label")["pos"].transform("max") + 1
    is_test = own["pos"] >= n * (1 - args.test_frac)
    train_own, test_own = own[~is_test].copy(), own[is_test]
    train_own["group"] = train_own["label"] + "_" + (train_own["pos"] // 30).astype(str)

    parts = [train_own[["label", "group", *config.FEATURE_COLS]]]
    if args.with_dataset:
        ds = pd.read_csv(config.FEATURES_CSV, dtype={"subject": str})
        ds["label"] = ds["label"].replace(config.MERGE_MAP)
        ds["group"] = "ds_" + ds["subject"]
        parts.append(ds[["label", "group", *config.FEATURE_COLS]])
    train_df = pd.concat(parts, ignore_index=True)

    X_train = train_df[config.FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["label"].values
    groups = train_df["group"].values
    X_train = np.vstack([X_train, flip_x(X_train)])
    y_train = np.concatenate([y_train, y_train])
    groups = np.concatenate([groups, groups])
    X_test = test_own[config.FEATURE_COLS].values.astype(np.float32)
    y_test = test_own["label"].values
    print(f"Train: {len(X_train)} | Test (vos données): {len(X_test)}")

    cv = GroupKFold(n_splits=5)
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
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    for name, gs in candidates.items():
        print(f"\n=== {name} ===")
        gs.fit(X_train, y_train, groups=groups)
        pred = gs.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        print("Meilleurs paramètres:", gs.best_params_)
        print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
        print(classification_report(y_test, pred, zero_division=0))
        results[name] = (f1, gs.best_estimator_)
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax,
                                                xticks_rotation=45, colorbar=False)
        ax.set_title(f"{name} - vos données (acc={acc:.3f})")
        fig.tight_layout()
        fig.savefig(config.OUTPUTS_DIR / f"confusion_own_{name}.png", dpi=120)
        plt.close(fig)

    best = max(results, key=lambda k: results[k][0])
    print(f"\n>>> Meilleur modèle : {best} (F1={results[best][0]:.4f})")
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": results[best][1], "name": best,
                 "classes": list(results[best][1].classes_)}, config.MODEL_PATH)
    print("Modèle sauvegardé ->", config.MODEL_PATH)


if __name__ == "__main__":
    main()