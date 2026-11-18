# ✋ Hand Gesture Recognition (MediaPipe + SVM / Random Forest, CPU)

Pipeline complet : **Prétraitement → Extraction (21 repères MediaPipe) → Apprentissage (SVM / RF) → Reconnaissance temps réel**.

```
hand_gesture_project/
├── requirements.txt
├── README.md
├── data/                  <- y placer le dataset Kaggle (leapGestRecog)
├── models/                <- modèle entraîné (gesture_model.joblib)
├── outputs/               <- matrices de confusion
└── src/
    ├── config.py              paramètres centraux
    ├── preprocessing.py       étape 1 : resize, gris, CLAHE
    ├── features.py            étape 2 : MediaPipe Hands + normalisation
    ├── extract_dataset.py     images Kaggle -> data/features.csv
    ├── train.py               étape 3 : SVM & Random Forest (GridSearch)
    ├── realtime.py            étape 4 : inférence webcam temps réel
    ├── collect_own_data.py    (recommandé) collecter VOS gestes à la webcam
    └── predict_image.py       prédire une image fixe
```

## Étapes à suivre

### 1. Installer l'environnement (Python 3.9 à 3.11 recommandé)
```bash
cd hand_gesture_project
python -m venv venv
# Windows : venv\Scripts\activate      |  Linux/Mac : source venv/bin/activate
pip install -r requirements.txt
```
> MediaPipe ne supporte pas encore toujours les toutes dernières versions de Python : en cas d'erreur d'installation, utilisez Python 3.10 ou 3.11.

### 2. Télécharger le dataset
Page : https://www.kaggle.com/datasets/gti-upm/leapgestrecog (compte Kaggle gratuit).
- Soit téléchargement manuel du ZIP, soit via la CLI :
```bash
pip install kaggle
kaggle datasets download -d gti-upm/leapgestrecog -p data --unzip
```

### 3. Vérifier l'emplacement
Les images doivent se trouver sous `data/leapGestRecog/` avec la structure `<sujet 00..09>/<geste>/frame_*.png`
(ex. `data/leapGestRecog/00/03_fist/frame_00_03_0001.png`).
Le dossier est parfois doublé (`leapGestRecog/leapGestRecog/00/...`) : ce n'est pas grave, le script cherche récursivement.
Sinon, indiquez le chemin avec `--data-dir`.

### 4. Extraire les caractéristiques (prétraitement + MediaPipe)
Test rapide (1 image sur 4, quelques minutes) :
```bash
python src/extract_dataset.py --step 4
```
Version complète (~20 000 images, plus long) :
```bash
python src/extract_dataset.py
```
Options : `--gray` (niveaux de gris), `--clahe` (contraste local).
Le script affiche le **taux de détection par geste**. ⚠️ Les images Kaggle sont des images infrarouges de Leap Motion : MediaPipe peut rater certaines mains, c'est normal (ces images sont ignorées). Si le taux est faible, essayez `--clahe`.

### 5. Entraîner et évaluer
```bash
python src/train.py --merge --augment-flip
```
- `--merge` : fusionne `fist_moved→fist` et `palm_moved→palm`
- `--augment-flip` : ajoute des mains en miroir (robustesse main gauche/droite)
- Évaluation **par sujet** (les sujets 08 et 09 servent de test) : pas de fuite de données, résultat honnête.
- Compare SVM (RBF) et Random Forest avec GridSearch, garde le meilleur, sauvegarde `models/gesture_model.joblib` et les matrices de confusion dans `outputs/`.

### 6. (Recommandé) Ajouter vos propres gestes à la webcam
Le dataset est infrarouge, donc un modèle entraîné uniquement dessus peut mal marcher avec une webcam normale. Enregistrez vos gestes (ESPACE = démarrer/pause) :
```bash
python src/collect_own_data.py --label palm --n 300
python src/collect_own_data.py --label fist --n 300
python src/collect_own_data.py --label peace --n 300    # victoire ✌️
python src/collect_own_data.py --label ok --n 300
```
Puis ré-entraînez en les incluant :
```bash
python src/train.py --merge --augment-flip --use-own
```
Astuce : variez distance, angle, éclairage, et utilisez les deux mains.

### 7. Reconnaissance en temps réel (CPU)
```bash
python src/realtime.py
```
Options : `--camera 1` (autre webcam), `--gray`, `--clahe`, `--threshold 0.7`, `--smooth 10`, `--no-mirror`. Quitter : `q` ou `Échap`.
Le texte affiche le geste, la confiance et le FPS. Un vote majoritaire sur les dernières frames stabilise l'affichage.

### 8. Image fixe
```bash
python src/predict_image.py chemin/vers/image.png
```

## Explication technique (pour votre rapport)
- **Prétraitement** : redimensionnement à largeur fixe (ratio conservé), niveaux de gris optionnels, CLAHE optionnel ; réconversion 3 canaux car MediaPipe attend du RGB.
- **Extraction** : MediaPipe Hands → 21 points × (x, y, z) = 63 valeurs. x et y repassés en pixels pour ne pas déformer les proportions.
- **Normalisation** : translation (poignet = origine) + mise à l'échelle (distance max au poignet) → invariance à la position et à la taille de la main dans l'image.
- **Classification** : `StandardScaler + SVC(RBF)` vs `RandomForest`, hyperparamètres choisis par `GridSearchCV` avec `GroupKFold` par sujet ; métriques : accuracy, F1 macro, matrice de confusion.
- **Inférence** : MediaPipe en mode vidéo (suivi) + prédiction sklearn + lissage temporel ; tout fonctionne sur CPU.

## Dépannage
| Problème | Solution |
|---|---|
| `Aucune image trouvée` | Vérifier le chemin du dataset (étape 3) ou utiliser `--data-dir` |
| Faible taux de détection | `--clahe`, ou s'appuyer davantage sur `collect_own_data.py` |
| Bon score test mais mauvais en webcam | Domaine différent (infrarouge vs RGB) : étape 6 |
| `module 'mediapipe' has no attribute 'solutions'` | `pip install mediapipe==0.10.14` |
| Webcam non ouverte | Essayer `--camera 1`, fermer les autres applis utilisant la caméra |
