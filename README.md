#  ASL Alphabet Recognition — Real-Time Hand Sign Detection

> Projet réalisé par **Hibat Allah Hadjkacem** & **Ktiti Amal** .

---

##  Description

Ce projet implémente un système de **reconnaissance de signes ASL (American Sign Language)** en temps réel à partir d'une webcam. Dans cette version, le modèle est entraîné et testé sur les **5 premières lettres de l'alphabet : A, B, C, D et E**, constituant une preuve de concept complète et fonctionnelle.

Il repose sur l'extraction de points clés de la main avec **MediaPipe** et la classification par un réseau de neurones **MLP (Multi-Layer Perceptron)** entraîné sur le dataset Kaggle ASL Alphabet.

L'application détecte la lettre signée par l'utilisateur, l'affiche à l'écran et la prononce à voix haute grâce à un moteur text-to-speech.

---


---

##  Structure du projet

```
asl-recognition/
│
├── asl_alphabet_train/          # Dataset brut (Kaggle)
├── processed_dataset/
│   ├── train/                   # Images prétraitées par lettre (.npy)
│   └── test/                    # Images de test par lettre (.npy)
│
├── preprocessing.ipynb          # Prétraitement des images
├── feature_extraction.ipynb     # Extraction des landmarks MediaPipe
├── train_mlp.ipynb              # Entraînement et évaluation du MLP
├── realtime_demo.ipynb          # Démonstration webcam temps réel
│
├── asl_mlp_model.pkl            # Modèle MLP sauvegardé
├── label_encoder.pkl            # Encodeur des labels sauvegardé
│
└── README.md
```

---

##  Pipeline du projet

```
Dataset ASL (Kaggle)
        ↓
Prétraitement (resize 64×64, normalisation, Gaussian Blur)
        ↓
Extraction de features (MediaPipe → 21 landmarks × 3 coords = 63 valeurs)
        ↓
Entraînement MLP (128 → 64 → softmax)
        ↓
Prédiction temps réel (webcam + buffer de stabilisation + TTS)
```

---

##  Dataset

- **Source** : [ASL Alphabet Dataset — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Contenu** : Images statiques des lettres A–Z (dataset complet)
- **Lettres utilisées dans ce projet** : **A, B, C, D, E uniquement** (5 classes)
- **Sélection** : 500 images par lettre → **2 500 images au total**
- **Split** : 85% entraînement / 15% test

---

##  Prétraitement des images

Pour chaque image :
1. Redimensionnement à **64×64 pixels**
2. Conversion **BGR → RGB**
3. Filtre **Gaussian Blur** (kernel 3×3) pour réduire le bruit
4. **Normalisation** des pixels entre 0 et 1
5. Sauvegarde au format `.npy` pour un chargement rapide

---

## Extraction de features — MediaPipe

Au lieu de travailler sur les pixels bruts (sensibles aux variations de fond et de luminosité), on utilise **MediaPipe Hands** pour extraire les **21 points clés (landmarks)** de la main, chacun avec 3 coordonnées **(x, y, z)**.

→ Vecteur de features final : **63 valeurs** par image

**Avantages :**
- Invariant à la couleur de peau et au fond
- Très léger et rapide
- Robuste aux variations d'éclairage

---

##  Modèle — MLP (MLPClassifier)

| Paramètre | Valeur |
|-----------|--------|
| Architecture | 63 → 128 → 64 → **5 classes (A–E)** |
| Activation | ReLU |
| Optimiseur | Adam |
| Batch size | 16 |
| Max itérations | 100 |
| Early stopping | Oui (validation_fraction=0.1) |

Le modèle est entraîné avec `sklearn.neural_network.MLPClassifier` et sauvegardé avec `joblib`.

---

##  Résultats

- Matrice de confusion générée après l'entraînement
- Rapport de classification (précision, rappel, F1-score) par lettre
- Accuracy globale affichée à la fin de l'entraînement

---

##  Démonstration temps réel

La détection en temps réel fonctionne comme suit :

1. Capture webcam frame par frame
2. Détection des landmarks avec MediaPipe
3. **Normalisation** des landmarks (centrage sur le poignet, mise à l'échelle)
4. Prédiction par le MLP
5. **Buffer de 7 frames** + vote majoritaire pour stabiliser la prédiction
6. Affichage de la lettre à l'écran
7. **Feedback vocal** (pyttsx3) : la lettre est prononcée une fois stabilisée pendant 8 frames consécutives

---

##  Installation

### Prérequis

- Python 3.11
- Environnement Conda `env_mediapipe` (recommandé)
- Jupyter Notebook / JupyterLab

### Dépendances

```bash
pip install opencv-python mediapipe scikit-learn numpy matplotlib seaborn joblib pyttsx3
```

### Lancer le projet

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/asl-recognition.git
cd asl-recognition

# 2. Activer l'environnement
conda activate env_mediapipe

# 3. Lancer Jupyter
jupyter notebook

# 4. Exécuter les notebooks dans l'ordre :
#    preprocessing.ipynb → feature_extraction.ipynb → train_mlp.ipynb → realtime_demo.ipynb
```

>  Pour l'exécution dans Jupyter, sélectionner le kernel : **Python 3.11 (env_mediapipe)**

---

##  Technologies utilisées

| Outil | Usage |
|-------|-------|
| OpenCV | Lecture et prétraitement des images, capture webcam |
| MediaPipe | Détection et extraction des landmarks de la main |
| scikit-learn | Modèle MLP, encodage des labels, évaluation |
| NumPy | Manipulation des tableaux de données |
| Matplotlib / Seaborn | Visualisation (matrice de confusion) |
| pyttsx3 | Synthèse vocale (text-to-speech) |
| joblib | Sauvegarde et chargement du modèle |

---

##  Limitations connues

- **Périmètre limité à A–E** : cette version est une preuve de concept ; les 21 autres lettres ne sont pas encore supportées
- Les lettres **J et Z** nécessitent du mouvement et ne peuvent pas être traitées avec des images statiques
- La détection peut être moins précise dans des conditions de faible éclairage
- Le modèle est entraîné uniquement sur des images statiques Kaggle ; les performances varient selon la morphologie de la main

---

##  Améliorations possibles

- **Étendre à l'alphabet complet** A–Z (en excluant J et Z dans un premier temps)
- Ajouter J et Z via un modèle séquentiel (LSTM ou GRU) sur des séquences de landmarks
- Étendre à la reconnaissance de mots complets
- Déployer en application web avec Flask ou Streamlit

---

##  Licence

Ce projet est réalisé à des fins académiques.
