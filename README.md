 # Reconnaissance d'images de Tajine et Kaftan marocains par Machine Learning

Français / English — Ce dépôt contient le code, les scripts et les instructions pour entraîner et déployer un modèle de reconnaissance d'images capable de classer des vêtements/traditions marocaines (ex. tajine décoratif, kaftan, etc.). L'objectif principal est de fournir une chaîne de bout en bout : préparation des données, entraînement, évaluation et service d'inférence.

Short English summary
This project trains and serves an image classification model for Moroccan cultural items (e.g., tajine decorations, kaftans). It includes dataset conventions, training and evaluation scripts, and a lightweight inference API.

---

## Fonctionnalités (Features)
- Préparation basique des données et augmentation d'images
- Modèles basés sur PyTorch (ex : ResNet, EfficientNet)
- Scripts pour entraîner, évaluer et inférer
- Export de modèle et instructions d'export ONNX
- Exemple d'API d'inférence (FastAPI / Uvicorn)
- Suggestions pour monitorer les expériences (TensorBoard / Weights & Biases)

---

## Structure du dépôt (Repository structure)
Exemple de structure attendue :
- data/                      # répertoire pour dataset local (non poussé)
  - train/
    - tajine/
    - kaftan/
    - autre_classes/
  - val/
  - test/
- notebooks/                 # notebooks d'exploration & visualisation
- src/
  - data_utils.py            # chargement, transformations, augmentations
  - model.py                 # définitions de modèles & helpers
  - train.py                 # script d'entraînement
  - evaluate.py              # évaluation & rapports
  - predict.py               # script d'inférence pour une image
  - app.py                   # API (FastAPI) pour servir le modèle
- requirements.txt           # dépendances Python
- README.md
- LICENSE

Adaptez la structure aux fichiers réellement présents dans le dépôt.

---

## Prérequis (Requirements)
- Python 3.8+
- CUDA (optionnel, pour entraînement GPU)
- git (pour cloner le dépôt)
- Recommandé : virtualenv ou conda

Exemples de bibliothèques utilisées :
- torch, torchvision
- numpy, pandas
- pillow (PIL)
- scikit-learn
- matplotlib, seaborn
- fastapi, uvicorn (pour servir le modèle)
- tensorboard / wandb (optionnel pour monitoring)

---

## Installation (local)
1. Cloner le dépôt
   git clone https://github.com/ISMAILELOUAZZANI/Reconnaissance-d-images-de-Tajine-et-Kaftan-marocains-par--Machine-Learning.git
   cd Reconnaissance-d-images-de-Tajine-et-Kaftan-marocains-par--Machine-Learning

2. Créer un environnement virtuel et l'activer
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows PowerShell

3. Installer les dépendances
   pip install -r requirements.txt

Si vous n'avez pas de requirements.txt, installez les paquets clés :
   pip install torch torchvision numpy pandas pillow scikit-learn matplotlib seaborn fastapi uvicorn

---

## Préparer les données (Dataset)
Organisez vos images comme suit :
- data/
  - train/
    - tajine/   (images .jpg/.png)
    - kaftan/
    - autre_classes/
  - val/
  - test/

Conseils :
- Redimensionnez/croppez les images à une taille uniforme (ex : 224x224).
- Appliquez des augmentations (flip, rotation, variations de couleur) pour améliorer la robustesse.
- Veillez à avoir un split équilibré entre classes.

Si vous utilisez un dataset public, documentez la source et les licences.

---

## Entraînement (Training)
Exemple de commande pour entraîner un modèle (script attendu : src/train.py) :

python src/train.py \
  --data-dir data \
  --model resnet50 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --pretrained \
  --checkpoint-dir checkpoints

Options courantes :
- --data-dir : chemin vers les dossiers train/val
- --model : resnet18 | resnet34 | resnet50 | efficientnet_b0 ...
- --epochs : nombre d'époques
- --batch-size : taille de batch
- --lr : learning rate
- --pretrained : utiliser des poids pré-entraînés
- --checkpoint-dir : dossier pour sauvegarder les checkpoints

Sorties attendues :
- Meilleur modèle sauvegardé (ex : checkpoints/best_model.pth)
- Logs/metrics (TensorBoard ou CSV)

---

## Évaluation (Evaluation)
Pour évaluer sur l'ensemble de test :
python src/evaluate.py --data-dir data --model-path checkpoints/best_model.pth --batch-size 32

Metrics usuelles :
- Accuracy top-1 / top-5
- Precision / Recall / F1 par classe
- Matrice de confusion
- Courbes ROC (si binaires ou pour chaque classe)

Générez un rapport (classification_report) et visualisez la matrice de confusion pour analyser les confusions entre classes (ex : tajine vs motifs similaires).

---

## Inférence & API (Inference & Serving)
Script d'inférence pour une image :
python src/predict.py --image path/to/image.jpg --model-path checkpoints/best_model.pth

Exemple minimal pour lancer une API FastAPI (src/app.py) :
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

Endpoints suggérés :
- POST /predict (multipart/form-data image) -> retourne la classe prédite et les probabilités
- GET /health -> health check

Conseils de production :
- Sérialiser le modèle (torch.save) ou exporter en ONNX pour servir avec d'autres runtimes.
- Utiliser Gunicorn + Uvicorn worker ou un service Kubernetes pour scalabilité.
- Mettre en place logging et monitoring.

Export ONNX (exemple) :
python - <<'PY'
import torch
model = torch.load("checkpoints/best_model.pth", map_location="cpu")
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "model.onnx", opset_version=11)
PY

---

## Résultats attendus (Expected results)
- Un modèle capable de distinguer correctement les classes cibles (objectif de précision dépendant de la qualité des données).
- Matrice de confusion et rapport détaillé montrant quelles classes sont confondues.
- Endpoint d'inférence prêt à servir des images en production.

---

## Bonnes pratiques (Notes / Tips)
- Commencez avec un modèle pré-entraîné (transfer learning).
- Normalisez les images avec les mêmes statistiques que celles utilisées pour l'entraînement des backbone pré-entraînés (ex : mean/std ImageNet).
- Vérifiez l'équilibrage des classes : si déséquilibré, utilisez poids de classe ou oversampling/undersampling.
- Sauvegardez les hyperparamètres et seeds pour reproductibilité.
- Testez l'API localement avec curl ou Postman.

---

## Contribuer (Contributing)
Contributions bienvenues : ajout de nouveaux datasets, scripts d'augmentation, notebooks d'analyse, pipelines CI pour tests, améliorations de modèle.

Processus :
1. Forkez le dépôt
2. Créez une branche feature/ma-fonctionnalité
3. Ajoutez tests et documentation
4. Ouvrez une Pull Request en décrivant les changements

---

## Licence (License)
Ajoutez le fichier LICENSE à la racine avec la licence souhaitée (MIT, Apache-2.0, etc.). Si vous n'avez pas encore décidé, MIT est un bon choix permissif.

---

## Contact
Mainteneur : ISMAILELOUAZZANI
- GitHub: https://github.com/ISMAILELOUAZZANI

Si vous voulez que je génère aussi :
- un fichier requirements.txt avec versions probables,
- un template de train.py / predict.py / app.py,
- un notebook d'exploration,
dites-moi lequel et je le crée.