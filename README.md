# DiagnosticKNN — Application Web Flask

Application web permettant d'utiliser deux modèles de classification KNN (entraînés avec scikit-learn) :
- **Diagnostic du cancer du sein** (Benign / Malignant)
- **Diagnostic du diabète** (positif / négatif)

## Structure du projet

```
webapp/
├── app.py                  # Application Flask (routes, chargement des modèles, prédiction)
├── requirements.txt        # Dépendances Python
├── models/                 # Modèles entraînés (générés par les notebooks)
│   ├── features_cancer_sein.pkl
│   ├── scaler_cancer_sein.pkl
│   ├── knn_cancer_sein.pkl
│   ├── features_diabetes.pkl
│   ├── scaler_diabetes.pkl
│   └── knn_diabetes.pkl
├── templates/               # Pages HTML (Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── cancer_sein.html
│   └── diabetes.html
└── static/css/style.css     # Style de l'application
```

## Installation et lancement

```bash
# 1. Créer un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate        # Sous Windows : venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
python app.py
```

L'application est alors accessible à l'adresse : http://127.0.0.1:5000

## Fonctionnement

1. Au démarrage, `app.py` charge les 6 fichiers `pickle` (3 par modèle : liste des variables,
   scaler, modèle KNN) présents dans le dossier `models/`.
   
2. Chaque page de diagnostic (`/cancer-du-sein` et `/diabete`) affiche un formulaire dont les
   champs sont générés dynamiquement à partir de la liste des variables (`features_*.pkl`),
   avec des valeurs par défaut pré-remplies (médianes du jeu de données d'apprentissage).

3. À la soumission du formulaire (POST), les valeurs saisies sont :
   - transformées avec le même `scaler` que celui utilisé lors de l'entraînement,
   - transmises au modèle KNN pour prédiction (`predict`) et calcul des probabilités
     (`predict_proba`),
   - affichées avec le diagnostic, la classe prédite et le niveau de confiance du modèle.

