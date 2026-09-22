import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def charger_modele(nom):
    """Charge les 3 fichiers pickle (features, scaler, modèle) associés à un dataset."""
    with open(os.path.join(MODELS_DIR, f"features_{nom}.pkl"), "rb") as f:
        features = list(pickle.load(f))
    with open(os.path.join(MODELS_DIR, f"scaler_{nom}.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, f"knn_{nom}.pkl"), "rb") as f:
        model = pickle.load(f)
    return features, scaler, model


# Chargement des modèles une seule fois, au démarrage de l'application
features_bc, scaler_bc, model_bc = charger_modele("cancer_sein")
features_db, scaler_db, model_db = charger_modele("diabetes")


# Définition des champs de formulaire (label affiché, valeur par défaut

LIBELLES_BC = {
    "radius_mean": "Rayon (moyenne)", "texture_mean": "Texture (moyenne)",
    "perimeter_mean": "Périmètre (moyenne)", "area_mean": "Aire (moyenne)",
    "smoothness_mean": "Régularité du contour (moyenne)", "compactness_mean": "Compacité (moyenne)",
    "concavity_mean": "Concavité (moyenne)", "concave points_mean": "Points concaves (moyenne)",
    "symmetry_mean": "Symétrie (moyenne)", "fractal_dimension_mean": "Dimension fractale (moyenne)",
    "radius_se": "Rayon (erreur type)", "texture_se": "Texture (erreur type)",
    "perimeter_se": "Périmètre (erreur type)", "area_se": "Aire (erreur type)",
    "smoothness_se": "Régularité du contour (erreur type)", "compactness_se": "Compacité (erreur type)",
    "concavity_se": "Concavité (erreur type)", "concave points_se": "Points concaves (erreur type)",
    "symmetry_se": "Symétrie (erreur type)", "fractal_dimension_se": "Dimension fractale (erreur type)",
    "radius_worst": "Rayon (pire valeur)", "texture_worst": "Texture (pire valeur)",
    "perimeter_worst": "Périmètre (pire valeur)", "area_worst": "Aire (pire valeur)",
    "smoothness_worst": "Régularité du contour (pire valeur)", "compactness_worst": "Compacité (pire valeur)",
    "concavity_worst": "Concavité (pire valeur)", "concave points_worst": "Points concaves (pire valeur)",
    "symmetry_worst": "Symétrie (pire valeur)", "fractal_dimension_worst": "Dimension fractale (pire valeur)",
}

DEFAUTS_BC = {
    "radius_mean": 13.37, "texture_mean": 18.84, "perimeter_mean": 86.24, "area_mean": 551.1,
    "smoothness_mean": 0.0959, "compactness_mean": 0.0926, "concavity_mean": 0.0615,
    "concave points_mean": 0.0335, "symmetry_mean": 0.1792, "fractal_dimension_mean": 0.0615,
    "radius_se": 0.3242, "texture_se": 1.108, "perimeter_se": 2.287, "area_se": 24.53,
    "smoothness_se": 0.0064, "compactness_se": 0.0204, "concavity_se": 0.0259,
    "concave points_se": 0.0109, "symmetry_se": 0.0187, "fractal_dimension_se": 0.0032,
    "radius_worst": 14.97, "texture_worst": 25.41, "perimeter_worst": 97.66, "area_worst": 686.5,
    "smoothness_worst": 0.1313, "compactness_worst": 0.2119, "concavity_worst": 0.2267,
    "concave points_worst": 0.0999, "symmetry_worst": 0.2822, "fractal_dimension_worst": 0.08,
}

GROUPES_BC = [
    ("Valeurs moyennes", [f for f in features_bc if f.endswith("_mean")]),
    ("Erreurs types", [f for f in features_bc if f.endswith("_se")]),
    ("Pires valeurs", [f for f in features_bc if f.endswith("_worst")]),
]

LIBELLES_DB = {
    "Pregnancies": "Nombre de grossesses",
    "Glucose": "Glycémie (mg/dL)",
    "BloodPressure": "Pression artérielle diastolique (mm Hg)",
    "SkinThickness": "Épaisseur du pli cutané du triceps (mm)",
    "Insulin": "Insuline sérique à 2h (mu U/mL)",
    "BMI": "Indice de masse corporelle (IMC)",
    "DiabetesPedigreeFunction": "Antécédents familiaux (indice de pedigree)",
    "Age": "Âge (années)",
}

DEFAUTS_DB = {
    "Pregnancies": 3, "Glucose": 117, "BloodPressure": 72, "SkinThickness": 23,
    "Insulin": 30.5, "BMI": 32.0, "DiabetesPedigreeFunction": 0.37, "Age": 29,
}


@app.route("/")
def accueil():
    return render_template("index.html")


@app.route("/cancer-du-sein", methods=["GET", "POST"])
def cancer_du_sein():
    resultat = None
    valeurs_saisies = DEFAUTS_BC.copy()

    if request.method == "POST":
        try:
            valeurs_saisies = {f: float(request.form.get(
                f, DEFAUTS_BC[f])) for f in features_bc}

            x_new = pd.DataFrame([valeurs_saisies], columns=features_bc)
            x_new_tr = scaler_bc.transform(x_new)

            prediction = model_bc.predict(x_new_tr)[0]
            probas = model_bc.predict_proba(x_new_tr)[0]
            classes = model_bc.classes_
            proba_dict = dict(zip(classes, probas))

            resultat = {
                "diagnostic": "Malignant (cancer détecté)" if prediction == "M" else "Benign (pas de cancer détecté)",
                "est_positif": prediction == "M",
                "confiance": round(max(probas) * 100, 2),
                "proba_M": round(proba_dict.get("M", 0) * 100, 2),
                "proba_B": round(proba_dict.get("B", 0) * 100, 2),
            }
        except (ValueError, TypeError):
            resultat = {
                "erreur": "Veuillez saisir des valeurs numériques valides pour tous les champs."}

    return render_template(
        "cancer_sein.html",
        groupes=GROUPES_BC,
        libelles=LIBELLES_BC,
        valeurs=valeurs_saisies,
        resultat=resultat,
    )


@app.route("/diabete", methods=["GET", "POST"])
def diabete():
    resultat = None
    valeurs_saisies = DEFAUTS_DB.copy()

    if request.method == "POST":
        try:
            valeurs_saisies = {f: float(request.form.get(
                f, DEFAUTS_DB[f])) for f in features_db}

            x_new = pd.DataFrame([valeurs_saisies], columns=features_db)
            x_new_tr = scaler_db.transform(x_new)

            prediction = model_db.predict(x_new_tr)[0]
            probas = model_db.predict_proba(x_new_tr)[0]
            classes = model_db.classes_
            proba_dict = dict(zip(classes, probas))

            resultat = {
                "diagnostic": "Diabète détecté" if prediction == 1 else "Pas de diabète détecté",
                "est_positif": prediction == 1,
                "confiance": round(max(probas) * 100, 2),
                "proba_1": round(proba_dict.get(1, 0) * 100, 2),
                "proba_0": round(proba_dict.get(0, 0) * 100, 2),
            }
        except (ValueError, TypeError):
            resultat = {
                "erreur": "Veuillez saisir des valeurs numériques valides pour tous les champs."}

    return render_template(
        "diabetes.html",
        features=features_db,
        libelles=LIBELLES_DB,
        valeurs=valeurs_saisies,
        resultat=resultat,
    )


if __name__ == "__main__":
    app.run(debug=True)
