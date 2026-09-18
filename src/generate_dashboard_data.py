"""
generate_dashboard_data.py
=============================
Score le test set avec le modèle entraîné et sauvegarde les résultats
dans un CSV, pour que le dashboard Streamlit les affiche sans avoir
à réentraîner ou recharger le modèle à chaque ouverture.

Usage : python src/generate_dashboard_data.py
(depuis la racine du projet, après avoir lancé train.py)
"""

import joblib
import pandas as pd
from train import load_data, prepare_data


def main():
    print("Chargement des données et du modèle...")
    df = load_data()
    X_train, X_test, y_train, y_test, _ = prepare_data(df)

    model = joblib.load("model.joblib")
    threshold = joblib.load("threshold.joblib")

    print("Scoring du test set...")
    fraud_proba = model.predict_proba(X_test)[:, 1]
    is_fraud_pred = (fraud_proba >= threshold).astype(int)

    # On garde les features (pour le calcul de drift) + les résultats
    results = X_test.copy()
    results["true_class"] = y_test.values
    results["fraud_probability"] = fraud_proba
    results["predicted_fraud"] = is_fraud_pred

    results.to_csv("data/dashboard_predictions.csv", index=False)
    print(f"Sauvegardé : data/dashboard_predictions.csv ({len(results)} lignes)")


if __name__ == "__main__":
    main()
