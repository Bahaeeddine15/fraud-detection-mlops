"""
train.py — Script d'entraînement consolidé
=============================================
Regroupe toute la logique validée dans le notebook : chargement des
données, split, scaling, entraînement XGBoost, calcul du seuil optimal,
et sauvegarde des artefacts nécessaires à l'API.

Usage : python src/train.py
(à exécuter depuis la racine du projet, car le chemin vers data/
est relatif à la racine)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Charge le dataset brut."""
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    """Split stratifié + mise à l'échelle de Amount + suppression de Time."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    X_train = X_train.drop(columns=["Time"])
    X_test = X_test.drop(columns=["Time"])

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train) -> XGBClassifier:
    """Entraîne le modèle retenu : XGBoost + scale_pos_weight."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_estimators=200,
        max_depth=5,
    )
    model.fit(X_train, y_train)
    return model


def compute_optimal_threshold(y_test, y_proba) -> tuple[float, float]:
    """Calcule le seuil qui maximise le F1-score, et renvoie aussi l'AUC-PR."""
    auc_pr = average_precision_score(y_test, y_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)

    return float(thresholds[best_idx]), auc_pr


def main():
    print("Chargement des données...")
    df = load_data()

    print("Préparation des données (split + scaling)...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    print("Entraînement du modèle XGBoost...")
    model = train_model(X_train, y_train)

    print("Évaluation et calcul du seuil optimal...")
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold, auc_pr = compute_optimal_threshold(y_test, y_proba)

    print(f"\nAUC-PR obtenu : {auc_pr:.4f}")
    print(f"Seuil optimal (F1 max) : {threshold:.4f}")

    # --- Critère de succès du cahier des charges : AUC-PR > 0.80 ---
    # exit(1) fait échouer le script avec un code d'erreur non-nul.
    # C'est ce signal que Jenkins utilisera plus tard pour décider
    # d'arrêter le pipeline si le modèle ne passe pas la barre (F9).
    if auc_pr <= 0.80:
        print("ÉCHEC : AUC-PR en dessous du seuil requis de 0.80")
        exit(1)

    print("\nSauvegarde des artefacts...")
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(X_train.columns.tolist(), "feature_columns.joblib")
    joblib.dump(threshold, "threshold.joblib")

    print("Terminé. Artefacts sauvegardés : model.joblib, scaler.joblib, "
          "feature_columns.joblib, threshold.joblib")


if __name__ == "__main__":
    main()