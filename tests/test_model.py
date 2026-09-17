"""
tests/test_model.py
=====================
Tests automatisés pour le pipeline de détection de fraude.

Important : ces tests utilisent des données SYNTHÉTIQUES, pas le vrai
dataset Kaggle (qui n'est pas dans le dépôt Git, trop volumineux).
L'objectif ici n'est PAS de vérifier la performance réelle du modèle,
mais de vérifier que le CODE fonctionne correctement : pas d'erreur,
bonnes dimensions, comportements attendus. C'est ce qu'on appelle des
tests unitaires, par opposition à une évaluation de performance.

Lancer avec : pytest tests/
"""

import sys
import os

# Permet d'importer src.train et src.app même si pytest est lancé
# depuis un autre dossier que la racine du projet.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.train import prepare_data, train_model, compute_optimal_threshold


def make_fake_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Génère un dataset synthétique avec la même structure que le vrai
    (colonnes V1-V28, Amount, Time, Class) et un déséquilibre de classes
    comparable, pour tester le pipeline sans le vrai fichier CSV.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=10,
        n_redundant=5,
        weights=[0.98, 0.02],  # déséquilibre volontaire, pas 0.17% exactement
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = np.random.uniform(0, 500, size=n_samples)
    df["Time"] = np.arange(n_samples)
    df["Class"] = y

    return df


# --- Tests sur la préparation des données ---

def test_prepare_data_shapes():
    """Vérifie que le split produit les bonnes dimensions et colonnes."""
    df = make_fake_dataset()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # 80/20 comme configuré dans train.py
    assert len(X_train) + len(X_test) == len(df)
    assert abs(len(X_test) / len(df) - 0.2) < 0.01

    # Time doit avoir été supprimée, Amount doit rester
    assert "Time" not in X_train.columns
    assert "Amount" in X_train.columns


def test_prepare_data_stratification():
    """Vérifie que le split préserve la proportion de fraudes (stratify=y)."""
    df = make_fake_dataset()
    _, _, y_train, y_test, _ = prepare_data(df)

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Les deux proportions doivent être très proches l'une de l'autre
    assert abs(train_ratio - test_ratio) < 0.02


# --- Tests sur l'entraînement du modèle ---

def test_train_model_returns_fitted_model():
    """Vérifie que le modèle s'entraîne et peut prédire des probabilités."""
    df = make_fake_dataset()
    X_train, X_test, y_train, y_test, _ = prepare_data(df)

    model = train_model(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # predict_proba doit renvoyer une probabilité par ligne de test,
    # toutes comprises entre 0 et 1
    assert len(y_proba) == len(X_test)
    assert (y_proba >= 0).all() and (y_proba <= 1).all()


# --- Tests sur le calcul du seuil ---

def test_compute_optimal_threshold_range():
    """Vérifie que le seuil calculé est une probabilité valide (entre 0 et 1)."""
    df = make_fake_dataset()
    X_train, X_test, y_train, y_test, _ = prepare_data(df)
    model = train_model(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold, auc_pr = compute_optimal_threshold(y_test, y_proba)

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= auc_pr <= 1.0


# --- Tests sur l'API ---
# Ces tests utilisent les VRAIS artefacts (model.joblib, etc.) s'ils
# existent sur la machine qui exécute les tests. S'ils sont absents
# (ex: premier run de Jenkins avant tout entraînement), on saute ces
# tests plutôt que de les faire échouer — c'est ce que fait pytest.mark.skipif.

ARTIFACTS_EXIST = all(
    os.path.exists(f) for f in
    ["model.joblib", "scaler.joblib", "feature_columns.joblib", "threshold.joblib"]
)


@pytest.mark.skipif(not ARTIFACTS_EXIST, reason="Artefacts du modèle non trouvés")
def test_api_root_endpoint():
    """Vérifie que l'API démarre et répond sur la route racine."""
    from fastapi.testclient import TestClient
    from src.app import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200


@pytest.mark.skipif(not ARTIFACTS_EXIST, reason="Artefacts du modèle non trouvés")
def test_api_predict_endpoint():
    """Vérifie que /predict répond correctement à une requête valide."""
    from fastapi.testclient import TestClient
    from src.app import app

    client = TestClient(app)

    # Transaction factice, juste pour vérifier que l'API répond
    # correctement à la BONNE structure, pas pour tester la précision.
    fake_transaction = {f"V{i}": 0.0 for i in range(1, 29)}
    fake_transaction["Amount"] = 100.0

    response = client.post("/predict", json=fake_transaction)

    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0