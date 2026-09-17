"""
Étape 7 : API FastAPI de scoring de fraude
=============================================
Objectif : exposer le modèle via un endpoint /predict qui reçoit
une transaction et renvoie une probabilité de fraude + une décision.

Lancer avec : uvicorn app:app --reload
Puis tester sur : http://localhost:8000/docs (interface Swagger auto-générée)
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- Charger le modèle et ses dépendances UNE SEULE FOIS au démarrage ---
# Charger un modèle est coûteux (I/O disque, désérialisation). On le
# fait une fois au lancement de l'API, pas à chaque requête.
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')
threshold = joblib.load('threshold.joblib')

app = FastAPI(
    title="API de détection de fraude bancaire",
    description="Scoring de transactions pour détecter les fraudes potentielles",
    version="1.0.0",
)


# --- Définir la structure attendue en entrée ---
# Pydantic valide automatiquement les types et la présence de chaque
# champ. Si un champ manque ou a le mauvais type, FastAPI renvoie
# une erreur 422 claire, sans qu'on ait à coder cette vérification.
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Montant de la transaction en euros")


# --- Structure de la réponse ---
class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float


@app.get("/")
def root():
    """Endpoint de vérification que l'API tourne bien."""
    return {"status": "API de détection de fraude opérationnelle"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Reçoit une transaction et renvoie une probabilité de fraude.
    """
    # --- Convertir la requête en DataFrame ---
    # .dict() transforme l'objet Pydantic en dictionnaire Python
    data = pd.DataFrame([transaction.dict()])

    # --- Appliquer la MÊME transformation que pendant l'entraînement ---
    # C'est un point critique : si on oublie cette étape, le modèle
    # reçoit un Amount à une échelle différente de celle apprise,
    # et les prédictions seront silencieusement fausses.
    data['Amount'] = scaler.transform(data[['Amount']])

    # --- Remettre les colonnes dans le bon ordre ---
    data = data[feature_columns]

    # --- Prédire ---
    fraud_proba = float(model.predict_proba(data)[:, 1][0])
    is_fraud = bool(fraud_proba >= threshold)

    return PredictionResponse(
        fraud_probability=round(fraud_proba, 4),
        is_fraud=is_fraud,
        threshold_used=threshold,
    )