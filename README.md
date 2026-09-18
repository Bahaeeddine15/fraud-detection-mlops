# Détection de fraude bancaire — Système MLOps

Projet portfolio démontrant une chaîne MLOps complète : entraînement, tracking d'expériences, API de scoring, conteneurisation, CI/CD, déploiement cloud et monitoring.

**Auteur :** Bahae Eddine Mekrane
**Dataset :** [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284 807 transactions, 492 fraudes (~0,17%)

---

## Architecture

```
Données (Kaggle CSV)
      │
      ▼
src/train.py (XGBoost + scale_pos_weight)
      │
      ▼
Comparaison au modèle en production (src/compare_and_promote.py)
      │
      ▼
API de scoring (FastAPI, src/app.py) ──► Conteneurisation (Docker)
      │
      ▼
Pipeline CI/CD (Jenkins) : tests → ré-entraînement → comparaison → build conditionnel
      │
      ▼
Déploiement (Azure Container Apps)
      │
      ▼
Dashboard de monitoring (Streamlit) : scores de risque, drift des features
```

**Stack :** Python, scikit-learn, XGBoost, MLflow, FastAPI, Docker, Jenkins, Azure Container Apps, Streamlit.

---

## Résultats clés

Voir [RESULTS.md](./RESULTS.md) pour le détail complet des expériences et métriques.

- **Modèle retenu :** XGBoost + `scale_pos_weight`
- **AUC-PR :** 0,8834 (objectif du cahier des charges : > 0,80)
- **Seuil de décision :** optimisé par F1-score (precision ≈ 0,99, recall ≈ 0,81)

---

## Structure du projet

```
├── src/
│   ├── train.py                  # Entraînement + sauvegarde du modèle candidat
│   ├── compare_and_promote.py    # Déploiement conditionnel (F9)
│   ├── app.py                    # API FastAPI (/predict)
│   └── generate_dashboard_data.py
├── tests/
│   └── test_model.py             # Tests unitaires (données synthétiques)
├── dashboard.py                  # Dashboard Streamlit
├── Dockerfile
├── Jenkinsfile
├── requirements.txt              # Dépendances de production/CI (minimal)
├── requirements-dev.txt          # Dépendances de dev local (notebook, MLflow, etc.)
└── Notebook.ipynb                 # Exploration initiale (non utilisé en production)
```

---

## Installation et lancement en local

### 1. Prérequis
- Python 3.11+
- Docker Desktop
- Le fichier `data/creditcard.csv` (téléchargé depuis Kaggle, à placer dans `data/`)

### 2. Environnement
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Entraîner le modèle
```bash
python src/train.py
python src/compare_and_promote.py   # promeut le candidat en "production" locale
```
Génère `model.joblib`, `scaler.joblib`, `feature_columns.joblib`, `threshold.joblib`, `baseline_stats.joblib`.

### 4. Lancer l'API
```bash
uvicorn src.app:app --reload
```
Interface Swagger : http://localhost:8000/docs

### 5. Lancer les tests
```bash
pytest tests/ -v
```

### 6. Construire et lancer le conteneur Docker
```bash
docker build -t fraude-api .
docker run -p 8000:8000 fraude-api
```

### 7. Lancer le dashboard de monitoring
```bash
python src/generate_dashboard_data.py
streamlit run dashboard.py
```

---

## Pipeline CI/CD (Jenkins)

Le `Jenkinsfile` définit un pipeline à 6 stages :
1. **Checkout SCM** — récupère le code depuis GitHub
2. **Préparer les données** — copie le dataset depuis un volume monté
3. **Installer les dépendances et tester** — `pytest` dans un environnement Python isolé
4. **Ré-entraîner le modèle (candidat)** — génère un nouveau modèle, sans l'exposer encore
5. **Comparer et promouvoir (F9)** — compare l'AUC-PR du candidat à la version en production ; ne promeut que s'il est strictement meilleur
6. **Construire l'image Docker** — **exécuté uniquement si le candidat a été promu** à l'étape précédente

Jenkins tourne en local via Docker, avec le socket Docker hôte monté pour permettre les builds d'image depuis le pipeline.

---

## Déploiement

L'API est déployée sur **Azure Container Apps** (région Sweden Central), via **Azure Container Registry** pour l'image Docker.

---

## Limites connues

- Le dataset est un jeu de données public déjà largement étudié ; les métriques obtenues sont optimistes par rapport à un cas réel en production bancaire.
- Le déploiement conditionnel (F9) compare uniquement l'AUC-PR ; une version plus avancée pourrait intégrer plusieurs métriques (recall à precision fixée, latence, etc.).
- Pas d'authentification sur l'API ni de haute disponibilité — hors périmètre du cahier des charges initial (projet portfolio, pas un système de production réel).
