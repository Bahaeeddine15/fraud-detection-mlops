"""
dashboard.py — Dashboard de monitoring du modèle de détection de fraude
==========================================================================
Deux onglets :
1. Vue d'ensemble : distribution des scores de risque + indicateur de
   drift, calculés sur des prédictions déjà générées (pas d'appel réseau).
2. Scorer une transaction : formulaire qui appelle l'API déployée sur
   Azure pour scorer une transaction en direct.

Lancer avec : streamlit run dashboard.py
"""

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Monitoring — Détection de fraude", layout="wide")

# --- URL de ton API déployée sur Azure ---
# Remplace par ta propre URL si elle diffère.
API_URL = "https://fraude-api.jollybay-76a27957.swedencentral.azurecontainerapps.io"

st.title("Dashboard de monitoring — Détection de fraude bancaire")

tab1, tab2 = st.tabs(["Vue d'ensemble (test set)", "Scorer une transaction (API live)"])


# =====================================================================
# ONGLET 1 : Vue d'ensemble sur données pré-calculées
# =====================================================================
with tab1:
    try:
        predictions = pd.read_csv("data/dashboard_predictions.csv")
        baseline_stats = joblib.load("baseline_stats.joblib")
    except FileNotFoundError:
        st.error(
            "Fichiers manquants. Lance d'abord :\n"
            "1. `python src/train.py`\n"
            "2. `python src/generate_dashboard_data.py`"
        )
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions analysées", len(predictions))
    col2.metric("Fraudes détectées", int(predictions["predicted_fraud"].sum()))
    col3.metric(
        "Taux de détection (recall)",
        f"{(predictions[predictions['true_class'] == 1]['predicted_fraud'].mean() * 100):.1f}%",
    )

    st.subheader("Distribution des scores de risque")
    st.caption(
        "Un modèle bien calibré doit produire des scores proches de 0 "
        "pour les transactions normales et proches de 1 pour les fraudes."
    )

    # Séparer les scores par vraie classe pour voir si le modèle les
    # distingue bien (deux distributions qui se chevauchent peu = bon signe)
    hist_data = pd.DataFrame({
        "Score de fraude": predictions["fraud_probability"],
        "Classe réelle": predictions["true_class"].map({0: "Normal", 1: "Fraude"}),
    })
    # pd.cut produit des intervalles (type Interval) comme labels de
    # bins. st.bar_chart (basé sur Altair) ne sait pas les afficher
    # directement — on les convertit en texte avec .astype(str).
    hist_data["bin"] = pd.cut(hist_data["Score de fraude"], bins=20).astype(str)

    # Important : on sépare les deux classes en deux graphiques distincts,
    # chacun avec sa propre échelle. Sur UN seul graphique partagé, les
    # 56 864 transactions normales écraseraient visuellement les 98
    # fraudes — la barre des fraudes existerait mais serait invisible.
    col_a, col_b = st.columns(2)

    with col_a:
        st.caption("Transactions normales (échelle : dizaines de milliers)")
        normal_counts = (
            hist_data[hist_data["Classe réelle"] == "Normal"]
            .groupby("bin").size()
        )
        st.bar_chart(normal_counts)

    with col_b:
        st.caption("Fraudes (échelle : dizaines)")
        fraud_counts = (
            hist_data[hist_data["Classe réelle"] == "Fraude"]
            .groupby("bin").size()
        )
        st.bar_chart(fraud_counts)

    st.subheader("Indicateur de drift (simplifié)")
    st.caption(
        "Compare la moyenne de chaque feature dans ce batch à la moyenne "
        "observée à l'entraînement. Un écart important (plusieurs écarts-"
        "types) peut signaler que les données ont changé depuis "
        "l'entraînement, et que le modèle pourrait devenir moins fiable."
    )

    feature_cols = [c for c in predictions.columns
                    if c not in ("true_class", "fraud_probability", "predicted_fraud")]

    drift_rows = []
    for col in feature_cols:
        current_mean = predictions[col].mean()
        baseline_mean = baseline_stats["mean"][col]
        baseline_std = baseline_stats["std"][col]

        # z-score : combien d'écarts-types sépare la moyenne actuelle
        # de la moyenne de référence. > 2 ou < -2 est souvent considéré
        # comme un écart notable (convention statistique courante,
        # pas une règle absolue).
        z_score = (current_mean - baseline_mean) / (baseline_std + 1e-10)
        drift_rows.append({"Feature": col, "Z-score": round(z_score, 3)})

    drift_df = pd.DataFrame(drift_rows).sort_values("Z-score", key=abs, ascending=False)
    n_drifted = (drift_df["Z-score"].abs() > 2).sum()

    if n_drifted > 0:
        st.warning(f"{n_drifted} feature(s) montrent un écart notable (|z-score| > 2).")
    else:
        st.success("Aucun drift notable détecté sur ce batch.")

    st.dataframe(drift_df, use_container_width=True, hide_index=True)


# =====================================================================
# ONGLET 2 : Scorer une transaction via l'API live
# =====================================================================
with tab2:
    st.subheader("Envoyer une transaction à l'API déployée")
    st.caption(f"Appelle directement : {API_URL}/predict")

    # Bouton pour pré-remplir avec un exemple réel du test set,
    # plutôt que de forcer une saisie manuelle de 29 champs.
    if "example" not in st.session_state:
        st.session_state.example = None

    if st.button("Charger un exemple aléatoire du test set"):
        try:
            sample = pd.read_csv("data/dashboard_predictions.csv").sample(1).iloc[0]
            st.session_state.example = sample
        except FileNotFoundError:
            st.error("Génère d'abord data/dashboard_predictions.csv (voir onglet 1).")

    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    values = {}

    cols = st.columns(4)
    for i, name in enumerate(feature_names):
        default = 0.0
        if st.session_state.example is not None:
            default = float(st.session_state.example[name])
        values[name] = cols[i % 4].number_input(name, value=default, format="%.6f")

    if st.button("Envoyer à l'API", type="primary"):
        try:
            response = requests.post(f"{API_URL}/predict", json=values, timeout=10)
            response.raise_for_status()
            result = response.json()

            col1, col2 = st.columns(2)
            col1.metric("Probabilité de fraude", f"{result['fraud_probability']:.4f}")
            col2.metric("Décision", "FRAUDE" if result["is_fraud"] else "Normale")

            if st.session_state.example is not None:
                true_class = st.session_state.example["true_class"]
                st.caption(f"Vraie classe de cet exemple : {'Fraude' if true_class == 1 else 'Normal'}")

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")