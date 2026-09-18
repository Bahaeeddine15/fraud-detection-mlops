"""
compare_and_promote.py — Déploiement conditionnel (F9)
=========================================================
Compare l'AUC-PR du modèle candidat (juste entraîné) à celui
actuellement en production. Ne "promeut" le candidat (= le renomme
pour devenir la version servie par l'API) que s'il est meilleur.

production_metrics.json et les fichiers model.joblib etc. ne sont PAS
dans Git (trop volumineux / générés) : ils persistent uniquement dans
le workspace Jenkins d'un build à l'autre (Jenkins ne nettoie pas le
workspace par défaut entre deux builds du même job).

Code de sortie :
- 0 si le candidat est promu (le pipeline continue vers build + déploiement)
- 1 si le candidat est rejeté (le pipeline s'arrête ici, rien n'est déployé)
"""

import json
import os
import shutil

CANDIDATE_METRICS = "candidate_metrics.json"
PRODUCTION_METRICS = "production_metrics.json"

# Fichiers à "promouvoir" : (nom du candidat, nom de production attendu par app.py/Dockerfile)
ARTIFACTS = [
    ("candidate_model.joblib", "model.joblib"),
    ("candidate_scaler.joblib", "scaler.joblib"),
    ("candidate_feature_columns.joblib", "feature_columns.joblib"),
    ("candidate_threshold.joblib", "threshold.joblib"),
    ("candidate_baseline_stats.joblib", "baseline_stats.joblib"),
]


def main():
    with open(CANDIDATE_METRICS) as f:
        candidate = json.load(f)

    candidate_auc_pr = candidate["auc_pr"]

    # --- Récupérer la performance de la version en production ---
    # Si aucun fichier production_metrics.json n'existe encore (tout
    # premier déploiement), on considère la barre de départ à 0 :
    # le premier modèle valide est toujours promu.
    if os.path.exists(PRODUCTION_METRICS):
        with open(PRODUCTION_METRICS) as f:
            production = json.load(f)
        production_auc_pr = production["auc_pr"]
    else:
        production_auc_pr = 0.0
        print("Aucune version en production trouvée (premier déploiement).")

    print(f"AUC-PR candidat    : {candidate_auc_pr:.4f}")
    print(f"AUC-PR production  : {production_auc_pr:.4f}")

    if candidate_auc_pr > production_auc_pr:
        print("\nCandidat MEILLEUR que la production -> PROMOTION")

        for candidate_name, production_name in ARTIFACTS:
            shutil.copy(candidate_name, production_name)

        with open(PRODUCTION_METRICS, "w") as f:
            json.dump(candidate, f, indent=2)

        print("Fichiers de production mis à jour. Le build Docker peut continuer.")
        exit(0)
    else:
        print("\nCandidat PAS meilleur que la production -> REJET")
        print("Aucun fichier de production modifié. Build/déploiement annulés.")
        exit(1)


if __name__ == "__main__":
    main()
