# --- Image de base ---
# "slim" = version allégée de Python (sans les outils inutiles pour
# une prod), plus rapide à télécharger et plus légère à faire tourner.
FROM python:3.11-slim

# --- Dossier de travail à l'intérieur du conteneur ---
# Toutes les commandes suivantes s'exécutent depuis ce dossier.
WORKDIR /app

# --- Copier et installer les dépendances D'ABORD ---
# Astuce importante : on copie requirements.txt avant le reste du code.
# Docker met en cache chaque étape ("layer"). Si seul ton code change
# (pas les dépendances), Docker réutilise le cache de pip install
# au lieu de tout réinstaller à chaque build — ça accélère énormément
# les reconstructions pendant le développement.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copier le code de l'application ---
# app.py est maintenant dans src/, on préserve cette structure dans
# l'image pour que l'import fonctionne pareil qu'en local.
COPY src/app.py ./src/app.py

# --- Copier les artefacts du modèle ---
# Ils restent à la racine du conteneur (générés par train.py à la
# racine du projet également).
COPY model.joblib .
COPY scaler.joblib .
COPY feature_columns.joblib .
COPY threshold.joblib .

# --- Exposer le port utilisé par l'API ---
# Ça documente le port utilisé ; ça n'ouvre rien tout seul, il faudra
# quand même le mapper explicitement avec "docker run -p".
EXPOSE 8000

# --- Commande de démarrage du conteneur ---
# --host 0.0.0.0 est ESSENTIEL : sans ça, l'API n'écoute que sur
# localhost DANS le conteneur, et serait invisible depuis l'extérieur
# (y compris depuis ta machine hôte).
# "src.app:app" car app.py est dans le sous-dossier src/.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]