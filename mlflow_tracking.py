import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --- Fixer explicitement où MLflow stocke ses données ---
# On ne laisse plus MLflow choisir un emplacement par défaut (source
# du bug qu'on a eu) : on force un chemin précis, dans le dossier
# courant. Lance TOUJOURS ce script ET 'mlflow ui' depuis ce même
# dossier pour qu'ils utilisent le même fichier.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# --- Nommer l'expérience ---
# Une "expérience" MLflow regroupe plusieurs "runs" liés au même
# objectif. Tous nos essais de détection de fraude vont dedans.
mlflow.set_experiment("fraude-bancaire-detection")


# --- Recharger et préparer les données (pour un script autonome) ---
df = pd.read_csv('data/creditcard.csv')
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])
X_train = X_train.drop(columns=['Time'])
X_test = X_test.drop(columns=['Time'])


# =====================================================================
# RUN 1 : Régression logistique (baseline)
# =====================================================================
with mlflow.start_run(run_name="logistic_regression_baseline"):
    params = {"class_weight": "balanced", "max_iter": 1000}

    model_lr = LogisticRegression(**params, random_state=42)
    model_lr.fit(X_train, y_train)

    y_proba_lr = model_lr.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_proba_lr)

    # log_param enregistre un hyperparamètre, log_metric un résultat
    mlflow.log_params(params)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_param("model_type", "LogisticRegression")

    # Sauvegarde le modèle lui-même comme artefact réutilisable
    mlflow.sklearn.log_model(model_lr, "model")

    print(f"[Run 1] Logistic Regression — AUC-PR : {auc_pr:.4f}")


# =====================================================================
# RUN 2 : XGBoost + scale_pos_weight
# =====================================================================
with mlflow.start_run(run_name="xgboost_scale_pos_weight"):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 200,
        "max_depth": 5,
    }

    model_xgb = XGBClassifier(**params, eval_metric='aucpr', random_state=42)
    model_xgb.fit(X_train, y_train)

    y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_proba_xgb)

    mlflow.log_params(params)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("imbalance_technique", "scale_pos_weight")

    mlflow.xgboost.log_model(model_xgb, "model")

    print(f"[Run 2] XGBoost + scale_pos_weight — AUC-PR : {auc_pr:.4f}")


# =====================================================================
# RUN 3 : XGBoost + SMOTE
# =====================================================================
with mlflow.start_run(run_name="xgboost_smote"):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    params = {"n_estimators": 200, "max_depth": 5}

    model_xgb_smote = XGBClassifier(**params, eval_metric='aucpr', random_state=42)
    model_xgb_smote.fit(X_train_smote, y_train_smote)

    y_proba_smote = model_xgb_smote.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_proba_smote)

    mlflow.log_params(params)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("imbalance_technique", "SMOTE")

    mlflow.xgboost.log_model(model_xgb_smote, "model")

    print(f"[Run 3] XGBoost + SMOTE — AUC-PR : {auc_pr:.4f}")


print("\nTous les runs sont enregistrés. Lance 'mlflow ui' dans le terminal")
print("puis ouvre http://localhost:5000 pour comparer les runs visuellement.")