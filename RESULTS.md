# Rapport de résultats — Détection de fraude bancaire

## 1. Données

- **Source :** Kaggle, Credit Card Fraud Detection Dataset
- **Volume :** 284 807 transactions, 492 fraudes (**0,17%** des transactions)
- **Features :** 28 variables anonymisées par PCA (V1-V28), montant (`Amount`), temps écoulé (`Time`, non utilisée pour l'entraînement final)
- **Split :** 80/20, stratifié sur la classe (`stratify=y`) pour garantir la même proportion de fraudes en train et en test (0,1729% en train, 0,1720% en test)

## 2. Métrique retenue

**AUC-PR** (aire sous la courbe precision-recall), plutôt que l'accuracy — trompeuse sur un dataset aussi déséquilibré (un modèle prédisant toujours "non-fraude" atteindrait 99,83% d'accuracy sans aucune utilité).

Objectif du cahier des charges : **AUC-PR > 0,80**.

## 3. Modèles testés

### 3.1 Régression logistique (baseline)
- `class_weight='balanced'` pour compenser le déséquilibre
- **AUC-PR : 0,7199**
- Au seuil par défaut (0,5) : precision 0,06 / recall 0,92 (beaucoup de faux positifs : 1439)
- Au seuil optimisé (F1 max) : precision 0,83 / recall 0,82 (16 faux positifs, 18 faux négatifs)

### 3.2 XGBoost + `scale_pos_weight` — **modèle retenu**
- `scale_pos_weight` calculé automatiquement (ratio négatifs/positifs, ≈ 577)
- **AUC-PR : 0,8834**
- Au seuil par défaut (0,5) : precision 0,89 / recall 0,83 (10 faux positifs, 17 faux négatifs)
- Au seuil optimisé (F1 max, ≈ 0,98) : precision 0,99 / recall 0,81 (1 seul faux positif sur 56 962 transactions test)

### 3.3 XGBoost + SMOTE (comparaison)
- Sur-échantillonnage synthétique de la classe minoritaire, appliqué uniquement sur le train set
- **AUC-PR : 0,8610** — inférieur à `scale_pos_weight` seul
- Hypothèse : les exemples synthétiques générés par interpolation linéaire sur des features déjà transformées par PCA introduisent du bruit plutôt que de l'information utile

## 4. Comparaison synthétique

| Modèle | AUC-PR | Precision (seuil optimisé) | Recall (seuil optimisé) |
|---|---|---|---|
| Régression logistique | 0,7199 | 0,83 | 0,82 |
| XGBoost + scale_pos_weight | **0,8834** | 0,99 | 0,81 |
| XGBoost + SMOTE | 0,8610 | 0,77 | 0,84 |

## 5. Choix du seuil de décision

Le seuil de classification n'est **pas** une propriété fixe du modèle — c'est un arbitrage business entre faux positifs (clients légitimes bloqués) et faux négatifs (fraudes non détectées) :

- **Seuil F1 max (retenu par défaut) :** precision 0,99 / recall 0,81 — minimise les fausses alertes
- **Seuil pour recall ≥ 0,90 :** trouvé à 0,001, avec une precision qui chute à 0,31 — illustre le compromis extrême si la priorité est de ne rater aucune fraude

Le seuil retenu pour l'API est ajustable selon la politique de risque souhaitée.

## 6. Tracking des expériences

Les 3 modèles ont été trackés avec MLflow (hyperparamètres, AUC-PR, artefact modèle), permettant une comparaison reproductible.

## 7. Validation de l'API

- Testée avec une transaction réelle non-fraude (première ligne du dataset) → `fraud_probability: 0`, `is_fraud: false` ✅
- Testée avec une transaction réelle frauduleuse → `fraud_probability: 1`, `is_fraud: true` ✅
- Comportement identique validé en local, dans le conteneur Docker, et sur Azure Container Apps

## 8. Limites et pistes d'amélioration

- **Dataset public déjà largement étudié** : les performances mesurées sont probablement optimistes par rapport à un déploiement sur des données bancaires réelles, dont la distribution évolue dans le temps (contrairement à ce jeu de données statique).
- **SMOTE n'a pas amélioré les résultats** ici, mais pourrait se comporter différemment avec un tuning plus poussé (variantes comme SMOTE-ENN, ADASYN) — non testé par manque de temps.
- **Le déploiement conditionnel** compare uniquement l'AUC-PR ; une version plus robuste pondérerait plusieurs métriques (recall à precision fixée, latence d'inférence).
- **Pas de test A/B ni de shadow deployment** avant promotion — le nouveau modèle remplace directement l'ancien s'il est meilleur sur le test set, sans validation sur du trafic réel.
