"""
Entraînement du modèle de Lead Scoring (XGBoost).

Étapes :
1. Charger les données
2. Transformer les données en features numériques
3. Séparer en un ensemble d'entraînement et un ensemble de test
4. Entraîner le modèle
5. Évaluer le modèle
6. Sauvegarder le modèle
"""

import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from feature_engineering import construire_features


# --- 1. Charger les données ---

df = pd.read_csv("output/synthetic_leads.csv")
df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])

print(f"{len(df)} leads chargés.")


# --- 2. Transformer les données ---

features = construire_features(df)
label = df["Converted"]

print(f"Nombre de features après encodage : {features.shape[1]}")


# --- 3. Séparer entraînement / test ---
# On trie d'abord par date, pour entraîner sur les leads les plus
# anciens et tester sur les plus récents (on simule le futur).

df = df.sort_values("CreatedDate")
features = features.loc[df.index]
label = label.loc[df.index]

taille_train = int(len(df) * 0.8)

X_train = features.iloc[:taille_train]
X_test = features.iloc[taille_train:]
y_train = label.iloc[:taille_train]
y_test = label.iloc[taille_train:]

print(f"Entraînement : {len(X_train)} leads | Test : {len(X_test)} leads")


# --- 4. Entraîner le modèle ---
# scale_pos_weight compense le fait qu'il y a moins de leads convertis
# (1) que de non-convertis (0).

nb_zeros = (y_train == 0).sum()
nb_uns = (y_train == 1).sum()
ratio_classes = nb_zeros / nb_uns

modele = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=ratio_classes,
    eval_metric="auc",
    random_state=42,
)

modele.fit(X_train, y_train)
print("Modèle entraîné.")


# --- 5. Évaluer le modèle ---

probabilites_test = modele.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probabilites_test)
print(f"AUC-ROC sur le test : {auc:.3f}")

importance = pd.Series(modele.feature_importances_, index=features.columns)
importance = importance.sort_values(ascending=False)
print("\nTop 10 features les plus importantes :")
print(importance.head(10))


# --- 6. Sauvegarder le modèle et la liste des colonnes ---

modele.save_model("output/lead_scoring_model.json")

with open("output/feature_columns.json", "w") as f:
    json.dump(list(features.columns), f, indent=2)

print("\nModèle et colonnes sauvegardés.")
