"""
Scoring de leads : charge le modèle entraîné et calcule un score pour
chaque lead.
"""

import json
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import construire_features, aligner_sur_colonnes_entrainement


def assigner_tier(score):
    """Convertit un score numérique (0-100) en catégorie lisible."""
    if score >= 80:
        return "Hot"
    elif score >= 50:
        return "Warm"
    elif score >= 20:
        return "Cool"
    else:
        return "Cold"


# --- Charger le modèle et la liste des colonnes ---

modele = XGBClassifier()
modele.load_model("outputs/lead_scoring_model.json")

with open("output/feature_columns.json") as f:
    colonnes_entrainement = json.load(f)


# --- Charger les leads à scorer ---

leads = pd.read_csv("output/synthetic_leads.csv")
print(f"{len(leads)} leads à scorer.")


# --- Feature engineering (même logique qu'à l'entraînement) ---

features = construire_features(leads)
features = aligner_sur_colonnes_entrainement(features, colonnes_entrainement)


# --- Calculer le score et le tier ---

probabilites = modele.predict_proba(features)[:, 1]
leads["Lead_Score"] = (probabilites * 100).round().astype(int)
leads["Lead_Score_Tier"] = leads["Lead_Score"].apply(assigner_tier)


# --- Sauvegarder le résultat ---

resultats = leads[["Lead_Id", "LeadSource", "Title", "Lead_Score", "Lead_Score_Tier"]]
resultats = resultats.sort_values("Lead_Score", ascending=False)
resultats.to_csv("output/leads_scored.csv", index=False)

print("\nTop 10 des leads les mieux notés :")
print(resultats.head(10).to_string(index=False))

print("\nRépartition par tier :")
print(leads["Lead_Score_Tier"].value_counts())
