"""
Feature engineering : transforme les leads bruts en valeurs numériques
que le modèle peut comprendre.

Ce fichier est utilisé à la fois par train_model.py (entraînement) et
predict.py / sf_integration.py (scoring), pour être sûr d'appliquer
exactement la même transformation des deux côtés.
"""

import pandas as pd

COLONNES_CATEGORIELLES = ["LeadSource", "Industry", "Title", "Country", "Email_Domain_Type"]
COLONNES_NUMERIQUES = ["NumberOfEmployees", "AnnualRevenue"]


def construire_features(df):
    """
    Transforme les colonnes texte (LeadSource, Title, ...) en colonnes
    0/1 (one-hot encoding), et garde les colonnes numériques telles
    quelles.
    """
    features = pd.get_dummies(df[COLONNES_CATEGORIELLES], prefix=COLONNES_CATEGORIELLES)

    for colonne in COLONNES_NUMERIQUES:
        features[colonne] = df[colonne]

    return features


def aligner_sur_colonnes_entrainement(features, colonnes_entrainement):
    """
    Remet le tableau de features dans le même format que celui utilisé
    à l'entraînement : mêmes colonnes, même ordre.

    - une colonne présente à l'entraînement mais absente ici -> ajoutée,
      remplie de 0
    - une colonne présente ici mais jamais vue à l'entraînement ->
      supprimée
    """
    return features.reindex(columns=colonnes_entrainement, fill_value=0)


# ---------------------------------------------------------------
# Title est un champ TEXTE LIBRE dans Salesforce ("VP of Sales",
# "Senior Account Manager"...), pas un picklist. On doit donc deviner
# le niveau hiérarchique à partir de mots-clés dans le texte.
# ---------------------------------------------------------------

def parser_title(titre_brut):
    """Déduit le niveau hiérarchique à partir d'un intitulé de poste réel."""

    if pd.isna(titre_brut) or str(titre_brut).strip() == "":
        return "Individual Contributor"

    titre = str(titre_brut).lower()

    if "chief" in titre or "ceo" in titre or "cto" in titre or "cfo" in titre or "president" in titre:
        return "C-Level"
    elif "vp" in titre or "vice president" in titre:
        return "VP"
    elif "director" in titre or "directeur" in titre or "directrice" in titre:
        return "Director"
    elif "senior manager" in titre:
        return "Senior Manager"
    elif "manager" in titre:
        return "Manager"
    elif "lead" in titre:
        return "Team Lead"
    elif "intern" in titre or "stagiaire" in titre or "student" in titre:
        return "Intern"
    else:
        return "Individual Contributor"
