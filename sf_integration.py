"""
Intégration Salesforce <-> modèle de Lead Scoring.

Étapes :
1. Se connecter à Salesforce (OAuth)
2. Extraire les leads non convertis (SOQL)
3. Préparer les données pour le modèle
4. Calculer le score et le tier
5. Écrire les résultats dans Salesforce
"""

import json
import os
import requests
import pandas as pd
from simple_salesforce import Salesforce
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv()
from feature_engineering import construire_features, aligner_sur_colonnes_entrainement, parser_title


# --- Identifiants (à remplacer par les tiens) ---

SF_DOMAIN = os.getenv("SF_DOMAIN")
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")

DOMAINES_PERSONNELS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]


# --- 1. Connexion à Salesforce ---

def se_connecter():
    url_token = f"https://{SF_DOMAIN}.my.salesforce.com/services/oauth2/token"
    reponse = requests.post(url_token, data={
        "grant_type": "client_credentials",
        "client_id": CONSUMER_KEY,
        "client_secret": CONSUMER_SECRET,
    })
    reponse.raise_for_status()
    donnees = reponse.json()

    sf = Salesforce(instance_url=donnees["instance_url"], session_id=donnees["access_token"])
    print("Connecté à Salesforce.")
    return sf


# --- 2. Extraction des leads ---

def extraire_leads(sf):
    requete = """
        SELECT Id, LeadSource, Industry, Title, NumberOfEmployees,
               AnnualRevenue, Country, Email, CreatedDate, Status
        FROM Lead
        WHERE IsConverted = false
    """
    resultat = sf.query_all(requete)
    leads = pd.DataFrame(resultat["records"])
    leads = leads.drop(columns="attributes")

    print(f"{len(leads)} leads extraits.")
    return leads


# --- 3. Préparer les données ---

def classifier_email(email):
    if pd.isna(email) or "@" not in str(email):
        return "Professionnel"
    domaine = str(email).split("@")[-1].lower()
    if domaine in DOMAINES_PERSONNELS:
        return "Personnel"
    else:
        return "Professionnel"


def preparer_donnees(leads):
    leads["LeadSource"] = leads["LeadSource"].fillna("Web")
    leads["Industry"] = leads["Industry"].fillna("Technology")
    leads["Country"] = leads["Country"].fillna("USA")
    leads["NumberOfEmployees"] = leads["NumberOfEmployees"].fillna(0)
    leads["AnnualRevenue"] = leads["AnnualRevenue"].fillna(0)

    # Title est un texte libre -> on le convertit en niveau hiérarchique
    leads["Title"] = leads["Title"].apply(parser_title)

    # Email -> Professionnel / Personnel
    leads["Email_Domain_Type"] = leads["Email"].apply(classifier_email)

    return leads


# --- 4. Scoring ---

def assigner_tier(score):
    if score >= 80:
        return "Hot"
    elif score >= 50:
        return "Warm"
    elif score >= 20:
        return "Cool"
    else:
        return "Cold"


def scorer_leads(leads):
    modele = XGBClassifier()
    modele.load_model("output/lead_scoring_model.json")

    with open("output/feature_columns.json") as f:
        colonnes_entrainement = json.load(f)

    features = construire_features(leads)
    features = aligner_sur_colonnes_entrainement(features, colonnes_entrainement)

    probabilites = modele.predict_proba(features)[:, 1]
    leads["Lead_Score__c"] = (probabilites * 100).round().astype(int)
    leads["Lead_Score_Tier__c"] = leads["Lead_Score__c"].apply(assigner_tier)

    return leads


# --- 5. Écriture des résultats ---

def ecrire_resultats(sf, leads):
    mises_a_jour = leads[["Id", "Lead_Score__c", "Lead_Score_Tier__c"]].to_dict("records")
    resultat = sf.bulk.Lead.update(mises_a_jour)

    nb_succes = 0
    for r in resultat:
        if r["success"]:
            nb_succes += 1

    print(f"{nb_succes} / {len(mises_a_jour)} leads mis à jour avec succès.")


# --- Programme principal ---

if __name__ == "__main__":
    sf = se_connecter()
    leads = extraire_leads(sf)

    if len(leads) == 0:
        print("Aucun lead à scorer.")
    else:
        leads = preparer_donnees(leads)
        leads = scorer_leads(leads)
        ecrire_resultats(sf, leads)

        print("\nAperçu des leads scorés :")
        colonnes_a_afficher = ["Id", "LeadSource", "Title", "Lead_Score__c", "Lead_Score_Tier__c"]
        print(leads[colonnes_a_afficher].to_string(index=False))
