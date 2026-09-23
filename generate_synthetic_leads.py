"""
Génération de données synthétiques pour le projet Lead Scoring.

Idée générale :
1. On crée des leads avec des caractéristiques aléatoires (mais pas
   n'importe comment -- certaines valeurs sont plus fréquentes que
   d'autres, comme dans la vraie vie).
2. Pour chaque lead, on calcule un "score caché" en additionnant des
   points selon des règles qu'on connaît déjà (ex: un VP qui vient par
   référencement, c'est un bon signe).
3. On transforme ce score en une probabilité de conversion (entre 0 et 1).
4. On tire au sort le résultat final (converti ou pas), en utilisant
   cette probabilité -- comme une pièce truquée, différente pour
   chaque lead.
"""

import random
import math
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)  # pour avoir toujours les mêmes résultats en relançant le script

NOMBRE_DE_LEADS = 3000


# ---------------------------------------------------------------
# Les valeurs possibles pour chaque champ, avec leur fréquence
# (le nombre à côté de chaque valeur = son "poids" ; plus il est
# grand, plus la valeur a de chances d'être tirée)
# ---------------------------------------------------------------

LEAD_SOURCES = ["Web", "Phone Inquiry", "Partner Referral", "Employee Referral",
                "Purchased List", "Trade Show", "Advertisement", "Public Relations",
                "Webinar", "Social Media", "Email Campaign", "Cold Call"]
POIDS_LEAD_SOURCE = [22, 10, 8, 6, 15, 8, 8, 4, 7, 9, 8, 5]

INDUSTRIES = ["Technology", "Retail", "Finance", "Banking", "Healthcare", "Manufacturing",
              "Education", "Consulting", "Insurance", "Telecommunications", "Biotechnology",
              "Construction", "Energy", "Media", "Government", "Not For Profit",
              "Transportation", "Hospitality", "Food & Beverage", "Chemicals", "Electronics",
              "Engineering", "Entertainment", "Environmental", "Machinery", "Recreation",
              "Shipping", "Utilities", "Apparel", "Communications", "Agriculture"]
POIDS_INDUSTRY = [14, 10, 8, 6, 8, 9, 6, 6, 4, 4, 3, 5, 4, 4, 3, 3, 3, 4, 4, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 2]

TITLES = ["C-Level", "VP", "Director", "Senior Manager", "Manager",
          "Team Lead", "Individual Contributor", "Intern"]
POIDS_TITLE = [3, 7, 15, 12, 25, 13, 20, 5]

COUNTRIES = ["USA", "France", "Morocco", "UK", "Germany", "Canada",
             "Spain", "Italy", "Netherlands", "Belgium", "UAE", "India"]
POIDS_COUNTRY = [26, 14, 10, 12, 10, 8, 6, 5, 4, 3, 4, 8]


def choisir_une_valeur(valeurs, poids):
    """Tire une seule valeur au hasard, en respectant les poids donnés."""
    return random.choices(valeurs, weights=poids, k=1)[0]


# ---------------------------------------------------------------
# Étape 1 : générer les leads (un par un, dans une boucle)
# ---------------------------------------------------------------

def generer_un_lead(id_lead):
    """Crée un seul lead avec des caractéristiques tirées au hasard."""

    # Taille de l'entreprise : la plupart sont petites, quelques-unes
    # sont très grandes. random.lognormvariate imite bien ce genre de
    # distribution (beaucoup de petites valeurs, une longue traîne de
    # grandes valeurs).
    nombre_employes = int(random.lognormvariate(4.5, 1.3))
    nombre_employes = max(1, min(nombre_employes, 50000))  # on garde ça dans une plage raisonnable

    # Revenu annuel : approximativement lié à la taille de l'entreprise,
    # avec du bruit pour rester réaliste
    revenu_par_employe = random.lognormvariate(11.0, 0.6)
    revenu_annuel = round(nombre_employes * revenu_par_employe, -3)

    # Email professionnel ou personnel
    email_type = random.choices(["Professionnel", "Personnel"], weights=[65, 35], k=1)[0]

    # Date de création, quelque part dans les 2 dernières années
    jours_avant_aujourdhui = random.randint(0, 730)
    date_creation = datetime(2026, 9, 17) - timedelta(days=jours_avant_aujourdhui)

    return {
        "Lead_Id": f"00Q{id_lead:07d}",
        "LeadSource": choisir_une_valeur(LEAD_SOURCES, POIDS_LEAD_SOURCE),
        "Industry": choisir_une_valeur(INDUSTRIES, POIDS_INDUSTRY),
        "Title": choisir_une_valeur(TITLES, POIDS_TITLE),
        "Country": choisir_une_valeur(COUNTRIES, POIDS_COUNTRY),
        "NumberOfEmployees": nombre_employes,
        "AnnualRevenue": revenu_annuel,
        "Email_Domain_Type": email_type,
        "CreatedDate": date_creation,
    }


# ---------------------------------------------------------------
# Étape 2 et 3 : calculer le score caché, puis la probabilité
# ---------------------------------------------------------------

# Ces dictionnaires représentent les règles métier qu'on "cache" dans
# les données. Le modèle ne les verra jamais directement -- il devra
# les redécouvrir tout seul à partir des exemples.

POINTS_LEAD_SOURCE = {
    "Employee Referral": 2.2, "Partner Referral": 1.8, "Webinar": 0.6,
    "Trade Show": 0.8, "Public Relations": 0.3, "Web": 0.0,
    "Phone Inquiry": 0.2, "Advertisement": -0.2, "Email Campaign": -0.3,
    "Social Media": -0.4, "Cold Call": -0.6, "Purchased List": -1.5,
}

POINTS_TITLE = {
    "C-Level": 2.4, "VP": 1.8, "Director": 1.2, "Senior Manager": 0.7,
    "Manager": 0.4, "Team Lead": 0.1, "Individual Contributor": -0.3, "Intern": -2.0,
}

SECTEURS_FORT_BUDGET = {"Technology", "Finance", "Banking", "Biotechnology",
                         "Telecommunications", "Energy", "Consulting", "Insurance"}
SECTEURS_FAIBLE_BUDGET = {"Not For Profit", "Government", "Recreation",
                           "Environmental", "Agriculture", "Hospitality"}


def calculer_score(lead):
    """Additionne les points de chaque règle, pour un seul lead."""

    score = 0.0
    score += POINTS_LEAD_SOURCE[lead["LeadSource"]]
    score += POINTS_TITLE[lead["Title"]]
    score += 0.8 if lead["Email_Domain_Type"] == "Professionnel" else -0.8

    # Taille de l'entreprise (effet modéré, sur une échelle logarithmique)
    score += 0.25 * math.log1p(lead["NumberOfEmployees"]) - 1.0

    # Revenu par employé (indicateur de "santé financière")
    revenu_par_employe = lead["AnnualRevenue"] / max(lead["NumberOfEmployees"], 1)
    score += 0.35 * math.log1p(revenu_par_employe) - 4.0

    # Secteur d'activité
    if lead["Industry"] in SECTEURS_FORT_BUDGET:
        score += 0.4
    elif lead["Industry"] in SECTEURS_FAIBLE_BUDGET:
        score -= 0.4

    # Un peu de bruit aléatoire, comme dans la vraie vie
    score += random.gauss(0, 1.0)

    # Décalage global pour obtenir un taux de conversion réaliste (~15-20%)
    score -= 2.3

    return score


def score_vers_probabilite(score):
    """Fonction sigmoïde : transforme n'importe quel score en un nombre entre 0 et 1."""
    return 1 / (1 + math.exp(-score))


# ---------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------

def main():
    leads = []

    for i in range(NOMBRE_DE_LEADS):
        lead = generer_un_lead(i)

        score = calculer_score(lead)
        probabilite = score_vers_probabilite(score)

        # Étape 4 : tirage au sort biaisé -> 1 avec probabilité `probabilite`, sinon 0
        lead["Converted"] = 1 if random.random() < probabilite else 0

        leads.append(lead)

    df = pd.DataFrame(leads)
    df.to_csv("output/synthetic_leads.csv", index=False)

    print(f"{len(df)} leads générés.")
    print(f"Taux de conversion global : {df['Converted'].mean():.1%}")
    print("\nTaux de conversion par LeadSource :")
    print(df.groupby("LeadSource")["Converted"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
