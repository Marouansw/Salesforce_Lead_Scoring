# Lead Scoring

This project builds a machine learning lead scoring pipeline for Salesforce leads.
It can generate synthetic lead data, train an XGBoost model, score leads, and push
scores back into Salesforce custom fields.

## What The Project Does

The project estimates how likely each lead is to convert. It returns:

- `Lead_Score`: a numeric score from 0 to 100
- `Lead_Score_Tier`: a readable category: `Hot`, `Warm`, `Cool`, or `Cold`

The model is trained on synthetic data that imitates Salesforce lead attributes
such as source, industry, title, country, company size, annual revenue, and email
domain type.

## Project Architecture

```text
Lead_Scoring/
|-- generate_synthetic_leads.py      # Creates synthetic training data
|-- feature_engineering.py           # Shared feature preparation logic
|-- train_model.py                   # Trains and saves the XGBoost model
|-- predict.py                       # Scores leads from a CSV file
|-- sf_integration.py                # Reads/writes lead scores in Salesforce
|-- .env.exemple                     # Environment variable template
|-- .env                             # Local Salesforce credentials, not for sharing
|-- output/
|   |-- synthetic_leads.csv          # Generated training data
|   |-- lead_scoring_model.json      # Trained XGBoost model
|   `-- feature_columns.json         # Training feature column order
`-- Leads_sf/
    `-- Leads_to_import.csv          # Salesforce import/export lead data
```

## Data Flow

```text
Synthetic lead generation
        |
        v
output/synthetic_leads.csv
        |
        v
Feature engineering
        |
        v
XGBoost training
        |
        v
output/lead_scoring_model.json
output/feature_columns.json
        |
        v
Lead scoring
        |
        v
Scores written to CSV or Salesforce
```

## Main Components

### `generate_synthetic_leads.py`

Generates 3,000 synthetic leads and saves them to:

```text
output/synthetic_leads.csv
```

The script creates realistic-looking lead attributes using weighted random
choices. It also creates a hidden conversion probability and then assigns each
lead a `Converted` value of `0` or `1`.

Important generated columns include:

- `Lead_Id`
- `LeadSource`
- `Industry`
- `Title`
- `Country`
- `NumberOfEmployees`
- `AnnualRevenue`
- `Email_Domain_Type`
- `CreatedDate`
- `Converted`

### `feature_engineering.py`

Contains shared transformation logic used by training, prediction, and
Salesforce scoring.

It does three main things:

- Converts categorical fields into one-hot encoded columns
- Keeps numeric fields such as employees and revenue
- Aligns scoring-time features with the exact columns used during training

It also includes `parser_title()`, which converts real Salesforce free-text job
titles such as `VP of Sales` or `Chief Technology Officer` into model-friendly
title levels such as `VP` or `C-Level`.

### `train_model.py`

Trains an `XGBClassifier` model using the generated synthetic data.

Training steps:

1. Load `output/synthetic_leads.csv`
2. Build numeric model features
3. Sort leads by `CreatedDate`
4. Train on the oldest 80 percent of leads
5. Test on the newest 20 percent of leads
6. Print the test AUC-ROC score
7. Save the model and feature columns

Generated files:

```text
output/lead_scoring_model.json
output/feature_columns.json
```

### `predict.py`

Loads a trained model, scores leads, assigns a tier, and writes scored results
to a CSV file.

Score tiers:

| Score Range | Tier |
| --- | --- |
| 80-100 | Hot |
| 50-79 | Warm |
| 20-49 | Cool |
| 0-19 | Cold |

When running locally from this project folder, make sure all model, feature,
input, and output paths in this file point to the local `output/...` folder.

### `sf_integration.py`

Connects to Salesforce, extracts unconverted leads, scores them, and updates
Salesforce with two custom fields:

```text
Lead_Score__c
Lead_Score_Tier__c
```

Salesforce flow:

1. Connect to Salesforce using OAuth client credentials
2. Query non-converted leads
3. Fill missing values
4. Parse job titles into standard title levels
5. Classify email domains as `Professionnel` or `Personnel`
6. Run the trained XGBoost model
7. Update Salesforce leads using the Bulk API

## Requirements

Use Python 3.10 or newer.

Install the required Python packages:

```bash
pip install pandas scikit-learn xgboost requests simple-salesforce python-dotenv
```

## Setup

### 1. Clone or open the project

Open a terminal in the project folder:

```bash
cd Lead_Scoring
```

### 2. Create the output folder

The repository already contains an `output` folder. If it does not exist, create
it:

```bash
mkdir output
```

### 3. Configure Salesforce credentials

Copy `.env.exemple` to `.env`:

```bash
cp .env.exemple .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.exemple .env
```

Fill in the values:

```env
SF_DOMAIN=your-domain
CONSUMER_KEY=your-connected-app-consumer-key
CONSUMER_SECRET=your-connected-app-consumer-secret
```

`SF_DOMAIN` is the Salesforce My Domain prefix. For example, if your Salesforce
URL is:

```text
https://example-dev-ed.my.salesforce.com
```

then:

```env
SF_DOMAIN=example-dev-ed
```

Do not commit `.env` because it contains credentials.

## How To Run The Full Pipeline

### Step 1. Generate synthetic leads

```bash
python generate_synthetic_leads.py
```

Expected output:

- A new or updated `output/synthetic_leads.csv`
- Printed conversion rate summary
- Conversion rate by lead source

### Step 2. Train the model

```bash
python train_model.py
```

Expected output:

- Number of loaded leads
- Number of encoded features
- Train/test split size
- AUC-ROC score
- Top feature importances
- Saved model files

Generated files:

```text
output/lead_scoring_model.json
output/feature_columns.json
```

### Step 3. Score leads locally

Before running locally, make sure `predict.py` uses these local paths:

```text
output/lead_scoring_model.json
output/feature_columns.json
output/synthetic_leads.csv
output/leads_scored.csv
```

Then run:

```bash
python predict.py
```

Expected output:

- Top 10 highest-scored leads
- Tier distribution
- Scored CSV file

### Step 4. Score Salesforce leads

Make sure these Salesforce custom fields exist on the `Lead` object:

```text
Lead_Score__c
Lead_Score_Tier__c
```

Then run:

```bash
python sf_integration.py
```

Expected output:

- Salesforce connection confirmation
- Number of extracted leads
- Number of successfully updated leads
- Preview of scored leads

## Salesforce Setup Notes

The integration uses the OAuth client credentials flow. In Salesforce, you need:

- A Connected App
- Client credentials flow enabled
- A valid consumer key
- A valid consumer secret
- API access permission
- Read and update access to the `Lead` object
- Access to the custom fields `Lead_Score__c` and `Lead_Score_Tier__c`

The SOQL query reads these fields from Salesforce:

```sql
SELECT Id, LeadSource, Industry, Title, NumberOfEmployees,
       AnnualRevenue, Country, Email, CreatedDate, Status
FROM Lead
WHERE IsConverted = false
```

## Model Features

The model uses these raw input fields:

- `LeadSource`
- `Industry`
- `Title`
- `Country`
- `Email_Domain_Type`
- `NumberOfEmployees`
- `AnnualRevenue`

Categorical fields are converted into one-hot encoded columns. Numeric fields
are passed through directly.

The list of training columns is saved in:

```text
output/feature_columns.json
```

This is important because new scoring data may not contain every category seen
during training. The project reindexes scoring features to match the training
columns exactly.

## Model Output

The model returns a conversion probability. The project converts that
probability into a score:

```text
Lead_Score = probability * 100
```

Then the score is converted into a tier:

```text
80-100  -> Hot
50-79   -> Warm
20-49   -> Cool
0-19    -> Cold
```

## Recommended Run Order

Use this order when starting from scratch:

```bash
python generate_synthetic_leads.py
python train_model.py
python predict.py
```

Use this order when updating Salesforce:

```bash
python generate_synthetic_leads.py
python train_model.py
python sf_integration.py
```

## Troubleshooting

### `FileNotFoundError` for model or feature files

Run training first:

```bash
python train_model.py
```

### `FileNotFoundError` for `synthetic_leads.csv`

Generate the data first:

```bash
python generate_synthetic_leads.py
```

### `predict.py` cannot find model or data files

Check that every path in `predict.py` uses the project `output` folder:

```text
output/lead_scoring_model.json
output/feature_columns.json
output/synthetic_leads.csv
output/leads_scored.csv
```

### Salesforce authentication fails

Check:

- `.env` exists
- `SF_DOMAIN` does not include `https://`
- The Connected App consumer key and secret are correct
- The Connected App supports client credentials flow
- The integration user has API access

### Salesforce update fails

Check:

- `Lead_Score__c` exists on the Lead object
- `Lead_Score_Tier__c` exists on the Lead object
- The Salesforce user has permission to update those fields
- The leads returned by the query are editable by the integration user

## Notes

- The current comments in the Python files are written mostly in French.
- The generated synthetic data is deterministic because `random.seed(42)` is used.
- The model is a prototype and should be validated with real historical data before
  production use.
- Synthetic conversion labels are based on handcrafted assumptions, so the model
  learns those assumptions rather than real customer behavior.
