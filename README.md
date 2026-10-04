# Loan Approval Prediction

A machine-learning portfolio application that estimates loan approval using trained classification models and a responsive Streamlit interface for desktop and mobile.

> This application is an educational demonstration. Its predictions are not a real banking decision, financial advice, or a guarantee of loan approval.

## Project overview

The app collects applicant and asset information, applies the categorical encoding and feature order expected by the saved loan classifiers, and displays each model's prediction alongside a majority decision. It also provides a compact overview of the included dataset and a monochrome light/dark interface.

## Problem statement

Loan applicants and analysts may want a quick, consistent way to explore how applicant details relate to a historical approval outcome. This project demonstrates how trained classification models can make predictions from those details; it does not replace a lender's review or underwriting process.

## Features

- Responsive Streamlit UI for desktop and mobile.
- Monochrome light and dark themes.
- Applicant form covering every feature expected by the loan classifiers.
- Separate predictions from four compatible loan-approval classifiers.
- Majority decision when one outcome receives more votes; reports a tie when votes are evenly split.
- Dataset overview with record count, feature count, target distribution, and numeric summaries.
- Required Documents reference section.
- No document upload, verification, or storage functionality is implemented.

## Machine learning models

The app loads these saved classifiers without retraining or modifying them:

| Model | Artifact |
| --- | --- |
| Logistic Regression | `logistic_model.pkl` |
| Decision Tree | `decision_tree_loan_model.pkl` |
| Random Forest | `loan_rf.pkl` |
| K-Nearest Neighbors | `knn_loan_model.pkl` |

The app determines the trained input feature order from these model artifacts. Accuracy metrics are not included in the project, so the UI does not report invented accuracy values.

`Random_forest.pkl` is also retained in the repository as an existing project artifact. Its saved estimator is a Random Forest regressor with vehicle-related features, not a loan-approval classifier, so the loan prediction app does not load it.

## Input features

The loan classifiers expect these features, in their saved training order:

1. Loan ID (`loan_id`)
2. Number of dependents (`no_of_dependents`)
3. Education (`education`)
4. Self-employed status (`self_employed`)
5. Annual income (`income_annum`)
6. Loan amount (`loan_amount`)
7. Loan term (`loan_term`)
8. CIBIL score (`cibil_score`)
9. Residential assets value (`residential_assets_value`)
10. Commercial assets value (`commercial_assets_value`)
11. Luxury assets value (`luxury_assets_value`)
12. Bank asset value (`bank_asset_value`)

The categorical mapping used by the app is **Graduate = 1, Not Graduate = 0** and **Yes = 1, No = 0**. Numeric inputs use the units and scale represented in the CSV dataset.

## Prediction workflow

1. Load the CSV and saved model artifacts from the project directory.
2. Collect the 12 applicant features through the Streamlit form.
3. Convert education and self-employment choices to the numeric mappings above.
4. Arrange all feature values in the order recorded by the trained classifiers.
5. Request a prediction from each of the four loan classifiers.
6. Display each prediction and the majority decision (or an even-split result).

## Technologies

- Python
- Streamlit
- pandas
- scikit-learn
- joblib

## Project structure

```text
Loan_approval/
├── app.py
├── loan_approval_dataset.csv
├── logistic_model.pkl
├── decision_tree_loan_model.pkl
├── loan_rf.pkl
├── knn_loan_model.pkl
├── Random_forest.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The scikit-learn dependency is pinned to the model artifacts' training version (1.7.2) to avoid version-mismatch warnings and reduce compatibility risk.

## Run the Streamlit app

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL to open in a browser.

## Required Documents

Keep the following documents ready for loan verification:

- Aadhaar Card — Identity Proof
- PAN Card — Financial/Tax Proof
- Address Proof — Aadhaar Card, Electricity Bill, or other valid address proof
- Income Proof — Salary Slip or ITR
- Bank Statement — Recent 6 months
- Employment / Business Proof — Job or Business related proof
- Passport Size Photo

The current UI includes an **Additional Documents — Coming Soon** note. Document uploads and verification are future functionality; no documents are collected or stored by this app.

## Future improvements

- Add document upload and verification only with appropriate consent and secure storage.
- Add project tests for model loading, category encoding, and prediction formatting.
- Add verified evaluation metrics when they are available from the original training process.
- Improve deployment guidance and accessibility checks.

## Author

**Dhruvil Usdad**

## GitHub repository

[https://github.com/dhruvilusdad/Loan_approval](https://github.com/dhruvilusdad/Loan_approval)
