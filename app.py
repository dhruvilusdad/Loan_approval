from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd
import streamlit as st


class LoanClassifier(Protocol):
    feature_names_in_: Any

    def predict(self, features: pd.DataFrame) -> Any: ...


APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "loan_approval_dataset.csv"
TARGET_COLUMN = "loan_status"

MODEL_SPECS = (
    ("Logistic Regression", "logistic_model.pkl"),
    ("Decision Tree", "decision_tree_loan_model.pkl"),
    ("Random Forest", "loan_rf.pkl"),
    ("K-Nearest Neighbors", "knn_loan_model.pkl"),
)

FEATURE_LABELS = {
    "loan_id": "Loan ID",
    "no_of_dependents": "Number of dependents",
    "education": "Education",
    "self_employed": "Self employed",
    "income_annum": "Annual income",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term",
    "cibil_score": "CIBIL score",
    "residential_assets_value": "Residential assets",
    "commercial_assets_value": "Commercial assets",
    "luxury_assets_value": "Luxury assets",
    "bank_asset_value": "Bank assets",
}

NUMERIC_INPUTS = (
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
)

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH)
    data.columns = data.columns.astype(str).str.strip()
    for column in ("education", "self_employed", TARGET_COLUMN):
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip()
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"The dataset is missing the required {TARGET_COLUMN!r} column.")
    return data


@st.cache_resource
def load_models() -> dict[str, LoanClassifier]:
    models: dict[str, LoanClassifier] = {}
    feature_order: tuple[str, ...] | None = None

    for model_name, filename in MODEL_SPECS:
        model = joblib.load(APP_DIR / filename)
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            raise ValueError(f"{filename} does not expose its trained feature names.")

        current_order = tuple(str(feature) for feature in model_features)
        if feature_order is None:
            feature_order = current_order
        elif current_order != feature_order:
            raise ValueError(
                f"{filename} uses a different feature order from the other loan models."
            )
        models[model_name] = model

    if feature_order is None:
        raise ValueError("No loan approval models are configured.")
    return models


def dataset_default(data: pd.DataFrame, column: str) -> int:
    value = pd.to_numeric(data[column], errors="raise").median()
    return int(round(value))


def dataset_minimum(data: pd.DataFrame, column: str) -> int:
    value = pd.to_numeric(data[column], errors="raise").min()
    return int(value)


def render_theme_styles(dark_mode: bool) -> None:
    if dark_mode:
        colors = {
            "background": "#0b0b0b",
            "surface": "#171717",
            "surface_alt": "#202020",
            "text": "#f4f4f4",
            "muted": "#b7b7b7",
            "border": "#383838",
            "input": "#171717",
            "button": "#f4f4f4",
            "button_text": "#111111",
        }
    else:
        colors = {
            "background": "#fafafa",
            "surface": "#ffffff",
            "surface_alt": "#f0f0f0",
            "text": "#141414",
            "muted": "#555555",
            "border": "#d7d7d7",
            "input": "#ffffff",
            "button": "#111111",
            "button_text": "#ffffff",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {"dark" if dark_mode else "light"};
            --app-bg: {colors["background"]};
            --surface: {colors["surface"]};
            --surface-alt: {colors["surface_alt"]};
            --text: {colors["text"]};
            --muted: {colors["muted"]};
            --border: {colors["border"]};
            --input-bg: {colors["input"]};
            --button-bg: {colors["button"]};
            --button-text: {colors["button_text"]};
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: var(--app-bg);
            color: var(--text);
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }}
        [data-testid="stSidebar"], [data-testid="stBottom"], footer {{
            display: none;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        hr {{
            border-color: var(--border);
        }}
        .main .block-container {{
            max-width: 1120px;
            padding: 2rem 2rem 3rem;
        }}
        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {{
            color: var(--text);
        }}
        h1 {{
            font-size: clamp(1.65rem, 3vw, 2.15rem);
            line-height: 1.2;
            letter-spacing: -0.035em;
            margin: 0.25rem 0 0.4rem;
        }}
        h2 {{
            font-size: 1.12rem;
            letter-spacing: -0.015em;
            margin: 0 0 0.3rem;
        }}
        p, [data-testid="stCaptionContainer"] {{
            color: var(--muted);
        }}
        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
            color: var(--text) !important;
            font-weight: 550;
        }}
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-baseweb="select"] > div,
        [data-testid="stTextArea"] textarea {{
            color: var(--text) !important;
            background: var(--input-bg) !important;
            border-color: var(--border) !important;
        }}
        [data-baseweb="popover"], [role="listbox"], [role="option"] {{
            color: var(--text) !important;
            background: var(--surface) !important;
        }}
        input, textarea {{
            color-scheme: {"dark" if dark_mode else "light"};
        }}
        [data-testid="stForm"], [data-testid="stExpander"] {{
            border-color: var(--border);
        }}
        [data-testid="stForm"] {{
            background: var(--surface);
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
        }}
        [data-testid="stExpander"] {{
            background: var(--surface);
            border-radius: 10px;
        }}
        [data-testid="stFormSubmitButton"] button {{
            min-height: 3.1rem;
            width: 100%;
            border: 1px solid var(--button-bg);
            border-radius: 8px;
            background: var(--button-bg);
            color: var(--button-text);
            font-weight: 700;
            letter-spacing: 0.035em;
        }}
        [data-testid="stFormSubmitButton"] button:hover {{
            border-color: var(--button-bg);
            color: var(--button-text);
            opacity: 0.88;
        }}
        [data-testid="stToggle"] label {{
            color: var(--text) !important;
        }}
        input[type="checkbox"] {{
            accent-color: var(--button-bg);
        }}
        .eyebrow {{
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .subtitle {{
            margin: 0;
            color: var(--muted);
            font-size: 0.95rem;
        }}
        .section-intro {{
            margin: 0.1rem 0 1rem;
            color: var(--muted);
            font-size: 0.9rem;
        }}
        .decision-card {{
            margin: 0.8rem 0 1.25rem;
            padding: 1.25rem 1.4rem;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--surface-alt);
        }}
        .decision-label {{
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }}
        .decision-value {{
            margin: 0.32rem 0 0;
            color: var(--text);
            font-size: clamp(1.4rem, 4vw, 1.85rem);
            font-weight: 750;
            line-height: 1.2;
        }}
        .decision-note {{
            margin: 0.55rem 0 0;
            color: var(--muted);
            font-size: 0.88rem;
        }}
        .documents-section {{
            margin: 1.6rem 0;
        }}
        .documents-title {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0;
            color: var(--text);
            font-size: 1.12rem;
            font-weight: 700;
        }}
        .documents-title-icon {{
            filter: grayscale(1);
            font-size: 1rem;
        }}
        .documents-subtitle {{
            margin: 0.35rem 0 1rem;
            color: var(--muted);
            font-size: 0.9rem;
        }}
        .documents-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
        }}
        .document-card {{
            min-width: 0;
            padding: 0.9rem 1rem;
            border: 1px solid var(--border);
            border-radius: 9px;
            background: var(--surface);
        }}
        .document-name {{
            margin: 0;
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 650;
            overflow-wrap: anywhere;
        }}
        .document-detail {{
            margin: 0.25rem 0 0;
            color: var(--muted);
            font-size: 0.83rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }}
        .additional-documents {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-top: 0.8rem;
            padding: 0.85rem 1rem;
            border: 1px dashed var(--border);
            border-radius: 9px;
            background: var(--surface-alt);
        }}
        .additional-documents-title {{
            margin: 0;
            color: var(--text);
            font-size: 0.88rem;
            font-weight: 650;
        }}
        .coming-soon {{
            flex: 0 0 auto;
            padding: 0.22rem 0.5rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .responsive-table {{
            width: 100%;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--surface);
        }}
        .responsive-table table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            color: var(--text);
        }}
        .responsive-table th, .responsive-table td {{
            padding: 0.72rem 0.85rem;
            border-bottom: 1px solid var(--border);
            text-align: left;
            overflow-wrap: anywhere;
            font-size: 0.88rem;
        }}
        .responsive-table th {{
            color: var(--muted);
            background: var(--surface-alt);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .responsive-table tr:last-child td {{
            border-bottom: 0;
        }}
        .page-note {{
            margin-top: 1.5rem;
            padding-top: 0.9rem;
            border-top: 1px solid var(--border);
            color: var(--muted);
            font-size: 0.78rem;
        }}
        @media (max-width: 700px) {{
            .main .block-container {{
                padding: 1.1rem 1rem 2rem;
            }}
            [data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap;
                gap: 0.65rem;
            }}
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }}
            [data-testid="stForm"] {{
                padding: 0.85rem;
            }}
            .documents-section {{
                margin: 1.25rem 0;
            }}
            .documents-grid {{
                grid-template-columns: minmax(0, 1fr);
                gap: 0.55rem;
            }}
            .document-card {{
                padding: 0.8rem 0.9rem;
            }}
            .responsive-table thead {{
                display: none;
            }}
            .responsive-table tbody tr {{
                display: grid;
                grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                border-bottom: 1px solid var(--border);
                padding: 0.25rem 0;
            }}
            .responsive-table tbody tr:last-child {{
                border-bottom: 0;
            }}
            .responsive-table tbody td {{
                display: block;
                min-width: 0;
                padding: 0.45rem 0.65rem;
                border: 0;
            }}
            .responsive-table tbody td:first-child {{
                grid-column: 1 / -1;
                font-weight: 650;
            }}
            .responsive-table tbody td:not(:first-child)::before {{
                display: block;
                margin-bottom: 0.15rem;
                color: var(--muted);
                content: attr(data-label);
                font-size: 0.67rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def responsive_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    head = "".join(f"<th scope='col'>{header}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td data-label='{header}'>{value}</td>"
            for header, value in zip(headers, row)
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<div class='responsive-table'><table>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


data = load_dataset()
models = load_models()
feature_order = tuple(next(iter(models.values())).feature_names_in_)
missing_features = [feature for feature in feature_order if feature not in data.columns]
if missing_features:
    raise ValueError(
        "The dataset is missing model features: " + ", ".join(missing_features)
    )

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

header, theme_control = st.columns([5, 1], vertical_alignment="center")
with header:
    st.markdown(
        """
        <h1>LOAN APPROVAL PREDICTION</h1>
        <p class="subtitle">Machine Learning Based Prediction System</p>
        <p class="section-intro">Predict loan approval using multiple Machine Learning models.</p>
        """,
        unsafe_allow_html=True,
    )
with theme_control:
    st.toggle("Dark mode", key="dark_mode")

render_theme_styles(st.session_state.dark_mode)
st.divider()

# Applicant form: keep the original trained names and encodings at inference time.
st.markdown("<h2>Applicant Information</h2>", unsafe_allow_html=True)
st.markdown(
    "<p class='section-intro'>Enter the applicant details used by the trained models. "
    "Values use the same units as the dataset.</p>",
    unsafe_allow_html=True,
)

loan_id_default = int(pd.to_numeric(data["loan_id"], errors="raise").max()) + 1
education_default = (
    data["education"].mode().iloc[0] if "education" in data.columns else "Graduate"
)
self_employed_default = (
    data["self_employed"].mode().iloc[0]
    if "self_employed" in data.columns
    else "No"
)

with st.form("loan_application"):
    st.caption(
        f"Loan ID is required by the saved models. The default is the next ID after the "
        f"{len(data):,} records in the dataset."
    )
    left, right = st.columns(2, gap="large")
    with left:
        loan_id = st.number_input(
            "Loan ID",
            min_value=0,
            value=loan_id_default,
            step=1,
            format="%d",
        )
        no_of_dependents = st.number_input(
            FEATURE_LABELS["no_of_dependents"],
            min_value=dataset_minimum(data, "no_of_dependents"),
            value=dataset_default(data, "no_of_dependents"),
            step=1,
            format="%d",
        )
        education = st.selectbox(
            FEATURE_LABELS["education"],
            options=("Graduate", "Not Graduate"),
            index=("Graduate", "Not Graduate").index(education_default),
        )
        self_employed = st.selectbox(
            FEATURE_LABELS["self_employed"],
            options=("Yes", "No"),
            index=("Yes", "No").index(self_employed_default),
        )
        income_annum = st.number_input(
            FEATURE_LABELS["income_annum"],
            min_value=dataset_minimum(data, "income_annum"),
            value=dataset_default(data, "income_annum"),
            step=1,
            format="%d",
        )
        loan_amount = st.number_input(
            FEATURE_LABELS["loan_amount"],
            min_value=dataset_minimum(data, "loan_amount"),
            value=dataset_default(data, "loan_amount"),
            step=1,
            format="%d",
        )
        loan_term = st.number_input(
            FEATURE_LABELS["loan_term"],
            min_value=dataset_minimum(data, "loan_term"),
            value=dataset_default(data, "loan_term"),
            step=1,
            format="%d",
        )
    with right:
        cibil_score = st.number_input(
            FEATURE_LABELS["cibil_score"],
            min_value=dataset_minimum(data, "cibil_score"),
            value=dataset_default(data, "cibil_score"),
            step=1,
            format="%d",
        )
        residential_assets_value = st.number_input(
            FEATURE_LABELS["residential_assets_value"],
            min_value=dataset_minimum(data, "residential_assets_value"),
            value=dataset_default(data, "residential_assets_value"),
            step=1,
            format="%d",
        )
        commercial_assets_value = st.number_input(
            FEATURE_LABELS["commercial_assets_value"],
            min_value=dataset_minimum(data, "commercial_assets_value"),
            value=dataset_default(data, "commercial_assets_value"),
            step=1,
            format="%d",
        )
        luxury_assets_value = st.number_input(
            FEATURE_LABELS["luxury_assets_value"],
            min_value=dataset_minimum(data, "luxury_assets_value"),
            value=dataset_default(data, "luxury_assets_value"),
            step=1,
            format="%d",
        )
        bank_asset_value = st.number_input(
            FEATURE_LABELS["bank_asset_value"],
            min_value=dataset_minimum(data, "bank_asset_value"),
            value=dataset_default(data, "bank_asset_value"),
            step=1,
            format="%d",
        )

    submitted = st.form_submit_button("PREDICT LOAN APPROVAL", type="primary")

if submitted:
    applicant = {
        "loan_id": loan_id,
        "no_of_dependents": no_of_dependents,
        "education": 1 if education == "Graduate" else 0,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    input_frame = pd.DataFrame(
        [[applicant[feature] for feature in feature_order]],
        columns=feature_order,
    )
    model_results = []
    for model_name, model in models.items():
        class_value = int(model.predict(input_frame)[0])
        if class_value not in (0, 1):
            raise ValueError(
                f"{model_name} returned unexpected prediction class {class_value!r}."
            )
        model_results.append(
            {
                "model": model_name,
                "approved": class_value == 1,
            }
        )

    approvals = sum(result["approved"] for result in model_results)
    rejections = len(model_results) - approvals
    if approvals > rejections:
        decision = "Loan Approved"
        vote_summary = f"{approvals} of {len(model_results)} models predicted approval."
    elif rejections > approvals:
        decision = "Loan Rejected"
        vote_summary = f"{rejections} of {len(model_results)} models predicted rejection."
    else:
        decision = "No majority"
        vote_summary = (
            f"The models split evenly: {approvals} approved and {rejections} rejected."
        )

    st.session_state.prediction_result = {
        "decision": decision,
        "vote_summary": vote_summary,
        "models": model_results,
    }

result = st.session_state.get("prediction_result")
if result:
    st.divider()
    st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <section class="decision-card" aria-live="polite">
            <div class="decision-label">Final Prediction</div>
            <div class="decision-value">{result["decision"]}</div>
            <p class="decision-note">{result["vote_summary"]}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h2>Model Performance / Predictions</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-intro'>Accuracy values are not included with the saved model files.</p>",
        unsafe_allow_html=True,
    )
    prediction_rows = [
        (
            item["model"],
            "Not recorded",
            "Approved" if item["approved"] else "Rejected",
        )
        for item in result["models"]
    ]
    st.markdown(
        responsive_table(("Model Name", "Accuracy", "Prediction"), prediction_rows),
        unsafe_allow_html=True,
    )

    decision_label = (
        result["decision"][5:]
        if result["decision"].startswith("Loan ")
        else result["decision"]
    )
    st.markdown(
        f"<h2>FINAL MODEL DECISION: {decision_label}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='section-intro'>The final decision is based on the predictions of "
        "the trained models. This is a model output, not a real banking decision.</p>",
        unsafe_allow_html=True,
    )

# Required documents reference section; upload and verification are not implemented.
st.markdown(
    """
    <section class="documents-section" aria-labelledby="required-documents-heading">
        <h2 id="required-documents-heading" class="documents-title">
            <span class="documents-title-icon" aria-hidden="true">📄</span>
            Required Documents
        </h2>
        <p class="documents-subtitle">
            Please keep the following documents ready for loan verification.
        </p>
        <div class="documents-grid">
            <article class="document-card">
                <p class="document-name">Aadhaar Card</p>
                <p class="document-detail">Identity Proof</p>
            </article>
            <article class="document-card">
                <p class="document-name">PAN Card</p>
                <p class="document-detail">Financial/Tax Proof</p>
            </article>
            <article class="document-card">
                <p class="document-name">Address Proof</p>
                <p class="document-detail">
                    Aadhaar Card / Electricity Bill / other valid address proof
                </p>
            </article>
            <article class="document-card">
                <p class="document-name">Income Proof</p>
                <p class="document-detail">Salary Slip / ITR</p>
            </article>
            <article class="document-card">
                <p class="document-name">Bank Statement</p>
                <p class="document-detail">Recent 6 months</p>
            </article>
            <article class="document-card">
                <p class="document-name">Employment / Business Proof</p>
                <p class="document-detail">Job or Business related proof</p>
            </article>
            <article class="document-card">
                <p class="document-name">Passport Size Photo</p>
            </article>
        </div>
        <div class="additional-documents">
            <p class="additional-documents-title">Additional Documents</p>
            <span class="coming-soon">Coming Soon</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Dataset context is available without taking attention away from the application form.
with st.expander("Dataset Overview"):
    feature_count = len([column for column in data.columns if column != TARGET_COLUMN])
    overview_left, overview_right = st.columns(2)
    with overview_left:
        st.metric("Records", f"{len(data):,}")
    with overview_right:
        st.metric("Features", f"{feature_count:,}")

    st.markdown("**Target distribution**")
    target_counts = data[TARGET_COLUMN].value_counts(dropna=False)
    target_rows = [
        (
            str(label),
            f"{int(count):,}",
            f"{count / len(data):.1%}",
        )
        for label, count in target_counts.items()
    ]
    st.markdown(
        responsive_table(("Loan status", "Records", "Share"), target_rows),
        unsafe_allow_html=True,
    )

    st.markdown("**Numeric feature summary**")
    numeric_columns = [
        column
        for column in NUMERIC_INPUTS
        if column in data.columns
    ]
    summary_rows = []
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="raise")
        summary_rows.append(
            (
                FEATURE_LABELS[column],
                f"{values.median():,.0f}",
                f"{values.min():,.0f}",
                f"{values.max():,.0f}",
            )
        )
    st.markdown(
        responsive_table(("Feature", "Median", "Minimum", "Maximum"), summary_rows),
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='page-note'>Educational portfolio application. Predictions are generated "
    "by saved machine learning models and should not be treated as financial advice.</div>",
    unsafe_allow_html=True,
)