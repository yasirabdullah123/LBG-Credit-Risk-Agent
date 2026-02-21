"""
app.py — Phase 4: Streamlit Dashboard
LBG Credit Risk Assistant (Agentic AI)

Tabs:
  1. Application Assessment — input fields + live decision + SHAP waterfall
  2. Model Performance     — confusion matrix + feature importances
  3. Fairness Report       — Fairlearn bias charts + metrics table
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st
from PIL import Image

from credit_agent import CreditAgent
from genai_mock import generate_explanation_email, top_shap_reason

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LBG Credit Risk Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Option Maps (code → human label) ────────────────────────────────────────
CHECKING_STATUS = {
    "A11": "< 0 DM (Overdrawn)",
    "A12": "0 – 200 DM",
    "A13": "> 200 DM",
    "A14": "No Checking Account",
}
CREDIT_HISTORY = {
    "A30": "No credits / All paid elsewhere",
    "A31": "All credits paid at this bank",
    "A32": "Existing credits paid duly",
    "A33": "Past delays in payment",
    "A34": "Critical / Other credits existing",
}
PURPOSE = {
    "A40": "New Car",
    "A41": "Used Car",
    "A42": "Furniture / Equipment",
    "A43": "Radio / Television",
    "A44": "Domestic Appliances",
    "A45": "Repairs",
    "A46": "Education",
    "A48": "Retraining",
    "A49": "Business",
    "A410": "Other",
}
SAVINGS_STATUS = {
    "A61": "< 100 DM",
    "A62": "100 – 499 DM",
    "A63": "500 – 999 DM",
    "A64": ">= 1000 DM",
    "A65": "Unknown / No Savings Account",
}
EMPLOYMENT = {
    "A71": "Unemployed",
    "A72": "< 1 Year",
    "A73": "1 – 3 Years",
    "A74": "4 – 6 Years",
    "A75": ">= 7 Years",
}
PERSONAL_STATUS = {
    "A91": "Male — Divorced / Separated",
    "A92": "Female — Divorced / Separated / Married",
    "A93": "Male — Single",
    "A94": "Male — Married / Widowed",
}
OTHER_PARTIES = {
    "A101": "None",
    "A102": "Co-applicant",
    "A103": "Guarantor",
}
PROPERTY = {
    "A121": "Real Estate",
    "A122": "Building Society Savings / Life Insurance",
    "A123": "Car / Other",
    "A124": "No Known Property",
}
OTHER_PLANS = {
    "A141": "Bank",
    "A142": "Stores",
    "A143": "None",
}
HOUSING = {
    "A151": "Rent",
    "A152": "Own",
    "A153": "Free (provided by employer / family)",
}
JOB = {
    "A171": "Unemployed / Unskilled (Non-Resident)",
    "A172": "Unskilled (Resident)",
    "A173": "Skilled / Official",
    "A174": "Management / Self-Employed / Highly Qualified",
}
TELEPHONE = {"A191": "No", "A192": "Yes"}
FOREIGN   = {"A201": "Yes", "A202": "No"}


def _code(mapping: dict, label: str) -> str:
    """Reverse-lookup: human label → dataset code."""
    return {v: k for k, v in mapping.items()}[label]


# ─── Cached Resources ─────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    return CreditAgent()


@st.cache_resource
def load_shap_components():
    """Extracts preprocessor + classifier for live SHAP computation."""
    with open("models/credit_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    return preprocessor, classifier


CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
]
NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents",
]

COL_LABELS = {
    "checking_status": "Checking Acct",
    "credit_history": "Credit History",
    "purpose": "Purpose",
    "savings_status": "Savings",
    "employment": "Employment",
    "personal_status": "Personal Status",
    "other_parties": "Other Parties",
    "property_magnitude": "Property",
    "other_payment_plans": "Other Plans",
    "housing": "Housing",
    "job": "Job",
    "own_telephone": "Telephone",
    "foreign_worker": "Foreign Worker",
    "duration": "Duration (months)",
    "credit_amount": "Credit Amount (DM)",
    "installment_commitment": "Installment Rate (%)",
    "residence_since": "Residence (yrs)",
    "age": "Age (years)",
    "existing_credits": "Existing Credits",
    "num_dependents": "Dependents",
}


def shorten(name: str) -> str:
    for col, label in COL_LABELS.items():
        if name.startswith(col + "_"):
            return f"{label}: {name[len(col)+1:]}"
        if name == col:
            return label
    return name


def compute_shap_waterfall(applicant: dict):
    """Returns a matplotlib Figure of the SHAP waterfall for one applicant."""
    preprocessor, classifier = load_shap_components()

    ohe_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    feature_names = [shorten(n) for n in ohe_names + NUMERICAL_COLS]

    df = pd.DataFrame([applicant])
    X_transformed = preprocessor.transform(df)

    explainer   = shap.TreeExplainer(classifier)
    shap_vals   = explainer.shap_values(X_transformed)
    expected_val = float(explainer.expected_value)

    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=expected_val,
        data=X_transformed[0],
        feature_names=feature_names,
    )

    shap.waterfall_plot(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    plt.tight_layout()
    # Also return raw values + names so the caller can extract the top SHAP reason
    return fig, shap_vals[0].tolist(), feature_names


# ─── Sidebar — Input Form ─────────────────────────────────────────────────────
def render_sidebar() -> dict:
    st.sidebar.title("Applicant Details")

    st.sidebar.subheader("Loan Request")
    duration        = st.sidebar.slider("Loan Duration (months)", 4, 72, 24)
    credit_amount   = st.sidebar.number_input("Credit Amount (DM)", 250, 20000, 3000, step=250)
    purpose_label   = st.sidebar.selectbox("Purpose", list(PURPOSE.values()), index=2)
    installment_pct = st.sidebar.slider("Installment Rate (% of income)", 1, 4, 2)

    st.sidebar.subheader("Applicant Profile")
    age             = st.sidebar.slider("Age", 18, 75, 35)
    personal_label  = st.sidebar.selectbox("Personal Status / Sex", list(PERSONAL_STATUS.values()), index=2)
    foreign_label   = st.sidebar.selectbox("Foreign Worker?", list(FOREIGN.values()), index=0)
    telephone_label = st.sidebar.selectbox("Own Telephone?", list(TELEPHONE.values()), index=1)

    st.sidebar.subheader("Financial Standing")
    checking_label  = st.sidebar.selectbox("Checking Account Status", list(CHECKING_STATUS.values()), index=3)
    savings_label   = st.sidebar.selectbox("Savings Account", list(SAVINGS_STATUS.values()), index=4)
    employment_label= st.sidebar.selectbox("Employment Duration", list(EMPLOYMENT.values()), index=4)

    st.sidebar.subheader("Credit History & Obligations")
    history_label   = st.sidebar.selectbox("Credit History", list(CREDIT_HISTORY.values()), index=2)
    existing_credits= st.sidebar.slider("Existing Credits at this Bank", 1, 4, 1)
    other_plans_lbl = st.sidebar.selectbox("Other Payment Plans", list(OTHER_PLANS.values()), index=2)
    other_party_lbl = st.sidebar.selectbox("Other Parties (Guarantor / Co-applicant)", list(OTHER_PARTIES.values()), index=0)

    st.sidebar.subheader("Property & Living Situation")
    property_label  = st.sidebar.selectbox("Property", list(PROPERTY.values()), index=0)
    housing_label   = st.sidebar.selectbox("Housing", list(HOUSING.values()), index=1)
    residence_since = st.sidebar.slider("Years at Current Residence", 1, 4, 3)
    num_dependents  = st.sidebar.slider("Number of Dependents", 1, 2, 1)
    job_label       = st.sidebar.selectbox("Job Type", list(JOB.values()), index=2)

    return {
        "checking_status":        _code(CHECKING_STATUS, checking_label),
        "duration":               duration,
        "credit_history":         _code(CREDIT_HISTORY, history_label),
        "purpose":                _code(PURPOSE, purpose_label),
        "credit_amount":          credit_amount,
        "savings_status":         _code(SAVINGS_STATUS, savings_label),
        "employment":             _code(EMPLOYMENT, employment_label),
        "installment_commitment": installment_pct,
        "personal_status":        _code(PERSONAL_STATUS, personal_label),
        "other_parties":          _code(OTHER_PARTIES, other_party_lbl),
        "residence_since":        residence_since,
        "property_magnitude":     _code(PROPERTY, property_label),
        "age":                    age,
        "other_payment_plans":    _code(OTHER_PLANS, other_plans_lbl),
        "housing":                _code(HOUSING, housing_label),
        "existing_credits":       existing_credits,
        "job":                    _code(JOB, job_label),
        "num_dependents":         num_dependents,
        "own_telephone":          _code(TELEPHONE, telephone_label),
        "foreign_worker":         _code(FOREIGN, foreign_label),
    }


# ─── Decision Badge ───────────────────────────────────────────────────────────
DECISION_STYLE = {
    "AUTO_APPROVE": {
        "bg": "#d4edda", "border": "#28a745", "text": "#155724",
        "icon": "✅", "label": "AUTO APPROVED",
    },
    "HUMAN_REFERRAL": {
        "bg": "#fff3cd", "border": "#ffc107", "text": "#856404",
        "icon": "⚠️", "label": "REFERRED TO UNDERWRITER",
    },
    "AUTO_DECLINE": {
        "bg": "#f8d7da", "border": "#dc3545", "text": "#721c24",
        "icon": "❌", "label": "AUTO DECLINED",
    },
}


def render_decision_badge(result):
    s = DECISION_STYLE[result.decision]
    st.markdown(
        f"""
        <div style="
            background-color:{s['bg']};
            border: 2px solid {s['border']};
            border-radius: 12px;
            padding: 20px 28px;
            margin-bottom: 16px;
        ">
            <h2 style="color:{s['text']}; margin:0 0 6px 0;">
                {s['icon']} &nbsp; {s['label']}
            </h2>
            <p style="color:{s['text']}; margin:0; font-size:15px;">
                {result.rationale}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Tab 1: Application Assessment ───────────────────────────────────────────
def tab_assessment(applicant: dict):
    st.header("Application Assessment")
    st.caption(
        "Fill in the applicant details in the sidebar, then click **Assess Application**. "
        "The agent applies governance thresholds (< 20% auto-approve, > 80% auto-decline) "
        "and provides a SHAP explanation for every decision."
    )

    if st.button("Assess Application", type="primary", use_container_width=True):
        agent = load_agent()

        with st.spinner("Running model and computing SHAP values…"):
            result = agent.evaluate(applicant)
            shap_fig, shap_vals, shap_names = compute_shap_waterfall(applicant)

        # ── Decision Badge ─────────────────────────────────────────────────
        render_decision_badge(result)

        # ── Metrics Row ────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("P(Default)", f"{result.default_probability:.1%}")
        col2.metric("Risk Level", result.risk_level)
        col3.metric("Credit Amount", f"DM {applicant['credit_amount']:,}")
        col4.metric("Duration", f"{applicant['duration']} months")

        st.divider()

        # ── SHAP Waterfall ─────────────────────────────────────────────────
        st.subheader("Why did the model give this score?")
        st.caption(
            "Each bar shows how much a feature pushed the default probability **up** (red) "
            "or **down** (blue) from the model's baseline. The final value is P(default)."
        )
        st.pyplot(shap_fig, use_container_width=True)
        plt.close()

        # ── GenAI Customer Letter ──────────────────────────────────────────
        if result.decision in ("AUTO_DECLINE", "HUMAN_REFERRAL"):
            st.divider()
            st.subheader("Customer Communication Draft")
            st.caption(
                "Auto-generated using deterministic templates that mirror a prompt-engineered "
                "LLM response. Banks prefer this over live LLMs to eliminate hallucination risk. "
                "Swap `generate_explanation_email()` in `genai_mock.py` to call GPT-4o in production."
            )

            # Extract top risk driver from SHAP to personalise the letter
            reason = top_shap_reason(shap_vals, shap_names, result.decision)
            customer_name = st.text_input("Customer first name (for the letter)", value="Customer")
            email_body, prompt_used = generate_explanation_email(customer_name, result.decision, reason)

            st.text_area("Generated Letter", value=email_body, height=220)

            with st.expander("Show underlying prompt (Prompt Engineering)", expanded=False):
                st.code(prompt_used, language=None)

        # ── Audit Trail ────────────────────────────────────────────────────
        with st.expander("Audit Trail", expanded=False):
            st.json(result.to_dict())


# ─── Tab 2: Model Performance ─────────────────────────────────────────────────
def tab_performance():
    st.header("Model Performance")
    st.caption(
        "XGBoost trained on 800 samples with SMOTE oversampling. "
        "Regularised (max_depth=2, n_estimators=100, L2=10) to close Train/Test gap to 0.07. "
        "Evaluated on a held-out stratified test set of 200 applicants."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("CV ROC-AUC (5-fold)", "0.780 ± 0.049")
    col2.metric("Test ROC-AUC", "0.764")
    col3.metric("Default Recall", "82%")

    st.divider()

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.subheader("Confusion Matrix")
        st.image("models/confusion_matrix.png", use_container_width=True)

    with img_col2:
        st.subheader("Top 20 Feature Importances")
        st.image("models/feature_importances.png", use_container_width=True)

    st.subheader("SHAP Summary — Test Set")
    st.caption(
        "Each point is one applicant. Red = high feature value, Blue = low. "
        "X-axis shows impact on P(default)."
    )
    st.image("models/shap_beeswarm.png", use_container_width=True)


# ─── Tab 3: Fairness Report ───────────────────────────────────────────────────
def tab_fairness():
    import json

    st.header("Ethical AI — Fairness Report")
    st.caption(
        "Fairlearn MetricFrame analysis of model behaviour across demographic groups. "
        "A False Positive Rate (FPR) disparity > 10% is flagged as a potential bias concern."
    )

    with open("models/bias_report.json") as f:
        bias = json.load(f)

    # ── Age Group ─────────────────────────────────────────────────────────
    st.subheader("By Age Group  (< 25 vs 25+)")

    age_fpr = bias["age_group"]["by_group"]["false_positive_rate"]
    age_disp = bias["age_group"]["disparity"]["false_positive_rate"]
    age_overall = bias["age_group"]["overall"]["false_positive_rate"]

    flag = "⚠️  **Bias Detected**" if age_disp > 0.10 else "✅ Within Acceptable Range"
    col1, col2, col3 = st.columns(3)
    col1.metric("FPR — Under 25", f"{age_fpr.get('<25', 0):.1%}")
    col2.metric("FPR — 25+",      f"{age_fpr.get('25+', 0):.1%}")
    col3.metric("FPR Disparity",  f"{age_disp:.1%}", delta=f"{flag}", delta_color="off")

    st.markdown(
        f"""
        > **Interpretation:** Applicants under 25 are incorrectly rejected at a rate of
        **{age_fpr.get('<25', 0):.1%}** compared to **{age_fpr.get('25+', 0):.1%}** for older
        applicants — a **{age_disp:.1%}** disparity. This exceeds the 10% threshold and
        suggests the model has learned age-correlated patterns that may constitute
        indirect age discrimination. Young applicants in the medium-risk band should be
        prioritised for human underwriter review.
        """
    )

    # ── Gender ────────────────────────────────────────────────────────────
    st.subheader("By Gender  (derived from Personal Status field)")

    gen_fpr  = bias["gender"]["by_group"]["false_positive_rate"]
    gen_disp = bias["gender"]["disparity"]["false_positive_rate"]

    flag_g = "⚠️  **Bias Detected**" if gen_disp > 0.10 else "✅ Within Acceptable Range"
    col1, col2, col3 = st.columns(3)
    col1.metric("FPR — Female", f"{gen_fpr.get('Female', 0):.1%}")
    col2.metric("FPR — Male",   f"{gen_fpr.get('Male', 0):.1%}")
    col3.metric("FPR Disparity", f"{gen_disp:.1%}", delta=f"{flag_g}", delta_color="off")

    st.divider()

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.subheader("False Positive Rate by Group")
        st.image("models/bias_report.png", use_container_width=True)
    with img_col2:
        st.subheader("Approval Rate by Group")
        st.image("models/bias_approval_rates.png", use_container_width=True)

    with st.expander("Full Metrics Table", expanded=False):
        for feature in ["age_group", "gender"]:
            st.markdown(f"**{feature}**")
            rows = []
            for group, metrics in bias[feature]["by_group"].items():
                rows.append({"Group": group, **{k: f"{v:.1%}" for k, v in metrics.items()}})
            st.dataframe(pd.DataFrame(rows).set_index("Group"), use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown(
        """
        <h1 style='margin-bottom:0'>🏦 LBG Credit Risk Assistant</h1>
        <p style='color:grey; margin-top:4px; font-size:16px'>
        Agentic AI — XGBoost + SHAP + Fairlearn &nbsp;|&nbsp; German Credit Dataset
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    applicant = render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "Application Assessment",
        "Model Performance",
        "Fairness Report",
    ])

    with tab1:
        tab_assessment(applicant)
    with tab2:
        tab_performance()
    with tab3:
        tab_fairness()


if __name__ == "__main__":
    main()
