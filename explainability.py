"""
explainability.py — Phase 3: Explainability & Ethics
Produces:
  1. SHAP waterfall plots (one approved, one declined applicant)
  2. SHAP beeswarm summary plot (test set)
  3. Fairlearn MetricFrame — FPR / FNR by Age Group and Gender
  4. bias_report.json — machine-readable fairness metrics for the Streamlit dashboard

Key design note on the pipeline:
  The saved pipeline is: preprocessor (OHE) → SMOTE (train-only) → XGBClassifier
  For SHAP we extract preprocessor + classifier separately, transform X_test,
  then run TreeExplainer directly on the XGB model with named OHE features.
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate, selection_rate

warnings.filterwarnings("ignore")

# ─── Column definitions (must match train_model.py) ───────────────────────────
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "target",
]
CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
]
NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents",
]

# German dataset personal_status codes → derived sex column
# A91 = male divorced/sep | A92 = female divorced/sep/married
# A93 = male single       | A94 = male married/widowed
MALE_CODES   = {"A91", "A93", "A94"}
FEMALE_CODES = {"A92", "A95"}


# ─── 1. Load data & reproduce the exact same train/test split ─────────────────
def load_test_set():
    df = pd.read_csv("data/german.data", sep=" ", header=None, names=COLUMNS)
    df["target"] = df["target"].map({1: 0, 2: 1})
    X = df.drop(columns=["target"])
    y = df["target"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# ─── 2. Load model and extract sub-components ─────────────────────────────────
def load_pipeline():
    with open("models/credit_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    return pipeline, preprocessor, classifier


def get_feature_names(preprocessor) -> list[str]:
    """Reconstructs the ordered feature list after OHE + passthrough."""
    ohe_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    return ohe_names + NUMERICAL_COLS


# ─── 3. SHAP Analysis ─────────────────────────────────────────────────────────
def run_shap(preprocessor, classifier, X_test: pd.DataFrame, feature_names: list[str]):
    """
    Computes SHAP values for every test applicant.
    Returns:
        explainer       — shap.TreeExplainer
        shap_explanation — shap.Explanation object (values + base + data + names)
        X_transformed   — np.ndarray of preprocessed test features
    """
    X_transformed = preprocessor.transform(X_test)

    explainer = shap.TreeExplainer(classifier)
    # shap_values returns shape (n_samples, n_features) for binary classification
    shap_values = explainer.shap_values(X_transformed)
    expected_value = float(explainer.expected_value)

    # Build a named Explanation object so waterfall/beeswarm get clean labels
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(X_test), expected_value),
        data=X_transformed,
        feature_names=feature_names,
    )
    return explainer, explanation, X_transformed


def _shorten_feature_name(name: str) -> str:
    """Makes OHE feature names readable in plots (e.g. 'checking_status_A11' → 'Checking: A11')."""
    col_labels = {
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
    for col, label in col_labels.items():
        if name.startswith(col + "_"):
            suffix = name[len(col) + 1:]
            return f"{label}: {suffix}"
        if name == col:
            return label
    return name


def plot_waterfall(explanation, idx: int, label: str, pipeline, X_test: pd.DataFrame):
    """Saves a SHAP waterfall plot for a single applicant."""
    # Shorten feature names for readability
    short_names = [_shorten_feature_name(n) for n in explanation.feature_names]

    single = shap.Explanation(
        values=explanation.values[idx],
        base_values=explanation.base_values[idx],
        data=explanation.data[idx],
        feature_names=short_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(single, max_display=15, show=False)
    fig = plt.gcf()

    # Annotate with the model's predicted probability
    prob = float(pipeline.predict_proba(X_test.iloc[[idx]])[0, 1])
    decision = "AUTO_APPROVE" if prob < 0.20 else ("AUTO_DECLINE" if prob > 0.80 else "HUMAN_REFERRAL")
    fig.suptitle(
        f"{label}\nP(Default) = {prob:.1%}  →  {decision}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    path = f"models/shap_waterfall_{label.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return prob


def plot_beeswarm(explanation):
    """Saves a SHAP beeswarm summary plot (test set overview)."""
    short_names = [_shorten_feature_name(n) for n in explanation.feature_names]
    named_exp = shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=short_names,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(named_exp, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Feature Impact — Test Set (n=200)", fontsize=12, fontweight="bold", y=1.01)
    plt.savefig("models/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: models/shap_beeswarm.png")


# ─── 4. Fairlearn Bias Analysis ───────────────────────────────────────────────
def derive_sensitive_features(X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Derives two sensitive feature columns from the raw test data.
      age_group   : '<25' vs '25+'   (regulatory focus on young borrowers)
      gender      : 'Male' vs 'Female' (derived from personal_status codes)
    """
    sf = pd.DataFrame(index=X_test.index)
    sf["age_group"] = X_test["age"].apply(lambda a: "<25" if a < 25 else "25+")
    sf["gender"] = X_test["personal_status"].apply(
        lambda s: "Male" if s in MALE_CODES else "Female"
    )
    return sf


def run_bias_analysis(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Computes MetricFrame for:
      - False Positive Rate  (good applicant incorrectly predicted as defaulter → unfair rejection)
      - False Negative Rate  (defaulter incorrectly approved → bank risk)
      - Selection Rate       (approval rate, should be similar across groups for fairness)
    Broken down by age_group and gender.
    """
    y_pred = pipeline.predict(X_test)
    sf = derive_sensitive_features(X_test)

    metrics = {
        "false_positive_rate": false_positive_rate,  # unfair rejection rate
        "false_negative_rate": false_negative_rate,  # missed default rate
        "selection_rate": selection_rate,            # % predicted as non-default
    }

    results = {}

    for sensitive_col in ["age_group", "gender"]:
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sf[sensitive_col],
        )
        results[sensitive_col] = {
            "by_group": mf.by_group.round(4).to_dict(),
            "overall": mf.overall.round(4).to_dict(),
            "disparity": mf.difference(method="between_groups").round(4).to_dict(),
        }
        _print_bias_table(sensitive_col, mf)

    return results


def _print_bias_table(feature: str, mf: MetricFrame):
    print(f"\n── Fairness Metrics by {feature} {'─'*30}")
    print(mf.by_group.to_string())
    diff = mf.difference(method="between_groups")
    print(f"\n  Disparity (max − min across groups):")
    for metric, val in diff.items():
        flag = " ⚠️  BIAS DETECTED" if val > 0.10 else ""
        print(f"    {metric:<25}: {val:.4f}{flag}")


def plot_bias(bias_results: dict):
    """
    Bar chart comparing FPR across age groups and gender.
    FPR = rate at which GOOD applicants are incorrectly rejected.
    A higher FPR for a demographic group signals potential discrimination.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fairness Analysis: False Positive Rate by Demographic Group",
                 fontsize=13, fontweight="bold")

    for ax, (feature, label) in zip(axes, [("age_group", "Age Group"), ("gender", "Gender")]):
        by_group = bias_results[feature]["by_group"]["false_positive_rate"]
        groups = list(by_group.keys())
        values = list(by_group.values())
        colors = ["#e74c3c" if v == max(values) else "#3498db" for v in values]

        bars = ax.bar(groups, values, color=colors, edgecolor="white", width=0.5)
        ax.axhline(
            bias_results[feature]["overall"]["false_positive_rate"],
            color="black", linestyle="--", linewidth=1.2, label="Overall FPR"
        )
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        disparity = bias_results[feature]["disparity"]["false_positive_rate"]
        ax.set_title(f"By {label}\n(FPR Disparity = {disparity:.1%})", fontsize=11)
        ax.set_ylabel("False Positive Rate\n(Good applicants incorrectly rejected)")
        ax.set_ylim(0, max(values) * 1.35)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("models/bias_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: models/bias_report.png")


def plot_approval_rates(bias_results: dict):
    """
    Companion chart: approval (selection) rate across groups.
    Helps distinguish whether disparate FPR translates to disparate outcomes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Selection Rate (Approval Rate) by Demographic Group",
                 fontsize=12, fontweight="bold")

    for ax, (feature, label) in zip(axes, [("age_group", "Age Group"), ("gender", "Gender")]):
        by_group = bias_results[feature]["by_group"]["selection_rate"]
        groups = list(by_group.keys())
        values = list(by_group.values())
        colors = ["#27ae60" if v == max(values) else "#95a5a6" for v in values]

        bars = ax.bar(groups, values, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(f"By {label}", fontsize=11)
        ax.set_ylabel("Approval Rate")
        ax.set_ylim(0, 1.1)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("models/bias_approval_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: models/bias_approval_rates.png")


# ─── 5. Save JSON Report ──────────────────────────────────────────────────────
def save_bias_json(bias_results: dict):
    """Serialise bias metrics to JSON for the Streamlit dashboard to read."""
    report = {}
    for feature, data in bias_results.items():
        report[feature] = {
            "by_group": {
                group: {k: float(v) for k, v in metrics.items()}
                for group, metrics in data["by_group"].items()
            },
            "overall": {k: float(v) for k, v in data["overall"].items()},
            "disparity": {k: float(v) for k, v in data["disparity"].items()},
        }
    with open("models/bias_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved: models/bias_report.json")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data and model...")
    X_test, y_test = load_test_set()
    pipeline, preprocessor, classifier = load_pipeline()
    feature_names = get_feature_names(preprocessor)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print(f"\nComputing SHAP values for {len(X_test)} test applicants...")
    explainer, explanation, X_transformed = run_shap(
        preprocessor, classifier, X_test, feature_names
    )
    print(f"Base value (expected log-odds): {float(explainer.expected_value):.4f}")

    # Pick one approved and one declined applicant for waterfall plots
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Most confident approval (lowest P(default) among true negatives)
    approved_idx = int(
        pd.Series(y_prob)[(y_pred == 0)].idxmin()
    )
    # Most confident decline (highest P(default) among true positives)
    declined_idx = int(
        pd.Series(y_prob)[(y_pred == 1)].idxmax()
    )

    print(f"\nWaterfall — Approved applicant (index {approved_idx}):")
    p_approve = plot_waterfall(explanation, approved_idx, "approved", pipeline, X_test)

    print(f"Waterfall — Declined applicant (index {declined_idx}):")
    p_decline = plot_waterfall(explanation, declined_idx, "declined", pipeline, X_test)

    print("\nBeeswarm summary plot (full test set):")
    plot_beeswarm(explanation)

    # ── Fairlearn Bias Analysis ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  FAIRLEARN BIAS ANALYSIS")
    print("="*55)
    bias_results = run_bias_analysis(pipeline, X_test, y_test)

    plot_bias(bias_results)
    plot_approval_rates(bias_results)
    save_bias_json(bias_results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PHASE 3 COMPLETE — Artifacts saved in ./models/")
    print("="*55)
    print("  models/shap_waterfall_approved.png")
    print("  models/shap_waterfall_declined.png")
    print("  models/shap_beeswarm.png")
    print("  models/bias_report.png")
    print("  models/bias_approval_rates.png")
    print("  models/bias_report.json")
