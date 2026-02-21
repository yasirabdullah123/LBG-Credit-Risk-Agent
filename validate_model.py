"""
validate_model.py — Model Validation Report
Runs 4 checks before production sign-off:

  CHECK 1 — Banker's Metrics   : ROC-AUC, Recall, + Train vs Test AUC (overfitting)
  CHECK 2 — SHAP Sanity        : Confirm feature directions match financial logic
  CHECK 3 — Ethical AI         : Approval & rejection rates by Age Group and Gender
  CHECK 4 — Agentic Stress Test: 3 synthetic applicants hit the right decision bands
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
import shap

from credit_agent import CreditAgent

warnings.filterwarnings("ignore")

# ─── Column Definitions ───────────────────────────────────────────────────────
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

MALE_CODES   = {"A91", "A93", "A94"}
FEMALE_CODES = {"A92", "A95"}

SEP  = "=" * 60
SEP2 = "-" * 60

# ─── Load ─────────────────────────────────────────────────────────────────────
def load_data_and_splits():
    df = pd.read_csv("data/german.data", sep=" ", header=None, names=COLUMNS)
    df["target"] = df["target"].map({1: 0, 2: 1})
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True))

def load_pipeline():
    with open("models/credit_model.pkl", "rb") as f:
        return pickle.load(f)


# ─── CHECK 1: Banker's Metrics ────────────────────────────────────────────────
def check_1_bankers_metrics(pipeline, X_train, X_test, y_train, y_test):
    print(f"\n{SEP}")
    print("  CHECK 1 — BANKER'S METRICS")
    print(SEP)

    # Test-set predictions
    y_pred  = pipeline.predict(X_test)
    y_prob  = pipeline.predict_proba(X_test)[:, 1]

    # Training-set predictions (checks for overfitting)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]

    test_auc  = roc_auc_score(y_test,  y_prob)
    train_auc = roc_auc_score(y_train, y_train_prob)
    gap       = train_auc - test_auc

    print("\n  Classification Report (Test Set):")
    print(classification_report(y_test, y_pred,
                                target_names=["Good Loan (0)", "Default (1)"],
                                digits=3))

    print(f"  ROC-AUC  |  Train : {train_auc:.4f}   Test : {test_auc:.4f}   Gap : {gap:.4f}")

    # Benchmarks
    recall_class1 = float(
        pd.Series(y_pred[y_test == 1] == 1).mean()
    )
    auc_grade  = "EXCELLENT" if test_auc >= 0.80 else ("ACCEPTABLE" if test_auc >= 0.70 else "BELOW TARGET")
    recall_grade = "PASS" if recall_class1 >= 0.60 else "BELOW TARGET"
    overfit_grade = "OK" if gap < 0.10 else "WARNING — possible overfitting"

    print(f"\n  Benchmark Summary:")
    print(f"    Test ROC-AUC  : {test_auc:.4f}   → {auc_grade}  (target: > 0.70)")
    print(f"    Default Recall: {recall_class1:.4f}  → {recall_grade}  (target: >= 0.60)")
    print(f"    Train/Test Gap: {gap:.4f}   → {overfit_grade}  (target: < 0.10)")

    return y_pred, y_prob


# ─── CHECK 2: SHAP Sanity ─────────────────────────────────────────────────────
def check_2_shap_sanity(pipeline, X_test):
    """
    Correct methodology:
      - Numerical features: Spearman rho between feature value and SHAP value.
        A risk-increasing feature (duration) should have rho > 0.
      - OHE binary features: mean SHAP only for rows where the feature is active (value=1).
        Mean SHAP across ALL rows is wrong for OHE — most rows are 0, which dominates.
    """
    print(f"\n{SEP}")
    print("  CHECK 2 --- SHAP SANITY (Feature Direction Validation)")
    print(SEP)
    print("\n  Computing SHAP values for test set...")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]

    ohe_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    feature_names = ohe_names + NUMERICAL_COLS

    X_transformed = preprocessor.transform(X_test)
    explainer  = shap.TreeExplainer(classifier)
    shap_vals  = explainer.shap_values(X_transformed)
    shap_df    = pd.DataFrame(shap_vals, columns=feature_names)
    feat_df    = pd.DataFrame(X_transformed, columns=feature_names)

    # Numerical features: Spearman correlation (feature value vs SHAP value)
    print(f"\n  Numerical Features  (Spearman rho: feature value vs SHAP value)")
    print(f"  {'Feature':<30}  {'Spearman rho':>13}  {'Direction':>10}  Logic")
    print(f"  {SEP2}")

    numerical_checks = [
        ("duration",               "+", "Longer loan -> more default risk"),
        ("credit_amount",          "+", "Larger debt -> more default risk"),
        ("age",                    "-", "Older applicant -> less default risk"),
        ("installment_commitment", "+", "Higher repayment % of income -> more stress"),
        ("existing_credits",       "+", "More existing debts -> more risk"),
    ]
    all_pass = True
    for feat, expected_sign, intuition in numerical_checks:
        rho = feat_df[feat].corr(shap_df[feat], method="spearman")
        actual_sign = "+" if rho >= 0 else "-"
        passed = actual_sign == expected_sign
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {feat:<30}  {rho:>+13.4f}  {status:>10}  {intuition}")

    # OHE binary features: mean SHAP only when the feature is active (value=1)
    print(f"\n  Categorical Features  (mean SHAP when feature is active)")
    print(f"  {'Feature':<35}  {'Active SHAP':>12}  {'Direction':>10}  Logic")
    print(f"  {SEP2}")

    cat_checks = [
        ("checking_status_A11", "+", "Overdrawn account -> high risk"),
        ("checking_status_A14", "-", "No checking account -> lower risk"),
        ("employment_A71",      "+", "Unemployed -> higher risk"),
        ("employment_A75",      "-", "7+ yrs employed -> lower risk"),
        ("savings_status_A61",  "+", "< 100 DM savings -> higher risk"),
        ("savings_status_A65",  "-", "Large/unknown savings -> lower risk"),
        ("credit_history_A34",  "+", "Critical credit history -> higher risk"),
        ("credit_history_A32",  "-", "Existing credits paid duly -> lower risk"),
    ]
    for feat, expected_sign, intuition in cat_checks:
        if feat not in shap_df.columns:
            print(f"  {feat:<35}  {'N/A':>12}  {'SKIP':>10}  (not in OHE output)")
            continue
        active_mask = feat_df[feat] == 1
        n_active = int(active_mask.sum())
        if n_active == 0:
            print(f"  {feat:<35}  {'N/A (0 active)':>12}  {'SKIP':>10}  {intuition}")
            continue
        mean_active_shap = shap_df.loc[active_mask, feat].mean()
        actual_sign = "+" if mean_active_shap >= 0 else "-"
        passed = actual_sign == expected_sign
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {feat:<35}  {mean_active_shap:>+12.4f}  {status:>10}  {intuition} (n={n_active})")

    verdict = "ALL CHECKS PASSED --- model directions match financial logic." if all_pass \
              else "SOME CHECKS FAILED --- review flagged features before deployment."
    print(f"\n  Verdict: {verdict}")


# ─── CHECK 3: Ethical AI ──────────────────────────────────────────────────────
def check_3_ethical_ai(pipeline, X_test, y_test):
    print(f"\n{SEP}")
    print("  CHECK 3 — ETHICAL AI (Approval & Rejection Rates by Group)")
    print(SEP)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results["predicted_default"] = y_pred
    results["actual_default"]    = y_test.values
    results["p_default"]         = y_prob

    # Age group
    results["age_group"] = results["age"].apply(
        lambda x: "Young (<25)" if x < 25 else "Older (25+)"
    )
    # Gender (derived from personal_status)
    results["gender"] = results["personal_status"].apply(
        lambda s: "Male" if s in MALE_CODES else "Female"
    )

    # ── Approval Rate ──────────────────────────────────────────────────────
    print("\n  Predicted Default Rate (lower = more approvals):")
    print(f"\n  {'Group':<20}  {'Default Rate':>14}  {'Approval Rate':>15}  {'N':>5}")
    print(f"  {SEP2}")

    THRESHOLD = 0.10  # > 10pp gap triggers a flag

    for col in ["age_group", "gender"]:
        groups = results.groupby(col)["predicted_default"].agg(["mean", "count"])
        groups.columns = ["default_rate", "n"]
        groups["approval_rate"] = 1 - groups["default_rate"]
        max_dr = groups["default_rate"].max()
        min_dr = groups["default_rate"].min()
        disparity = max_dr - min_dr
        flag = "  ⚠️  DISPARITY DETECTED" if disparity > THRESHOLD else ""

        print(f"\n  By {col}:{flag}")
        for group, row in groups.iterrows():
            print(f"    {group:<18}  {row['default_rate']:>13.1%}  {row['approval_rate']:>14.1%}  {int(row['n']):>5}")
        print(f"    {'Disparity':<18}  {disparity:>13.1%}")

    # ── False Positive Rate (unfair rejection of good applicants) ──────────
    print("\n  False Positive Rate (good applicants incorrectly predicted as defaulters):")
    print(f"  {'Group':<20}  {'FPR':>8}  Meaning")
    print(f"  {SEP2}")

    for col in ["age_group", "gender"]:
        for group, grp in results.groupby(col):
            good = grp[grp["actual_default"] == 0]
            fpr = (good["predicted_default"] == 1).mean() if len(good) > 0 else 0
            print(f"  {str(group):<20}  {fpr:>7.1%}  of genuinely good {group} applicants rejected")

    print(f"\n  Key Insight:")
    young = results[results["age_group"] == "Young (<25)"]
    older = results[results["age_group"] == "Older (25+)"]
    y_fpr = (young[young["actual_default"]==0]["predicted_default"]==1).mean()
    o_fpr = (older[older["actual_default"]==0]["predicted_default"]==1).mean()
    print(f"    Young applicants wrongly rejected: {y_fpr:.1%}")
    print(f"    Older applicants wrongly rejected: {o_fpr:.1%}")
    if (y_fpr - o_fpr) > 0.10:
        print(f"    ⚠️  Age bias confirmed ({y_fpr-o_fpr:.1%} gap).")
        print(f"    Mitigation: route all <25 applicants in the 20–80% band to human review.")
    else:
        print(f"    Age FPR gap: {abs(y_fpr-o_fpr):.1%} — within acceptable range.")


# ─── CHECK 4: Agentic Stress Test ─────────────────────────────────────────────
def check_4_agentic_stress_test():
    print(f"\n{SEP}")
    print("  CHECK 4 — AGENTIC STRESS TEST (3 Synthetic Applicants)")
    print(SEP)

    agent = CreditAgent()

    stress_cases = [
        {
            "name": "Mr. Perfect",
            "description": "High savings, short loan, long employment, owns home",
            "expected_decision": "AUTO_APPROVE",
            "expected_p_max": 0.20,
            "applicant": {
                "checking_status": "A13",   # > 200 DM
                "duration": 6,
                "credit_history": "A34",    # all credits paid
                "purpose": "A43",           # radio/TV
                "credit_amount": 1000,
                "savings_status": "A65",    # >= 1000 DM
                "employment": "A75",        # >= 7 years
                "installment_commitment": 1,
                "personal_status": "A93",   # male single
                "other_parties": "A101",    # none
                "residence_since": 4,
                "property_magnitude": "A121", # real estate
                "age": 50,
                "other_payment_plans": "A143", # none
                "housing": "A152",          # own
                "existing_credits": 1,
                "job": "A174",              # management / highly qualified
                "num_dependents": 1,
                "own_telephone": "A192",    # yes
                "foreign_worker": "A201",   # no
            },
        },
        {
            "name": "Mr. Risky",
            "description": "Unemployed, huge loan, no savings, overdrawn, renting",
            "expected_decision": "AUTO_DECLINE",
            "expected_p_min": 0.80,
            "applicant": {
                "checking_status": "A11",   # < 0 DM (overdrawn)
                "duration": 60,
                "credit_history": "A30",    # no credits taken / critical history
                "purpose": "A410",          # other
                "credit_amount": 18000,
                "savings_status": "A61",    # < 100 DM
                "employment": "A71",        # unemployed
                "installment_commitment": 4,
                "personal_status": "A91",   # male divorced
                "other_parties": "A101",
                "residence_since": 1,
                "property_magnitude": "A124", # no known property
                "age": 23,
                "other_payment_plans": "A141", # bank
                "housing": "A151",          # rent
                "existing_credits": 4,
                "job": "A171",              # unemployed / unskilled non-resident
                "num_dependents": 2,
                "own_telephone": "A191",    # no
                "foreign_worker": "A202",   # yes
            },
        },
        {
            "name": "Ms. Borderline",
            "description": "Average profile — medium loan, some savings, renting, mid-employment",
            "expected_decision": "HUMAN_REFERRAL",
            "expected_p_range": (0.20, 0.80),
            "applicant": {
                "checking_status": "A12",   # 0–200 DM
                "duration": 18,
                "credit_history": "A32",    # existing credits paid duly
                "purpose": "A42",           # furniture
                "credit_amount": 3500,
                "savings_status": "A62",    # 100–499 DM
                "employment": "A73",        # 1–3 years
                "installment_commitment": 3,
                "personal_status": "A93",   # male single (proxy for borderline)
                "other_parties": "A101",
                "residence_since": 2,
                "property_magnitude": "A122", # building society savings
                "age": 34,
                "other_payment_plans": "A143", # none
                "housing": "A151",          # rent
                "existing_credits": 1,
                "job": "A173",              # skilled
                "num_dependents": 1,
                "own_telephone": "A192",
                "foreign_worker": "A201",
            },
        },
    ]

    all_pass = True
    print()
    for case in stress_cases:
        result = agent.evaluate(case["applicant"])
        p      = result.default_probability
        dec    = result.decision

        # Determine pass/fail
        if "expected_p_max" in case:
            passed = dec == case["expected_decision"] and p < case["expected_p_max"]
        elif "expected_p_min" in case:
            passed = dec == case["expected_decision"] and p > case["expected_p_min"]
        else:
            lo, hi = case["expected_p_range"]
            passed = dec == case["expected_decision"] and lo <= p <= hi

        status = "PASS" if passed else "FAIL ⚠"
        if not passed:
            all_pass = False

        print(f"  [{status}] {case['name']}  —  {case['description']}")
        print(f"         P(default)={p:.1%}  |  Decision: {dec}  |  Expected: {case['expected_decision']}")
        print()

    verdict = "All 3 stress tests passed." if all_pass else "One or more stress tests failed — check thresholds."
    print(f"  Verdict: {verdict}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  CREDIT RISK MODEL — VALIDATION REPORT")
    print(SEP)

    print("\nLoading data and model...")
    X_train, X_test, y_train, y_test = load_data_and_splits()
    pipeline = load_pipeline()

    y_pred, y_prob = check_1_bankers_metrics(pipeline, X_train, X_test, y_train, y_test)
    check_2_shap_sanity(pipeline, X_test)
    check_3_ethical_ai(pipeline, X_test, y_test)
    check_4_agentic_stress_test()

    print(f"\n{SEP}")
    print("  VALIDATION COMPLETE")
    print(SEP)
    print()
