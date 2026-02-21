"""
train_model.py — Phase 1: German Credit Risk Model
Trains an XGBoost classifier on the German Credit Dataset with:
  - Full categorical encoding (OneHotEncoder)
  - SMOTE oversampling for class imbalance
  - Hyperparameter tuning via cross-validation
  - Saves model + preprocessor artifacts to /models
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Column Names ─────────────────────────────────────────────────────────
# From the UCI dataset documentation
COLUMNS = [
    "checking_status",       # A1x — status of existing checking account
    "duration",              # in months
    "credit_history",        # A3x
    "purpose",               # A4x
    "credit_amount",         # in DM
    "savings_status",        # A6x — savings account/bonds
    "employment",            # A7x — present employment since
    "installment_commitment",# installment rate as % of disposable income
    "personal_status",       # A9x — personal status and sex
    "other_parties",         # A10x — other debtors/guarantors
    "residence_since",       # present residence since (years)
    "property_magnitude",    # A12x
    "age",                   # in years
    "other_payment_plans",   # A14x
    "housing",               # A15x
    "existing_credits",      # number of existing credits at this bank
    "job",                   # A17x
    "num_dependents",        # number of people being liable to provide maintenance for
    "own_telephone",         # A19x
    "foreign_worker",        # A20x
    "target",                # 1 = Good, 2 = Bad
]

# Categorical and numerical feature split
CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
]
NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents",
]

# ─── 2. Load Data ─────────────────────────────────────────────────────────────
def load_data(path: str = "data/german.data") -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS)
    # Map target: 1 (Good) → 0, 2 (Bad) → 1  (Default = positive class)
    df["target"] = df["target"].map({1: 0, 2: 1})
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts()}")
    print(f"  Good (0): {(df['target']==0).sum()} | Bad/Default (1): {(df['target']==1).sum()}")
    return df

# ─── 3. Preprocessing ─────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """One-hot encodes categoricals; passes numericals through."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERICAL_COLS),
        ]
    )

# ─── 4. Train ─────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified split preserves class ratio in both train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    preprocessor = build_preprocessor()

    # scale_pos_weight = #negative / #positive (handles imbalance inside XGB too)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={neg}, pos={pos})")

    xgb = XGBClassifier(
        n_estimators=100,     # reduced from 300 — stops learning noise
        max_depth=2,          # reduced from 3 — simpler trees generalise better
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10,        # L2 regularisation — penalises complex rules
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # Full pipeline: preprocess → SMOTE → XGBoost
    # SMOTE is applied ONLY on training data inside each CV fold (no leakage)
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("classifier", xgb),
    ])

    # 5-fold stratified cross-validation on training set
    print("\nRunning 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on full training data
    print("\nFitting final model on full training set...")
    pipeline.fit(X_train, y_train)

    # ─── 5. Evaluation ────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n── Test Set Results ──────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Good (0)", "Default (1)"]))
    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc:.4f}")

    # Confusion matrix plot
    os.makedirs("models", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Default"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix  (ROC-AUC = {roc:.3f})")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    print("Saved: models/confusion_matrix.png")

    # Feature importances (from XGBoost, after OHE)
    ohe_features = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    all_features = ohe_features + NUMERICAL_COLS
    importances = pipeline.named_steps["classifier"].feature_importances_

    fi_df = (
        pd.DataFrame({"feature": all_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
    ax2.set_xlabel("Importance (F-score)")
    ax2.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("models/feature_importances.png", dpi=150)
    print("Saved: models/feature_importances.png")

    return pipeline, X_test, y_test, roc

# ─── 6. Save Artifacts ────────────────────────────────────────────────────────
def save_artifacts(pipeline):
    with open("models/credit_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("Saved: models/credit_model.pkl")

    # Also save column lists for the Streamlit app to use
    metadata = {
        "categorical_cols": CATEGORICAL_COLS,
        "numerical_cols": NUMERICAL_COLS,
        "all_cols": CATEGORICAL_COLS + NUMERICAL_COLS,
        "columns": COLUMNS[:-1],  # exclude target
    }
    with open("models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("Saved: models/metadata.pkl")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data("data/german.data")
    pipeline, X_test, y_test, roc = train(df)
    save_artifacts(pipeline)
    print(f"\nPhase 1 complete. Final ROC-AUC: {roc:.4f}")
    print("Artifacts saved in ./models/")
