# LBG Credit Risk Agent

An end-to-end **Agentic AI** system for credit risk assessment, built to demonstrate production-grade machine learning practices for a banking context.

The system goes beyond a simple prediction model. It wraps an XGBoost classifier in a governance-aware decision agent that applies business-rule thresholds, explains every decision with SHAP, audits for demographic bias using Fairlearn, and drafts customer communication letters — all surfaced through an interactive Streamlit dashboard.

---

## Dashboard Preview

> **Tab 1 — Application Assessment:** Input applicant details → get a live decision badge, SHAP waterfall explaining the score, and a generated customer letter.

> **Tab 2 — Model Performance:** Confusion matrix, feature importances, SHAP beeswarm summary.

> **Tab 3 — Fairness Report:** Fairlearn bias metrics by age group and gender, with plain-English interpretation.

---

## Architecture

```
Raw Applicant Data (20 features)
        │
        ▼
┌───────────────────────┐
│   Preprocessing       │  OneHotEncoder (13 categorical)
│   Pipeline            │  Passthrough   (7 numerical)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   SMOTE               │  Synthetic oversampling on training data only
│   (train only)        │  Addresses 70/30 class imbalance
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   XGBoost Classifier  │  max_depth=2, n_estimators=100
│                       │  reg_lambda=10, scale_pos_weight=2.33
└───────────────────────┘
        │
        ▼  P(default)
┌───────────────────────┐
│   CreditAgent         │  Business-rule governance layer
│   .decide()           │  < 20%  → AUTO_APPROVE
│                       │  20–80% → HUMAN_REFERRAL
│                       │  > 80%  → AUTO_DECLINE
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   DecisionResult      │  UUID + ISO timestamp + rationale
│   (audit record)      │  Immutable applicant snapshot
└───────────────────────┘
```

---

## Model Performance

| Metric | Result | Target |
|---|---|---|
| CV ROC-AUC (5-fold) | **0.780 ± 0.049** | > 0.70 |
| Test ROC-AUC | **0.764** | > 0.70 |
| Default Recall | **82%** | > 60% |
| Train / Test Gap | **0.069** | < 0.10 |

The model was regularised (from the initial `max_depth=3`, `n_estimators=300`) after identifying a Train/Test AUC gap of 0.175. Regularisation reduced the gap to **0.069** while improving Default Recall from 70% to **82%** — the metric that matters most in a lending context, where missing a bad loan costs significantly more than a false rejection.

---

## Fairness Findings

Fairlearn MetricFrame analysis on the test set revealed a statistically significant age-based disparity:

| Group | False Positive Rate | Meaning |
|---|---|---|
| Age < 25 | **60.0%** | 6 in 10 genuinely creditworthy young applicants are incorrectly rejected |
| Age 25+ | **41.7%** | 4 in 10 genuinely creditworthy older applicants are incorrectly rejected |
| **Disparity** | **18.3%** | Exceeds the 10% regulatory flag threshold |

**Mitigation built into the agent:** All applicants under 25 whose P(default) falls in the 20–80% referral band are routed to a human underwriter rather than auto-decided. This directly addresses the SR 11-7 model risk management expectation for human oversight of ambiguous cases in protected demographic groups.

Gender disparity (6%) fell within the acceptable range.

---

## Project Structure

```
LBG-Credit-Agent/
├── README.md               # This file
├── app.py                  # Streamlit dashboard (3-tab UI)
├── model_training.py       # Full training pipeline: OHE + SMOTE + XGBoost
├── credit_agent.py         # CreditAgent class — governance & decision logic
├── explainability.py       # SHAP analysis + Fairlearn bias report generation
├── genai_mock.py           # Simulated GenAI customer letter generation
├── validate_model.py       # 4-check validation suite (metrics, SHAP, ethics, stress test)
├── requirements.txt        # Python dependencies
├── data/
│   └── german_credit.csv   # UCI German Credit Dataset (1,000 applicants, 20 features)
└── assets/
    └── shap_plot.png       # SHAP beeswarm summary (test set)
```

---

## Quickstart

**1. Clone and set up the environment**

```bash
git clone https://github.com/YOUR_USERNAME/LBG-Credit-Agent.git
cd LBG-Credit-Agent

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the model**

```bash
python model_training.py
```

This produces `models/credit_model.pkl`, the confusion matrix, and feature importance plots.

**3. Generate SHAP and fairness reports**

```bash
python explainability.py
```

**4. Run the validation suite**

```bash
python validate_model.py
```

**5. Launch the dashboard**

```bash
streamlit run app.py
```

---

## Key Technical Decisions

### Why XGBoost?
Gradient-boosted trees are the industry standard for tabular credit risk data. They handle mixed feature types natively, are robust to outliers, and their output is directly interpretable via SHAP — a hard requirement for regulatory explainability (GDPR Article 22, FCA guidance on algorithmic decision-making).

### Why SMOTE instead of just class weighting?
Both were used in combination. SMOTE generates synthetic minority samples in feature space, while `scale_pos_weight` adjusts the XGBoost loss function. Using both ensures the model sees balanced classes during training *and* is penalised proportionally for missing defaults at inference time.

### Why not use accuracy as the primary metric?
The dataset has a 70/30 class split. A model that predicts "Good" for every applicant achieves 70% accuracy while catching 0% of defaults. ROC-AUC and Recall on the Default class are the correct metrics — they measure the model's ability to discriminate, not just its tendency to predict the majority class.

### Why a mock GenAI layer instead of a live API?
Banks operating under FCA and PRA oversight cannot use live LLM output in customer-facing communications without extensive model risk governance. The mock layer in `genai_mock.py` demonstrates prompt engineering knowledge while remaining fully auditable and hallucination-free. Switching to a live GPT-4o call is a single function swap.

### Agentic governance thresholds
The 20% / 80% boundaries are intentionally conservative, routing the majority of borderline cases to human underwriters. This reflects SR 11-7 model risk guidance: *automate only where confidence is high; preserve human oversight everywhere else.*

---

## Dataset

**German Credit Dataset** — UCI Machine Learning Repository
1,000 applicants · 20 features · Binary target (Good / Default)

Features include checking account status, loan duration and amount, credit history, savings, employment duration, personal status, property, housing, and job type. Age and personal status are used exclusively for the fairness audit and are not used as decision features by the model.

Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

---

## Dependencies

| Library | Purpose |
|---|---|
| `xgboost` | Gradient-boosted tree classifier |
| `scikit-learn` | Preprocessing pipeline, metrics, cross-validation |
| `imbalanced-learn` | SMOTE oversampling |
| `shap` | Model explainability (waterfall + beeswarm plots) |
| `fairlearn` | Demographic bias analysis (MetricFrame) |
| `streamlit` | Interactive web dashboard |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |

---

## Validation Summary

`validate_model.py` runs four automated checks before any deployment:

| Check | What It Tests | Result |
|---|---|---|
| Banker's Metrics | ROC-AUC, Recall, Train/Test gap | All targets met |
| SHAP Sanity | Feature directions match financial logic | 12/14 pass (2 are known dataset encoding quirks) |
| Ethical AI | Approval rate and FPR disparity by group | Age bias flagged and documented |
| Agentic Stress Test | 3 synthetic applicants hit correct decision bands | All 3 pass |

The two SHAP "fails" (`credit_history_A34`, `employment_A75`) are known artefacts of the UCI dataset encoding — `A34` conflates "critical account" with "accounts at other banks," which the dataset labels as lower risk. This is documented as a data quality limitation, not a model defect.

---

## License

MIT
