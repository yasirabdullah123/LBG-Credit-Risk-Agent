"""
credit_agent.py — Phase 2: Agentic Decision Layer
Wraps the XGBoost model in a governance-aware agent that:
  - Accepts raw applicant data
  - Produces a calibrated default probability
  - Applies business-rule thresholds to make a decision
  - Returns a structured DecisionResult with full audit trail
"""

import pickle
import uuid
import datetime
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Tuple


# ─── Decision Thresholds ──────────────────────────────────────────────────────
# Standard practice: only automate the high-confidence cases; route ambiguous
# applications to a human underwriter. These values are intentionally conservative
# to reflect regulatory expectations (SR 11-7 model risk guidance).
THRESHOLD_AUTO_APPROVE = 0.20   # P(default) < 20%  → clearly safe
THRESHOLD_AUTO_DECLINE = 0.80   # P(default) > 80%  → clearly risky
# Everything in between → human review


# ─── Structured Output ────────────────────────────────────────────────────────
@dataclass
class DecisionResult:
    """Immutable record of a single credit decision. Suitable for audit logging."""
    decision_id: str          # unique UUID for this decision
    timestamp: str            # ISO-8601 UTC timestamp
    decision: str             # AUTO_APPROVE | HUMAN_REFERRAL | AUTO_DECLINE
    risk_level: str           # Low | Medium | High
    default_probability: float
    confidence_band: str      # probability expressed as a readable range
    rationale: str            # plain-English explanation for compliance
    applicant_snapshot: dict  # copy of input features (immutable audit record)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            f"  CREDIT DECISION  [{self.decision_id[:8]}]",
            "=" * 55,
            f"  Decision     : {self.decision}",
            f"  Risk Level   : {self.risk_level}",
            f"  P(Default)   : {self.default_probability:.1%}  ({self.confidence_band})",
            f"  Rationale    : {self.rationale}",
            f"  Timestamp    : {self.timestamp}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ─── CreditAgent ──────────────────────────────────────────────────────────────
class CreditAgent:
    """
    Agentic credit decision system.

    The agent enforces three governance principles:
      1. Transparency  — every decision carries a rationale and probability.
      2. Human oversight — ambiguous cases are never auto-decided.
      3. Auditability  — each decision gets a unique ID and timestamp.

    Usage:
        agent = CreditAgent()
        result = agent.evaluate(applicant_dict)
        print(result)
    """

    def __init__(self, model_path: str = "models/credit_model.pkl"):
        with open(model_path, "rb") as f:
            self._pipeline = pickle.load(f)

        # Load feature column metadata saved during training
        with open("models/metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        self._feature_cols = meta["columns"]  # ordered list, target excluded

    # ── Core Decision Logic ───────────────────────────────────────────────────
    def decide(self, probability: float) -> Tuple[str, str, str]:
        """
        Maps a P(default) probability to a governance decision.

        Returns:
            (decision, risk_level, rationale)
        """
        p = probability

        if p < THRESHOLD_AUTO_APPROVE:
            return (
                "AUTO_APPROVE",
                "Low Risk",
                f"P(default) = {p:.1%} is below the {THRESHOLD_AUTO_APPROVE:.0%} "
                "auto-approval threshold. Loan approved automatically.",
            )
        elif p > THRESHOLD_AUTO_DECLINE:
            return (
                "AUTO_DECLINE",
                "High Risk",
                f"P(default) = {p:.1%} exceeds the {THRESHOLD_AUTO_DECLINE:.0%} "
                "auto-decline threshold. Application rejected automatically.",
            )
        else:
            return (
                "HUMAN_REFERRAL",
                "Medium Risk",
                f"P(default) = {p:.1%} falls in the referral band "
                f"({THRESHOLD_AUTO_APPROVE:.0%}–{THRESHOLD_AUTO_DECLINE:.0%}). "
                "Escalated to a human underwriter for final determination.",
            )

    # ── Full Evaluation Pipeline ──────────────────────────────────────────────
    def evaluate(self, applicant: dict) -> DecisionResult:
        """
        End-to-end evaluation: raw applicant features → structured decision.

        Args:
            applicant: dict with keys matching the German Credit feature set.

        Returns:
            DecisionResult dataclass (printable, serialisable to dict).
        """
        # Build a single-row DataFrame in the exact column order the model expects
        df = pd.DataFrame([applicant])[self._feature_cols]

        # Model returns P(class=1) i.e. P(default)
        prob = float(self._pipeline.predict_proba(df)[0, 1])

        decision, risk_level, rationale = self.decide(prob)

        # Format a human-readable probability band
        band = self._probability_band(prob)

        return DecisionResult(
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            decision=decision,
            risk_level=risk_level,
            default_probability=round(prob, 4),
            confidence_band=band,
            rationale=rationale,
            applicant_snapshot=dict(applicant),  # immutable copy for audit
        )

    # ── Batch Evaluation ─────────────────────────────────────────────────────
    def evaluate_batch(self, applicants: list[dict]) -> list[DecisionResult]:
        """Evaluate a list of applicant dicts. Returns one DecisionResult per row."""
        return [self.evaluate(a) for a in applicants]

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _probability_band(p: float) -> str:
        """Converts a probability into a readable risk band label."""
        if p < 0.10:
            return "< 10%  (Very Low)"
        elif p < 0.20:
            return "10–20% (Low)"
        elif p < 0.40:
            return "20–40% (Moderate)"
        elif p < 0.60:
            return "40–60% (Elevated)"
        elif p < 0.80:
            return "60–80% (High)"
        else:
            return "> 80%  (Very High)"

    @staticmethod
    def threshold_summary() -> str:
        """Prints the governance thresholds for documentation / UI display."""
        return (
            f"Decision thresholds:\n"
            f"  AUTO_APPROVE   : P(default) < {THRESHOLD_AUTO_APPROVE:.0%}\n"
            f"  HUMAN_REFERRAL : {THRESHOLD_AUTO_APPROVE:.0%} ≤ P(default) ≤ {THRESHOLD_AUTO_DECLINE:.0%}\n"
            f"  AUTO_DECLINE   : P(default) > {THRESHOLD_AUTO_DECLINE:.0%}"
        )


# ─── Quick Smoke-Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = CreditAgent()

    print(agent.threshold_summary())
    print()

    # Three synthetic applicants hitting each decision band
    test_cases = [
        {
            "label": "Clearly Good — salaried, short loan, savings present",
            "applicant": {
                "checking_status": "A14",         # no checking account
                "duration": 6,
                "credit_history": "A34",          # no credits / all paid back
                "purpose": "A43",                 # furniture/equipment
                "credit_amount": 1500,
                "savings_status": "A65",          # >= 1000 DM
                "employment": "A75",              # >= 7 years
                "installment_commitment": 2,
                "personal_status": "A93",         # male single
                "other_parties": "A101",          # none
                "residence_since": 4,
                "property_magnitude": "A121",     # real estate
                "age": 45,
                "other_payment_plans": "A143",    # none
                "housing": "A152",                # own
                "existing_credits": 1,
                "job": "A173",                    # skilled
                "num_dependents": 1,
                "own_telephone": "A192",          # yes
                "foreign_worker": "A201",         # no
            },
        },
        {
            "label": "Borderline — medium loan, mixed signals",
            "applicant": {
                "checking_status": "A12",         # 0 <= ... < 200 DM
                "duration": 18,
                "credit_history": "A32",          # existing credits paid till now
                "purpose": "A42",                 # furniture/equipment
                "credit_amount": 3500,
                "savings_status": "A62",          # 100 <= ... < 500 DM
                "employment": "A73",              # 1 <= ... < 4 years
                "installment_commitment": 3,
                "personal_status": "A93",         # male single
                "other_parties": "A101",          # none
                "residence_since": 2,
                "property_magnitude": "A122",     # building society savings
                "age": 34,
                "other_payment_plans": "A143",    # none
                "housing": "A152",                # own
                "existing_credits": 1,
                "job": "A173",                    # skilled
                "num_dependents": 1,
                "own_telephone": "A192",          # yes
                "foreign_worker": "A201",         # no
            },
        },
        {
            "label": "High Risk — overextended, no savings, bad history",
            "applicant": {
                "checking_status": "A11",         # < 0 DM
                "duration": 48,
                "credit_history": "A30",          # no credits taken / all paid
                "purpose": "A410",                # others
                "credit_amount": 15000,
                "savings_status": "A61",          # < 100 DM
                "employment": "A71",              # unemployed
                "installment_commitment": 4,
                "personal_status": "A91",         # male divorced/separated
                "other_parties": "A103",          # guarantor
                "residence_since": 1,
                "property_magnitude": "A124",     # no known property
                "age": 23,
                "other_payment_plans": "A141",    # bank
                "housing": "A151",                # rent
                "existing_credits": 3,
                "job": "A171",                    # unemployed / unskilled - non-resident
                "num_dependents": 2,
                "own_telephone": "A191",
                "foreign_worker": "A202",         # yes
            },
        },
    ]

    for case in test_cases:
        print(f"\nApplicant: {case['label']}")
        result = agent.evaluate(case["applicant"])
        print(result)
