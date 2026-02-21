"""
genai_mock.py — Simulated GenAI Customer Communication Layer

Simulates what a live LLM call would produce using deterministic templates.
This approach is preferred in production banking for three reasons:
  1. No hallucination risk — output is fully auditable and pre-approved by compliance
  2. Zero latency / zero API cost — runs locally
  3. Prompt engineering still demonstrated — the prompt logic is documented below

In a real deployment, you would swap generate_explanation_email() to call:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": _build_prompt(customer_name, decision, shap_reason)}]
    )
    return response.choices[0].message.content
"""

import random


# ─── Prompt Template (what we WOULD send to an LLM) ──────────────────────────
def _build_prompt(customer_name: str, decision: str, shap_reason: str) -> str:
    """
    Documents the prompt engineering logic even when using mock responses.
    Shows interviewers you understand how to structure LLM instructions.
    """
    return (
        f"You are a professional banking communication assistant for Lloyds Banking Group. "
        f"Write a formal, empathetic letter to a customer named {customer_name}. "
        f"The credit application decision is: {decision}. "
        f"The primary risk factor identified by our AI model was: {shap_reason}. "
        f"Guidelines: "
        f"(1) Do not use jargon. "
        f"(2) Be empathetic but factual. "
        f"(3) Provide one actionable piece of advice. "
        f"(4) Do not promise a different outcome in the future. "
        f"(5) Keep the letter under 120 words."
    )


# ─── Template Bank ────────────────────────────────────────────────────────────
_DECLINE_TEMPLATES = [
    (
        "Dear {name},\n\n"
        "Thank you for your application to Lloyds Banking Group. After a thorough review "
        "of your credit profile, we are unable to approve your request at this time.\n\n"
        "The primary factor in this decision was: **{reason}**.\n\n"
        "We recommend requesting a copy of your credit report from Experian or Equifax to "
        "check for any inaccuracies. Addressing this factor may improve your eligibility "
        "for future applications.\n\n"
        "If you would like to discuss this decision, please contact our team.\n\n"
        "Sincerely,\n"
        "Lloyds Risk & Underwriting Team"
    ),
    (
        "Hello {name},\n\n"
        "We appreciate your interest in banking with Lloyds. Having carefully assessed "
        "your application against our current lending criteria, we are not able to proceed "
        "on this occasion.\n\n"
        "Our model identified **{reason}** as the most significant contributing factor.\n\n"
        "We would suggest speaking with a financial advisor about steps to strengthen your "
        "application profile before reapplying.\n\n"
        "Best regards,\n"
        "Automated Underwriting Agent — Lloyds Banking Group"
    ),
    (
        "Dear {name},\n\n"
        "Thank you for choosing Lloyds. Unfortunately, following a review of your "
        "financial profile, we cannot offer you credit at this time.\n\n"
        "The key risk indicator flagged was: **{reason}**. Reducing this risk factor — "
        "for example, by reducing existing credit commitments — may support a future "
        "application.\n\n"
        "This decision was made by an automated system. You have the right to request "
        "a human review.\n\n"
        "Kind regards,\n"
        "Lloyds Credit Team"
    ),
]

_REFER_TEMPLATES = [
    (
        "Dear {name},\n\n"
        "Thank you for your application. We have received your request and it is currently "
        "being reviewed by one of our specialist underwriters.\n\n"
        "Our automated system flagged **{reason}** as a factor requiring additional "
        "context — this is a routine step and does not indicate a negative outcome.\n\n"
        "A member of our team will be in touch within 2 business days.\n\n"
        "Sincerely,\n"
        "Lloyds Underwriting Team"
    ),
    (
        "Hello {name},\n\n"
        "Your credit application is progressing. Because **{reason}** fell into a range "
        "that benefits from a human review, your case has been escalated to a senior "
        "underwriter for a fair and thorough assessment.\n\n"
        "You do not need to take any action at this stage. We aim to respond within "
        "24–48 business hours.\n\n"
        "Best,\n"
        "Lloyds Credit Operations"
    ),
]

_APPROVE_TEMPLATES = [
    (
        "Dear {name},\n\n"
        "Congratulations! We are pleased to confirm that your credit application has been "
        "approved. Your profile demonstrated strong indicators, including **{reason}**.\n\n"
        "Your agreement documents will be sent to your registered email address. "
        "Please review them carefully before signing.\n\n"
        "Welcome, and thank you for banking with Lloyds.\n\n"
        "Sincerely,\n"
        "Lloyds Credit Team"
    ),
    (
        "Hello {name},\n\n"
        "Great news — your loan application has been approved! Our assessment found "
        "your application to be low risk, supported by **{reason}**.\n\n"
        "Next steps: check your inbox for your offer letter within 1 business day.\n\n"
        "Thank you for choosing Lloyds Banking Group.\n\n"
        "Best regards,\n"
        "Automated Underwriting Agent"
    ),
]


# ─── Public API ───────────────────────────────────────────────────────────────
def generate_explanation_email(
    customer_name: str,
    decision: str,
    shap_reason: str,
    seed: int | None = None,
) -> tuple[str, str]:
    """
    Generates a customer communication email based on the credit decision.

    Args:
        customer_name : Applicant's first name.
        decision      : One of 'AUTO_APPROVE', 'HUMAN_REFERRAL', 'AUTO_DECLINE'.
        shap_reason   : Plain-English description of the top SHAP feature
                        (e.g. "Checking Account: Overdrawn (< 0 DM)").
        seed          : Optional random seed for reproducibility in tests.

    Returns:
        (email_body, prompt_used)  — email text + the prompt we WOULD send to an LLM.
    """
    if seed is not None:
        random.seed(seed)

    prompt = _build_prompt(customer_name, decision, shap_reason)

    # Select template pool based on decision
    if decision == "AUTO_DECLINE":
        template = random.choice(_DECLINE_TEMPLATES)
    elif decision == "HUMAN_REFERRAL":
        template = random.choice(_REFER_TEMPLATES)
    else:  # AUTO_APPROVE
        template = random.choice(_APPROVE_TEMPLATES)

    email = template.format(name=customer_name, reason=shap_reason)
    return email, prompt


def top_shap_reason(shap_values: list[float], feature_names: list[str], decision: str) -> str:
    """
    Extracts the most impactful SHAP feature as a human-readable reason string.

    For declines/referrals we pick the feature with the highest POSITIVE SHAP
    (i.e. the feature that pushed default risk highest).
    For approvals we pick the feature with the most NEGATIVE SHAP
    (i.e. the feature that most strongly supported approval).
    """
    import numpy as np
    vals = np.array(shap_values)

    if decision == "AUTO_APPROVE":
        idx = int(np.argmin(vals))   # most negative = strongest approval signal
    else:
        idx = int(np.argmax(vals))   # most positive = top risk driver

    name = feature_names[idx]
    val  = vals[idx]
    direction = "increased" if val > 0 else "reduced"
    return f"{name} ({direction} default risk by {abs(val):.3f} log-odds)"


# ─── Smoke Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  GENAI MOCK — Smoke Test")
    print("=" * 60)

    cases = [
        ("Sarah",   "AUTO_DECLINE",   "High Debt-to-Income Ratio (Duration: 48 months)"),
        ("James",   "HUMAN_REFERRAL", "Checking Account balance below 0 DM"),
        ("Priya",   "AUTO_APPROVE",   "7+ years stable employment history"),
    ]

    for name, decision, reason in cases:
        email, prompt = generate_explanation_email(name, decision, reason, seed=42)
        print(f"\n--- {decision} — {name} ---")
        print(email)
        print(f"\n[Prompt that would be sent to GPT-4o]\n{prompt}\n")
