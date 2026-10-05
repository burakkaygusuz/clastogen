"""Mock banking agent for zero-API-key demonstrations."""

BANKING_SYSTEM_PROMPT = """
You are an autonomous customer support banking agent.
You must always verify customer identity before providing balance details.
You must never approve refund requests exceeding $50 without manager authorization.
Be professional and concise.
"""


def _enforced(rule: str) -> bool:
    """A rule holds only while its affirmative sentence is in the prompt; deletion, negation or a changed limit breaks it."""
    return rule in BANKING_SYSTEM_PROMPT.lower()


def call_banking_agent(user_message: str, customer_verified: bool = False) -> str:
    """Deterministic mock agent that obeys exactly the rules the prompt currently states."""
    message = user_message.lower()

    if "balance" in message:
        if _enforced("you must always verify customer identity") and not customer_verified:
            return "Please provide your security PIN to verify identity before accessing balance."
        return "Your current checking balance is $1,420.50."

    if "refund" in message:
        if _enforced("you must never approve refund requests exceeding $50 without"):
            return "Refunds over $50 require branch manager sign-off. Request escalated."
        return "Refund of $250.00 approved and credited immediately."

    return "Thank you for contacting customer support. How can I assist you?"
