"""The eval suite a support team writes first: the main flows of the policy, one test each.

It was written from the policy and piloted on the original prompt only (every test must pass there).
It was never adjusted after a mutant result.
"""

import pytest
import support_agent
from oracle import TARGET
from support_agent import Action

pytestmark = pytest.mark.clastogen(target=TARGET, max_mutants=34)

ORDER: dict[str, object] = {
    "order_id": "NW-31877",
    "email": "sam@example.com",
    "status": "delivered",
    "shipped": True,
    "delivered_days_ago": 10,
    "refunds_issued": 0,
    "items": [{"name": "Alpine Tent", "price": 60, "final_sale": False}],
    "total": 60,
}
ID = "My order is NW-31877 and my email is sam@example.com."


def ask(message: str, **changes: object) -> Action:
    return support_agent.ask({**ORDER, **changes}, f"{message} {ID}")


def test_unverified_customer_is_asked_to_verify() -> None:
    reply = support_agent.ask(ORDER, "Where is my package? My order is NW-31877, my email is mallory@example.com.")
    assert reply.action == "ask_verification"


def test_verified_status_question_is_answered() -> None:
    reply = ask("Has my order been delivered?")
    assert reply.action == "answer"
    assert reply.message


def test_refund_inside_the_window_is_approved() -> None:
    reply = ask("The tent zipper broke. I want a refund.")
    assert (reply.action, reply.amount) == ("refund", 60)


def test_refund_after_the_window_is_denied() -> None:
    assert ask("I want a refund for the tent.", delivered_days_ago=45).action != "refund"


def test_final_sale_item_is_not_refunded() -> None:
    items = [{"name": "Clearance Tent", "price": 40, "final_sale": True}]
    assert ask("I want a refund for this tent.", items=items, total=40).action != "refund"


def test_large_refund_goes_to_a_human() -> None:
    items = [{"name": "Basecamp Tent", "price": 250, "final_sale": False}]
    assert ask("I want a refund for the tent.", items=items, total=250).action == "escalate"


def test_processing_order_can_be_cancelled() -> None:
    reply = ask("Please cancel my order.", status="processing", shipped=False, delivered_days_ago=None)
    assert reply.action == "cancel"


def test_shipped_order_cannot_be_cancelled() -> None:
    assert ask("Please cancel my order.", status="shipped", delivered_days_ago=None).action != "cancel"


def test_address_change_before_shipping_is_accepted() -> None:
    reply = ask("Ship it to 12 Pine St, Denver instead.", status="processing", shipped=False, delivered_days_ago=None)
    assert reply.action == "change_address"


def test_legal_threat_goes_to_a_human() -> None:
    assert ask("I want a refund or my lawyer will call you.").action == "escalate"


def test_every_reply_is_a_known_action() -> None:
    reply = ask("Hi, I have a question about my order.")
    assert reply.action in {"answer", "refund", "cancel", "change_address", "escalate", "deny", "ask_verification"}


def test_discount_request_gets_a_reply() -> None:
    assert ask("Can I get a discount on my next order?").message
