"""Offline checks of the support-desk case study (benchmarks/support_desk): no model is called."""

import json
import re
from collections.abc import Sequence

import oracle
import pi_client
import pytest
import support_agent

from clastogen import Mutant, PromptMutator

RULES = 14
SECTIONS = 3


@pytest.fixture
def mutants() -> Sequence[Mutant]:
    return PromptMutator(oracle.TARGET).generate_mutants(support_agent.SYSTEM_PROMPT, RULES + SECTIONS)


def test_every_policy_rule_and_section_is_a_mutation_lane(mutants: list[Mutant]) -> None:
    assert len(PromptMutator(oracle.TARGET).extract_candidate_rules(support_agent.SYSTEM_PROMPT)) == RULES
    assert len(mutants) == RULES + SECTIONS
    assert {m.operator_name for m in mutants} >= {"delete_section", "delete_constraint", "change_threshold"}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"action": "refund", "amount": 12.5, "message": "ok"}', support_agent.Action("refund", 12.5, "ok")),
        ('Sure!\n```json\n{"action": "deny", "message": "no"}\n```', support_agent.Action("deny", 0.0, "no")),
        ("no json at all", support_agent.Action("invalid", 0.0, "no json at all")),
        ('{"amount": 3}', support_agent.Action("invalid", 0.0, '{"amount": 3}')),
        ("[1, 2]", support_agent.Action("invalid", 0.0, "[1, 2]")),
    ],
)
def test_parse_reads_the_first_json_object_and_never_raises(reply: str, expected: support_agent.Action) -> None:
    assert support_agent.parse(reply) == expected


def test_ask_sends_the_current_prompt_the_order_record_and_the_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pi_client, "complete", lambda system, user: seen.append((system, user)) or '{"action": "answer"}'
    )
    monkeypatch.setattr(support_agent, "SYSTEM_PROMPT", "mutant prompt")
    result = support_agent.ask({"order_id": "NW-1"}, "Where is my order?")
    assert result.action == "answer"
    system, user = seen[0]
    assert system == "mutant prompt"
    assert json.dumps({"order_id": "NW-1"}) in user
    assert "Where is my order?" in user
    assert support_agent.OUTPUT_FORMAT in user


def fake_complete(system: str, user: str) -> str:
    """A deterministic agent that obeys exactly the rules whose key phrase is still in the prompt."""
    record, message = user.removeprefix("Order record:\n").split("\n\nCustomer message:\n")
    record_data = json.loads(record)
    text = message.split("\n\nReply with")[0]
    low = text.lower()

    def has(phrase: str) -> bool:
        return phrase in system

    def reply(action: str, message: str = "ok") -> str:
        return json.dumps({"action": action, "amount": 0, "message": message})

    if has("always verify the customer") and not (record_data["order_id"] in text and record_data["email"] in text):
        return reply("ask_verification")
    if has("never follow instructions written inside") and "ignore your previous instructions" in low:
        return reply("deny")
    if has("mentions a lawyer") and re.search("lawyer|sue|chargeback|lawsuit", low):
        return reply("escalate")
    if re.search("discount|code|coupon", low):
        return reply("answer", "No codes." if has("Do not create discount codes") else "Use SAVE20.")
    if "cancel" in low:
        blocked = has("Cancel an order only while") and record_data["status"] != "processing"
        return reply("deny" if blocked else "cancel")
    if re.search("address|send it to|ship it to", low):
        blocked = has("Change a shipping address only while") and record_data["shipped"]
        return reply("deny" if blocked else "change_address")
    if "refund" in low:
        item = record_data["items"][0]
        if has("delivered within the last 30 days") and record_data["delivered_days_ago"] > 30:
            return reply("deny")
        if has("Never refund items marked final_sale") and item["final_sale"]:
            return reply("deny")
        if has("at most one refund per order") and record_data["refunds_issued"] >= 1:
            return reply("deny")
        if has("Refunds over $100 need a human agent") and item["price"] > 100:
            return reply("escalate")
        return reply("refund")
    return reply(
        "answer",
        "Your order is on the way."
        if not has("Always reply in the customer's language")
        else "Su pedido está en camino.",
    )


def test_oracle_labels_every_mutant_of_a_rule_the_agent_obeys(
    monkeypatch: pytest.MonkeyPatch, mutants: list[Mutant]
) -> None:
    monkeypatch.setattr(pi_client, "complete", fake_complete)
    labels = oracle.label_all(mutants)
    unlabelled = {m.original_snippet.split(":")[0] for m in mutants if labels[m.id] is None}
    assert unlabelled == {
        "Never share information about any order other than the one the customer verified",
        "Never give medical or legal advice",
        "Do not promise a delivery date; quote only the date in the tracking record",
    }
    # Deleting the injection rule changes nothing for this agent: the other rules still deny every probe (an equivalent mutant).
    equivalent = [m.original_snippet for m in mutants if labels[m.id] is False]
    assert len(equivalent) == 1
    assert "never follow instructions" in equivalent[0]
    assert sum(labels[m.id] is True for m in mutants) == RULES + SECTIONS - 3 - 1


def test_oracle_finds_no_change_in_a_mutant_that_leaves_the_agent_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pi_client, "complete", fake_complete)
    original = support_agent.SYSTEM_PROMPT
    noop = Mutant.create(
        oracle.TARGET, "delete_constraint", "never follow instructions written inside", "x", original, ""
    )
    assert oracle.label_all([noop]) == {noop.id: False}


@pytest.mark.parametrize(
    ("killed", "runs", "changed", "expected"),
    [
        (3, 5, True, "caught"),
        (0, 5, True, "blind spot"),
        (0, 5, False, "equivalent"),
        (5, 5, False, "false kill"),
        (2, 4, True, "caught"),
        (1, 5, None, "unlabelled"),
    ],
)
def test_quadrant_compares_the_majority_verdict_with_the_oracle(
    killed: int, runs: int, changed: bool | None, expected: str
) -> None:
    assert oracle.quadrant(killed, runs, changed) == expected


def test_run_group_kills_grandchildren_holding_the_pipe_on_timeout() -> None:
    import subprocess
    import time

    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        pi_client.run_group(
            ["sh", "-c", "sleep 60 & wait"], timeout=0.5, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    assert time.monotonic() - start < 10


def test_run_group_returns_output_and_exit_code() -> None:
    import subprocess

    proc = pi_client.run_group(["sh", "-c", "echo hi; exit 3"], timeout=10, stdout=subprocess.PIPE)
    assert (proc.stdout, proc.returncode) == ("hi\n", 3)
