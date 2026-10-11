from __future__ import annotations

import re

from clastogen.mutation.mutator import PromptMutator
from clastogen.mutation.structure import find_blocks

PROMPT = """You are a support agent.

# Policy
Never approve refunds over $50.

## Tone
Always be polite to the customer.

# Examples
User: refund $500
Agent: I cannot approve that.

```
# Not a heading
```

<example name="a">
User: hi
Agent: hello
</example>
"""


def _spans(prompt: str) -> list[tuple[str, str]]:
    return [(b.kind, prompt[b.start : b.end].splitlines()[0]) for b in find_blocks(prompt)]


def test_blocks_follow_heading_levels_fences_and_example_tags() -> None:
    assert _spans(PROMPT) == [
        ("section", "# Policy"),
        ("section", "## Tone"),
        ("example", "# Examples"),
        ("example", '<example name="a">'),
    ]
    policy = find_blocks(PROMPT)[0]
    assert PROMPT[policy.start : policy.end].endswith("polite to the customer.\n\n")  # subheading stays inside


def test_examples_wrapper_is_kept_and_its_children_are_blocks() -> None:
    wrapped = "<examples><example>a</example><example>b</example></examples>"
    assert [(b.kind, wrapped[b.start : b.end]) for b in find_blocks(wrapped)] == [
        ("example", "<example>a</example>"),
        ("example", "<example>b</example>"),
    ]


def test_delete_section_and_example_leave_balanced_tags_and_collapse_blank_lines() -> None:
    mutants = PromptMutator().generate_mutants(PROMPT, max_mutants=20)
    assert {"delete_section", "delete_example"} <= {m.operator_name for m in mutants}
    for m in mutants:
        if m.operator_name in ("delete_section", "delete_example"):
            assert m.original_snippet not in m.mutated_prompt
            assert m.mutated_prompt.count("<example") == m.mutated_prompt.count("</example>")
            assert not re.search(r"\n{3,}", m.mutated_prompt)


def test_section_without_a_rule_and_block_over_half_the_prompt_are_not_mutated() -> None:
    ruleless = "# Greeting\nSay hello warmly to everyone.\n# Policy\nNever share account data with anyone else."
    ops = {m.operator_name for m in PromptMutator().generate_mutants(ruleless, max_mutants=9)}
    assert "delete_section" not in ops  # Policy is over half of the prompt, Greeting holds no rule


def test_block_lane_is_reachable_at_a_small_budget() -> None:
    rules = "\n".join(f"Never share item {i} with anyone." for i in range(8))
    prompt = f"# Rules\n{rules}\n<example>\nUser: hi\n</example>\nAlways greet the user."
    ops = {m.operator_name for m in PromptMutator().generate_mutants(prompt, max_mutants=5)}
    assert "delete_example" in ops


def test_deep_nesting_fences_and_identical_blocks_are_handled() -> None:
    deep = "".join(f"<t{i}>" for i in range(1500)) + "x" + "".join(f"</t{i}>" for i in reversed(range(1500)))
    assert len(find_blocks(deep)) == 1500

    nested_fence = "```\n~~~\n# Fake\n```\n\n# Real\nYou must do x.\n"
    assert [b.label for b in find_blocks(nested_fence)] == ["Real"]

    twin = "<example>User asks about pricing today.</example>\nMid text.\n<example>User asks about pricing today.</example>\nMust end."
    ids = [m.id for m in PromptMutator().generate_mutants(twin, max_mutants=9)]
    assert len(ids) == len(set(ids))


def test_pathological_whitespace_stays_fast_and_long_numbers_stay_exact() -> None:
    import time

    start = time.perf_counter()
    PromptMutator().generate_mutants("# x" + " " * 40_000 + "y\nYou must x" + " " * 40_000 + "y.", max_mutants=5)
    assert time.perf_counter() - start < 1.0

    (m,) = PromptMutator().generate_mutants("You must send at most 123456789012345678901234567890123 items.", 1)[:1]
    assert "1234567890123456789012345678901230" in m.mutated_prompt
