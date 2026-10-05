# Clastogen 5-Minute Zero-API-Key Quickstart

This example demonstrates how Clastogen detects **evaluation blind spots** without requiring an external API key or token expense.

## Overview

- [`examples/mock_agent.py`](mock_agent.py): A customer support bot governed by `BANKING_SYSTEM_PROMPT`.
- [`examples/test_banking_eval.py`](test_banking_eval.py): A pytest eval suite containing:
  - A **strong test** (`test_identity_verification_is_enforced`) that asserts identity checks.
  - A **superficial test** (`test_refund_handling_superficial_eval`) that only checks response length without verifying authorization limits.
- [`examples/test_stats_usage.py`](test_stats_usage.py): Statistical assertions (`assert_pass_rate`, `assert_no_regression`) on seeded fake evaluators; the same code is shown in the top-level README.

## Run

```bash
pytest --clastogen examples/test_banking_eval.py
```

## What Happens

1. Pytest reports **`2 passed`** (100% green tests).
2. Each test is re-run 9 more times unmutated to measure its baseline pass rate (Laplace p0).
3. Clastogen injects 5 prompt mutants into memory and judges each with an SPRT.
4. The strong test **KILLS** both identity mutants (rule deleted, rule inverted) in 2 trials each.
5. The superficial test lets all three refund mutants (rule deleted, rule inverted, $50 limit raised to $500) **SURVIVE** after 6 trials each.
6. Clastogen outputs the truth: **`Mutation Score: 40.0%`** (2 of 5 caught), exposing the blind spot before production.

The mock agent honors the prompt literally: a rule only holds while its affirmative sentence is present, so deleting, negating or re-numbering it changes the agent's behavior.

Run the statistical assertion examples with `pytest examples/test_stats_usage.py`.
