# Clastogen

![Clastogen logo](https://raw.githubusercontent.com/burakkaygusuz/clastogen/main/assets/logo.svg)

**Mutation testing and statistical assertion framework for LLMs and AI Agents.**

[![PyPI](https://img.shields.io/pypi/v/clastogen)](https://pypi.org/project/clastogen/)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/burakkaygusuz/clastogen/badge)](https://scorecard.dev/viewer/?uri=github.com/burakkaygusuz/clastogen)

Clastogen injects controlled faults (deleting constraints, inverting rules, changing numeric limits) into your system prompts to verify whether your test suite actually catches prompt breakages, using Sequential Probability Ratio Tests (SPRT) to stop early and save API budget.

---

## Installation

```bash
pip install clastogen
# or
uv add --dev clastogen
```

For local development, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Usage

### 1. Pytest Plugin (Recommended)

Mark your LLM evaluation test with `@pytest.mark.clastogen` and point it to the target prompt variable:

```python
import pytest


@pytest.mark.clastogen(target="my_app.agent:SYSTEM_PROMPT")
def test_agent_behavior():
    response = call_agent("transfer $500")
    assert "Verification code" in response
```

Run tests with mutation testing enabled:

```bash
# Run all tests with mutation testing
pytest --clastogen

# Target a specific test file
pytest --clastogen tests/test_agent.py

# Verbose output with step-by-step logs
pytest --clastogen -s --log-cli-level=INFO
```

#### Marker Options

```python
@pytest.mark.clastogen(
    target="my_app.agent:SYSTEM_PROMPT",  # Required: 'module:VAR' or 'module:Class.ATTR'
    max_mutants=5,  # Max mutants generated (default: 5)
    delta=0.30,  # Minimum detectable pass-rate drop (default: 0.30)
    p0=0.90,  # Known baseline pass rate (default: measured over 10 baseline runs; setting it skips them)
    max_steps=20,  # Maximum evaluation runs per mutant (default: 20)
)
```

#### Prompt must be read at call time

Clastogen mutates the target by swapping the module (or class) attribute for the duration of each trial, then restoring it. It also swaps module-level aliases that hold the very same string object under the same name (e.g. `from my_app.agent import SYSTEM_PROMPT` at the top of another module). Your code therefore has to read the prompt when it calls the model. Copies made earlier are never updated, so mutants against them always appear `SURVIVED`.

Function-scoped fixtures are torn down and rebuilt inside the mutated prompt on every trial, so a fixture such as `bot = {"system": agent.PROMPT}` sees the mutant and is fine. Module- and session-scoped fixtures are built once and keep the original prompt, as do import-time captures; mutants against them also appear `SURVIVED`.

Wrong: the prompt is captured once at import time, so the mutated attribute is never read.

```python
# my_app/agent.py
SYSTEM_PROMPT = "You must never approve refunds over $50."
CONFIG = {"system": SYSTEM_PROMPT}  # frozen copy; also true for f-strings, default args, clients built at import


def ask(question: str) -> str:
    return client.chat(system=CONFIG["system"], user=question)
```

Right: the module attribute is looked up on every call (a `from my_app.agent import SYSTEM_PROMPT` inside the function body works too).

```python
# my_app/agent.py
SYSTEM_PROMPT = "You must never approve refunds over $50."


def ask(question: str) -> str:
    return client.chat(system=SYSTEM_PROMPT, user=question)
```

#### Parallel runs (pytest-xdist)

`pytest --clastogen -n 4` works: each worker returns its records with the test report and the controller merges them into one summary. Killed-mutant short-circuiting and baseline measurement are per-process, so every worker measures its own baselines and may re-evaluate a mutant already killed on another worker. Parallel runs trade some redundant work for wall-clock time.

---

### 2. Statistical Assertions (Python API)

Assert non-deterministic LLM behavior with statistical guarantees instead of a single flaky `assert`. `assert_pass_rate` samples a zero-argument callable returning `bool` with Wald's SPRT, stopping as soon as the evidence is decisive (at most `max_samples` calls), and raises `AssertionError` when the pass rate is judged below `min_rate` (see [Pass-rate guarantees](#pass-rate-guarantees)). `assert_no_regression` runs a baseline and a candidate callable (paired McNemar test by default, Fisher exact with `paired=False`) and raises only on a statistically significant drop. Both return a result object, and `evaluate_pass_rate` / `evaluate_regression` return it without asserting. `compute_wilson_interval` is exported too.

The example below is [`examples/test_stats_usage.py`](examples/test_stats_usage.py); it uses seeded fake evaluators, so it runs with `pytest examples/test_stats_usage.py` and no API key:

<!-- example: examples/test_stats_usage.py -->
```python
"""Statistical assertions on seeded fake evaluators: zero API keys, deterministic."""

import random
from collections.abc import Callable

import pytest

from clastogen import assert_no_regression, assert_pass_rate


def fake_llm(pass_rate: float, seed: int) -> Callable[[], bool]:
    rng = random.Random(seed)
    return lambda: rng.random() < pass_rate


def test_order_intent_pass_rate() -> None:
    result = assert_pass_rate(fake_llm(0.97, seed=1), min_rate=0.90, description="order intent")
    assert result.observed_rate >= 0.90


def test_prompt_upgrade_does_not_regress() -> None:
    assert_no_regression(fake_llm(0.90, seed=2), fake_llm(0.90, seed=3), description="prompt v1 -> v2")


def test_broken_prompt_is_detected_as_regression() -> None:
    with pytest.raises(AssertionError, match="regression"):
        assert_no_regression(fake_llm(0.95, seed=4), fake_llm(0.40, seed=5), description="prompt v1 -> v3")
```

---

## Demo

Zero API keys required. [`examples/mock_agent.py`](examples/mock_agent.py) is a deterministic mock banking agent that obeys exactly the rules its system prompt currently states, and [`examples/test_banking_eval.py`](examples/test_banking_eval.py) holds two evals: a strong identity-verification test and a deliberately weak refund test that only checks the response is non-empty.

```bash
uv run pytest --clastogen examples/test_banking_eval.py
```

```text
======================= Clastogen Mutation Testing Summary =======================
Total Unique Mutants    : 5
Killed (Caught by Suite): 2
Survived (Blind Spots)  : 3
Inconclusive (Truncated): 0
Execution Errors (Excl.): 0
Skipped (Excl.)         : 0
Suppressed (Excl.)      : 0
Mutation Score          : 40.0%
--------------------------------------------------------------------------------
  [2caac4fe522b] ✗ SURVIVED       (6 runs, LLR=-2.38) -> Deleted load-bearing constraint: 'You must never approve refund requests exceeding $...'
  [45e77dddc073] ✓ KILLED         (2 runs, LLR=+3.05) (killed by test_identity_verification_is_enforced) -> Inverted constraint: 'You must always verify customer identity' -> 'You must NEVER verify customer identity '
  [55063f25e7dd] ✓ KILLED         (2 runs, LLR=+3.05) (killed by test_identity_verification_is_enforced) -> Deleted load-bearing constraint: 'You must always verify customer identity before pr...'
  [7d1281020f96] ✗ SURVIVED       (6 runs, LLR=-2.38) -> Inverted constraint: 'You must never approve refund requests e' -> 'You must ALWAYS approve refund requests '
  [add892bdcea2] ✗ SURVIVED       (6 runs, LLR=-2.38) -> Changed threshold: '50' -> '500' in 'You must never approve refund requests e'
--------------------- Baselines (Laplace p0 used for SPRT) ---------------------
  examples/test_banking_eval.py::test_identity_verification_is_enforced: 10/10 baseline runs passed, p0=0.917 -> examples.mock_agent:BANKING_SYSTEM_PROMPT
  examples/test_banking_eval.py::test_refund_handling_superficial_eval: 10/10 baseline runs passed, p0=0.917 -> examples.mock_agent:BANKING_SYSTEM_PROMPT
================================================================================
```

Both tests pass in a normal run, yet only 2 of the 5 mutants are caught (**Mutation Score 40.0%**). The strong identity test kills both the deleted and the inverted identity rule (2 runs each), while the weak refund test keeps passing when the refund rule is deleted, inverted to "ALWAYS approve" or its limit raised from $50 to $500, so those three mutants SURVIVE: an eval blind spot you would otherwise ship.

### Reading the report

- **Statuses:** `KILLED` (a test failed under the mutant), `SURVIVED`, `INCONCLUSIVE` (SPRT hit `max_steps` without a verdict), `ERROR` (the evaluation itself broke; the reason is in the JSON `error` field), `SKIPPED`, `SUPPRESSED`.
- **Mutation Score** = `KILLED / (KILLED + SURVIVED + INCONCLUSIVE)`. `ERROR`, `SKIPPED` and `SUPPRESSED` mutants are not measured and stay out of the score. When several tests evaluate the same mutant, the strongest outcome wins (`KILLED` > `SURVIVED` > `INCONCLUSIVE` > `ERROR` > `SKIPPED` > `SUPPRESSED`).
- **Baselines:** each marked test is first re-run 9 more times unmutated (10 runs in total). p0 is the Laplace estimate `(s + 1) / (n + 2)`, and a test whose raw pass rate is below 80% is rejected as flaky and not mutated. The terminal "Baselines" section lists them.
- **Flags:** `--clastogen-fail-under MIN_SCORE` fails the run below that score; `--clastogen-json PATH` writes `mutation_score`, `total_mutants`, `counts` (all six statuses), `results` (one merged record per mutant), `executions` (every test's record per mutant, the data behind the HTML kill matrix) and `baselines`; `sample_count` and `llr` are `null` for mutants the SPRT never ran (`ERROR`, `SKIPPED`, `SUPPRESSED`). `--clastogen-html PATH` renders the same data as a self-contained HTML report.

## Suppressing Mutants

When a mutant survives because the foundation model inherently obeys the rule from pre-training (an equivalent mutant), suppress it so it does not skew your score. Suppressed mutants are reported as `SUPPRESSED` and excluded from the Mutation Score. Mutant IDs are the 12-character hashes printed in the summary, for example `7d1281020f96` (the inverted refund rule in the [Demo](#demo)).

List them in `.clastogen/suppressions.toml` under your pytest `rootdir`. Every entry needs a `reason`:

```toml
[[suppressions]]
mutant_id = "7d1281020f96"
reason = "Model refuses large refunds regardless of the prompt"
```

A malformed file aborts a `--clastogen` run with a usage error; runs without `--clastogen` never read it.

---

## Empirical SPRT Performance & Limits

SPRT dynamically sizes samples using Wald's sequential boundaries. Severe defects halt in as few as 2 calls with a measured 10/10 baseline (3 with an explicit $p_0 = 0.90$), while ambiguous boundaries cap at $N_{\max}$.

Results from Monte Carlo simulation (`PYTHONPATH=src uv run python scripts/simulate_sprt.py`, 10,000 trials per row, seed 42):

### Scenario 1: Standard Evaluation ($p_0 = 0.90, p_1 = 0.60, N_{\max} = 20$)

| True P | KILLED % | SURVIVED % | INCONCLUSIVE % | ASN (Mean) | Call Range | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.95** | 0.28% | 98.84% | 0.88% | **7.54** | [4, 20] | Clean pass |
| **0.90 ($H_0$)** | **3.10%** | 91.66% | 5.24% | **9.38** | [3, 20] | False Kill $\le \alpha = 5\%$ |
| **0.85** | 9.81% | 76.70% | 13.49% | **10.94** | [3, 20] | Slight degradation |
| **0.75** | 39.92% | 39.42% | **20.66%** | **11.90** | [3, 20] | Indifference zone |
| **0.60 ($H_1$)** | 84.74% | **8.50%** | 6.76% | **9.03** | [3, 20] | False Survive $\le \beta = 10\%$ |
| **0.40** | 99.37% | 0.51% | 0.12% | **5.36** | [3, 20] | Severe defect |
| **0.10** | **100.00%** | 0.00% | 0.00% | **3.33** | [3, 9] | Severe defect |
| **0.00** | **100.00%** | 0.00% | 0.00% | **3.00** | [3, 3] | **Halted in 3 calls** |

### Scenario 2: Flaky Baseline ($p_0 = 0.75, p_1 = 0.50, N_{\max} = 25$)

| True P | KILLED % | SURVIVED % | INCONCLUSIVE % | ASN (Mean) | Call Range | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.85** | 0.16% | 98.28% | 1.56% | **9.95** | [6, 25] | High pass |
| **0.75 ($H_0$)** | **2.92%** | 79.95% | 17.13% | **14.63** | [5, 25] | Flaky baseline false kill $\le 5\%$ |
| **0.65** | 19.16% | 43.53% | **37.31%** | **18.08** | [5, 25] | **High ambiguity / wandering** |
| **0.50 ($H_1$)** | 71.34% | **7.23%** | 21.43% | **15.91** | [5, 25] | False Survive $\le \beta = 10\%$ |
| **0.35** | 97.88% | 0.42% | 1.70% | **10.19** | [5, 25] | Severe defect |
| **0.10** | **100.00%** | 0.00% | 0.00% | **5.69** | [5, 16] | Severe defect |
| **0.00** | **100.00%** | 0.00% | 0.00% | **5.00** | [5, 5] | Halted at min bound |

### Scenario 3: Full Plugin Flow (10-run Laplace baseline, $\delta = 0.30$, $N_{\max} = 20$)

The plugin measures the unmutated test 10 times, estimates $p_0 = (s + 1) / (n + 2)$ (Laplace rule of succession), rejects baselines whose raw pass rate is below 80%, then runs the SPRT on each mutant with $p_1 = p_0 - 0.30$. Rates below are over accepted baselines.

| Baseline P | Mutant P | Baseline rejected % | KILLED % | SURVIVED % | INCONCLUSIVE % | ASN (Mean) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.95** | 0.95 | 0.92% | **0.38%** | 98.13% | 1.48% | **7.40** | Unchanged mutant |
| **0.90** | 0.90 | 5.28% | **2.19%** | 92.61% | 5.20% | **8.57** | Unchanged mutant |
| **0.85** | 0.85 | 14.00% | **5.02%** | 86.05% | 8.93% | **9.58** | Unchanged mutant |
| **0.95** | 0.60 | 1.00% | 76.61% | 11.05% | 12.34% | **9.64** | Moderate defect |
| **0.95** | 0.30 | 0.83% | **99.73%** | 0.13% | 0.14% | **4.52** | Severe defect |
| **0.95** | 0.00 | 0.83% | **100.00%** | 0.00% | 0.00% | **2.43** | Total failure |

> **Indifference Zone Tradeoff:** In ambiguous zones where the mutant pass rate is close to $(p_0 + p_1) / 2$, sequential tests cannot make a definitive call without infinite samples. Clastogen caps trials at $N_{\max}$ and honestly classifies borderline outcomes as `INCONCLUSIVE` instead of guessing.

### Pass-rate guarantees

`assert_pass_rate` tests $H_0$: rate $=$ `min_rate` against $H_1$: rate $=$ `min_rate - tolerance`, with both error rates set to `1 - confidence`. A true rate at or above `min_rate` passes with probability of about `confidence`, a true rate at or below `min_rate - tolerance` fails with about the same probability, and rates in between (the indifference zone) can go either way. If `max_samples` runs out before a decision, the sign of the log-likelihood ratio decides; `result.decided` tells you whether the SPRT stopped on its own. Narrow `tolerance` for a sharper verdict at the cost of more samples.

Scenario 4 of the simulation script, defaults (`min_rate=0.90, tolerance=0.20, confidence=0.95, max_samples=50`):

| True P | PASSED % | ASN (Mean) | Note |
| :--- | :--- | :--- | :--- |
| **0.95** | 99.74% | **16.59** | Clearly good |
| **0.90 (`min_rate`)** | **95.53%** | **23.50** | Pass $\ge$ confidence |
| **0.85** | 76.89% | **29.42** | Indifference zone |
| **0.80** | 44.22% | **29.46** | Indifference zone |
| **0.70 (`min_rate - tolerance`)** | **5.94%** | **19.30** | Fail $\approx$ confidence |
| **0.60** | 0.44% | **11.63** | Clearly bad |

---

## Known Limitations (v0.1)

1. **Mutation Operator Scope:** Rules are found lexically: a sentence is a candidate only if it contains a keyword such as `must`, `never`, `always`, `avoid`, `only`, `may not` or `require`. Rules phrased without one (for example conditionals like "Escalate to a human if …") are not mutated. Three operators run on each rule: `delete_constraint`, `invert_negation` and `change_threshold` (first number ×10). Semantic LLM-guided mutations, RAG context poisoning, and tool schema mutators are planned for v0.2/v0.3.
2. **Equivalent Mutants:** A prompt mutation can occasionally result in identical agent behavior (e.g. if the underlying foundation model inherently obeys a safety constraint from pre-training). Clastogen handles this pragmatically via triage suppression (`.clastogen/suppressions.toml`) rather than automated semantic equivalence proofs.
3. **Effect Size:** The SPRT looks for an absolute drop of `delta` (default 0.30) from a 10-run Laplace baseline, which caps p0 at 11/12 ≈ 0.917. Small regressions such as 0.95 → 0.88 need roughly 60-70 runs per mutant to decide, so with `max_steps=20` they end `INCONCLUSIVE` or `SURVIVED`. Lower `delta` and raise `max_steps` per test (`@pytest.mark.clastogen(delta=0.10, max_steps=100)`) when such drops matter, at the matching API cost.
4. **Baseline Noise:** The first baseline run is the already-passing test and p0 is a point estimate, so for baselines near the 80% acceptance floor the per-mutant false-kill rate exceeds alpha. For an unchanged mutant (`delta=0.30`, `max_steps=20`, among accepted baselines) a true baseline of 0.90 gives 2.2%, 0.85 gives 5.0%, 0.80 gives 8.6% and 0.75 gives 13.2%. Raising the 80% floor does not fix it (with a 90% floor, 0.85 still gives 7.0%): stabilize the test or set p0 explicitly.
5. **Differential Execution:** Clastogen deduplicates already-killed mutants across tests, but does not yet construct a pre-execution static dependency graph.

---

## Commands Summary

| Command | Purpose |
| --- | --- |
| `pytest` | Run normal test suite (clastogen dormant) |
| `pytest --clastogen` | Run test suite with mutation testing & summary score |
| `PYTHONPATH=src uv run python scripts/simulate_sprt.py` | Run Monte Carlo SPRT power simulation |
| `uv run ruff check .` | Run static code analysis & linter |
| `uv run mypy src tests scripts` | Run strict static type checking |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
