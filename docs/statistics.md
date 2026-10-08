# Statistics

LLM output is random, so one run of a test proves little. Clastogen uses Wald's Sequential Probability Ratio Test (SPRT): it runs a test again only until the evidence is sufficient. The important result: severe prompt regressions are found with few calls (2 for a clear kill), and ambiguous cases are reported as `INCONCLUSIVE` instead of guessed.

## Statistical assertions (Python API)

One `assert` on LLM output is flaky. Use these functions to get statistical guarantees:

- `assert_pass_rate` calls a callable that has no arguments and returns a `bool`. It uses Wald's SPRT and stops when the evidence is sufficient. It calls the callable `max_samples` times at most. If the pass rate is less than `min_rate`, it raises `AssertionError`. See [Pass-rate guarantees](#pass-rate-guarantees).
- `assert_no_regression` calls a baseline callable and a candidate callable. By default, it uses a paired McNemar test. With `paired=False`, it uses a Fisher exact test. It raises `AssertionError` only if the decrease is statistically significant.
- `evaluate_pass_rate` and `evaluate_regression` return the same result objects, but they do not assert.
- `compute_wilson_interval` is also available.

The example below is [`examples/test_stats_usage.py`](../examples/test_stats_usage.py). It uses fake evaluators with a seed. Thus, you can run it with `pytest examples/test_stats_usage.py` and without an API key:

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

## SPRT performance and limits

The SPRT uses Wald's sequential boundaries to set the number of samples. For a severe defect with a measured 10/10 baseline, the SPRT stops after only 2 calls. With an explicit $p_0 = 0.90$, it stops after 3 calls. For results near the boundary, the SPRT stops at $N_{\max}$.

These results come from a Monte Carlo simulation (`PYTHONPATH=src uv run python scripts/simulate_sprt.py`, 10,000 trials for each row, seed 42):

### Scenario 1: Standard Evaluation ($p_0 = 0.90, p_1 = 0.60, N_{\max} = 20$)

| True P           | KILLED %    | SURVIVED % | INCONCLUSIVE % | ASN (Mean) | Call Range | Note                             |
| :--------------- | :---------- | :--------- | :------------- | :--------- | :--------- | :------------------------------- |
| **0.95**         | 0.28%       | 98.84%     | 0.88%          | **7.54**   | [4, 20]    | Clean pass                       |
| **0.90 ($H_0$)** | **3.10%**   | 91.66%     | 5.24%          | **9.38**   | [3, 20]    | False Kill $\le \alpha = 5\%$    |
| **0.85**         | 9.81%       | 76.70%     | 13.49%         | **10.94**  | [3, 20]    | Slight degradation               |
| **0.75**         | 39.92%      | 39.42%     | **20.66%**     | **11.90**  | [3, 20]    | Indifference zone                |
| **0.60 ($H_1$)** | 84.74%      | **8.50%**  | 6.76%          | **9.03**   | [3, 20]    | False Survive $\le \beta = 10\%$ |
| **0.40**         | 99.37%      | 0.51%      | 0.12%          | **5.36**   | [3, 20]    | Severe defect                    |
| **0.10**         | **100.00%** | 0.00%      | 0.00%          | **3.33**   | [3, 9]     | Severe defect                    |
| **0.00**         | **100.00%** | 0.00%      | 0.00%          | **3.00**   | [3, 3]     | **Halted in 3 calls**            |

### Scenario 2: Flaky Baseline ($p_0 = 0.75, p_1 = 0.50, N_{\max} = 25$)

| True P           | KILLED %    | SURVIVED % | INCONCLUSIVE % | ASN (Mean) | Call Range | Note                                |
| :--------------- | :---------- | :--------- | :------------- | :--------- | :--------- | :---------------------------------- |
| **0.85**         | 0.16%       | 98.28%     | 1.56%          | **9.95**   | [6, 25]    | High pass                           |
| **0.75 ($H_0$)** | **2.92%**   | 79.95%     | 17.13%         | **14.63**  | [5, 25]    | Flaky baseline false kill $\le 5\%$ |
| **0.65**         | 19.16%      | 43.53%     | **37.31%**     | **18.08**  | [5, 25]    | **High ambiguity / wandering**      |
| **0.50 ($H_1$)** | 71.34%      | **7.23%**  | 21.43%         | **15.91**  | [5, 25]    | False Survive $\le \beta = 10\%$    |
| **0.35**         | 97.88%      | 0.42%      | 1.70%          | **10.19**  | [5, 25]    | Severe defect                       |
| **0.10**         | **100.00%** | 0.00%      | 0.00%          | **5.69**   | [5, 16]    | Severe defect                       |
| **0.00**         | **100.00%** | 0.00%      | 0.00%          | **5.00**   | [5, 5]     | Halted at min bound                 |

### Scenario 3: Full Plugin Flow (10-run Laplace baseline, $\delta = 0.30$, $N_{\max} = 20$)

The plugin runs the test without a mutant 10 times. It calculates $p_0 = (s + 1) / (n + 2)$ (the Laplace rule of succession). If the raw pass rate is less than 80%, the plugin rejects the baseline. Then it runs the SPRT on each mutant with $p_1 = p_0 - 0.30$. The rates below include only the accepted baselines.

| Baseline P | Mutant P | Baseline rejected % | KILLED %    | SURVIVED % | INCONCLUSIVE % | ASN (Mean) | Note             |
| :--------- | :------- | :------------------ | :---------- | :--------- | :------------- | :--------- | :--------------- |
| **0.95**   | 0.95     | 0.92%               | **0.38%**   | 98.13%     | 1.48%          | **7.40**   | Unchanged mutant |
| **0.90**   | 0.90     | 5.28%               | **2.19%**   | 92.61%     | 5.20%          | **8.57**   | Unchanged mutant |
| **0.85**   | 0.85     | 14.00%              | **5.02%**   | 86.05%     | 8.93%          | **9.58**   | Unchanged mutant |
| **0.95**   | 0.60     | 1.00%               | 76.61%      | 11.05%     | 12.34%         | **9.64**   | Moderate defect  |
| **0.95**   | 0.30     | 0.83%               | **99.73%**  | 0.13%      | 0.14%          | **4.52**   | Severe defect    |
| **0.95**   | 0.00     | 0.83%               | **100.00%** | 0.00%      | 0.00%          | **2.43**   | Total failure    |

> **Indifference zone:** Sometimes the mutant pass rate is near $(p_0 + p_1) / 2$. In this zone, a sequential test cannot make a decision with a finite number of samples. Clastogen stops at $N_{\max}$ and shows the result as `INCONCLUSIVE`. It does not guess.

### Savings against a fixed sample size

A fixed-N test with the same error rates ($\alpha = 0.05$, $\beta = 0.10$) is the smallest binomial test that fails at most $\alpha$ of the time at $p_0$ and passes at most $\beta$ of the time at $p_1$. The script computes it and compares it with the SPRT's mean calls:

| Scenario                     | Fixed N (KILLED if passes $\le c$) | SPRT ASN at $p_0$ / $p_1$ | Saved         | `INCONCLUSIVE` at $p_0$ / $p_1$ |
| :--------------------------- | :--------------------------------- | :------------------------ | :------------ | :------------------------------ |
| 1 ($p_0 = 0.90, p_1 = 0.60$) | 18 ($c = 13$)                      | 9.38 / 9.03               | 47.9% / 49.9% | 5.2% / 6.8%                     |
| 2 ($p_0 = 0.75, p_1 = 0.50$) | 33 ($c = 20$)                      | 14.63 / 15.91             | 55.7% / 51.8% | 17.1% / 21.4%                   |

The saving is not free: a fixed-N test always decides, while the truncated SPRT stops some runs at $N_{\max}$ as `INCONCLUSIVE`. These runs are counted in the ASN at $N_{\max}$ and in the score's denominator. Scenario 2 is worse because $N_{\max} = 25$ is below the fixed N of 33. Raise `max_steps` if inconclusive mutants matter more than API calls.

### Pass-rate guarantees

`assert_pass_rate` tests $H_0$: rate $=$ `min_rate` against $H_1$: rate $=$ `min_rate - tolerance`. Both error rates are `1 - confidence`.

- If the true rate is `min_rate` or more, the assertion passes with a probability of approximately `confidence`.
- If the true rate is `min_rate - tolerance` or less, the assertion fails with approximately the same probability.
- If the true rate is between these two values (the indifference zone), the assertion can pass or fail.

If the SPRT uses all `max_samples` before a decision, the sign of the log-likelihood ratio gives the decision. This is different from the pytest plugin: there, a mutant without a decision gets the status `INCONCLUSIVE`. An assertion must pass or fail, so `assert_pass_rate` always decides. `result.decided` tells you if the SPRT stopped by itself. A smaller `tolerance` gives a more precise result, but it needs more samples.

Scenario 4 of the simulation script, with the default values (`min_rate=0.90, tolerance=0.20, confidence=0.95, max_samples=50`):

| True P                            | PASSED %   | ASN (Mean) | Note                      |
| :-------------------------------- | :--------- | :--------- | :------------------------ |
| **0.95**                          | 99.74%     | **16.59**  | Clearly good              |
| **0.90 (`min_rate`)**             | **95.53%** | **23.50**  | Pass $\ge$ confidence     |
| **0.85**                          | 76.89%     | **29.42**  | Indifference zone         |
| **0.80**                          | 44.22%     | **29.46**  | Indifference zone         |
| **0.70 (`min_rate - tolerance`)** | **5.94%**  | **19.30**  | Fail $\approx$ confidence |
| **0.60**                          | 0.44%      | **11.63**  | Clearly bad               |
