# Benchmark: a real model, two eval suites

One system prompt, one real model, two eval suites that use the same three inputs. Clastogen mutates the prompt and
runs each suite 20 times. The question: does Clastogen tell a weak eval suite from a strong one, and how stable is its verdict
when the model's output is random?

This is a single measurement: one prompt, one model, one provider. Read the [limitations](#limitations) before you quote a number.

## Setup

- **Model:** `openrouter/deepseek/deepseek-v4.1-flash`, called through pi 1.1.0 with `-p --no-session --no-tools --no-extensions --no-context-files --no-skills --no-prompt-templates --no-mcp --offline --thinking off`. Only the system prompt and the user message reach the model ([`bank_agent.py`](bank_agent.py)).
- **Prompt:** a banking agent with three rules (verify identity before a balance, no refund over $50 without a manager, only banking topics). Clastogen makes 10 mutants from it.
- **Weak suite** ([`test_weak.py`](test_weak.py)): checks that the agent replied. Assertions such as "the reply is not empty" or "the reply mentions the order".
- **Strong suite** ([`test_strong.py`](test_strong.py)): same inputs, but each test asserts the decision that one rule demands.
- **Settings:** `p0=0.9` fixed in the marker (no baseline measurement), `alpha=0.05`, `beta=0.10`, serial runs without xdist, 20 runs per suite.
- **Fixed-N:** the `Calls` comparison is a fixed-N test with the same error rates (18 calls per test and mutant). It is hypothetical: it was not run.

## Results

Run on 2026-10-10 with clastogen 0.5.3 on commit `68da7ba` plus the uncommitted call-savings change (dirty tree). The summary is in [`results/deepseek_deepseek-v4.1-flash/`](results/deepseek_deepseek-v4.1-flash/summary.md). The raw JSON and logs of the 40 runs (1.2 MB) are git-ignored; `run.py` writes them next to the summary.

| Suite | Runs | Mutation Score % (min-max) | Calls (min-max) | Fixed-N calls | Saving % (min-max) |
| --- | --- | --- | --- | --- | --- |
| weak | 20 | 0.0 (0.0-0.0) | 181 (180-184) | 540 | 66.5 (65.9-66.7) |
| strong | 20 | 54.5 (40.0-70.0) | 159.3 (130-196) | 434.7 | 63.5 (58.1-68.6) |

Share of runs that killed each mutant, with the Wilson 95% interval:

| Mutant | weak | strong |
| --- | --- | --- |
| Deleted constraint: identity verification | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] |
| Deleted constraint: refund limit | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] |
| Inverted constraint: banking topics -> 'MUST NOT only discuss' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] |
| Changed threshold: $50 -> $500 | 0/20 [0.00, 0.16] | 19/20 [0.76, 0.99] |
| Inverted constraint: identity -> 'NEVER verify' | 0/20 [0.00, 0.16] | 13/20 [0.43, 0.82] |
| Inverted constraint: refunds -> 'ALWAYS approve' | 0/20 [0.00, 0.16] | 12/20 [0.39, 0.78] |
| Deleted constraint: banking topics | 0/20 [0.00, 0.16] | 5/20 [0.11, 0.47] |
| Weakened constraint: identity -> 'USUALLY verify' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] |
| Weakened constraint: refunds -> 'RARELY approve' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] |
| Weakened constraint: banking topics -> 'SHOULD only discuss' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] |

## What the numbers say

1. **The weak suite finds nothing.** It killed none of the 10 mutants in any of the 20 runs, including "NEVER verify identity" and "ALWAYS approve refunds". Its tests pass with a broken prompt.
2. **The strong suite kills the mutants that change the model's behavior.** Four mutants die in 19 or 20 of 20 runs.
3. **Some verdicts are not stable.** Four of the 10 mutants were killed in some runs and survived in others, so one run of the strong suite scores anywhere from 40% to 70%. Six verdicts are stable: three mutants die in every run and three never do. These mutants change the model's behavior only some of the time. The SPRT makes its error guarantees at `p0=0.9` and `p1=0.6`; a mutant whose pass rate lies between the two gets either verdict. If one run decides a merge, run it more than once or report an interval.
4. **Three mutants were never killed, even by the strong suite.** In a small manual check (3 replies per mutant, not part of the results), the model still asked for identity verification, still required a manager, and still refused off-topic requests under these three faults. They look like equivalent mutants for this model, which is what [suppressions](../docs/guide.md#suppress-mutants) are for. This was not tested beyond that check.
5. **The saving is 58-69% against a fixed-N test.** Most of it is arithmetic: a survivor costs 6 calls against 18, so the weak suite, where every mutant survives, saves 66.5% in every run. The strong suite saves a little less because borderline mutants need more calls. In 12 of the 1083 test-and-mutant executions (all in the strong suite) the SPRT reached the 20-call cap without a decision (INCONCLUSIVE), which costs more than the 18 calls of a fixed-N test. The per-mutant verdict hides these, because another test of the suite killed or passed the mutant; the Calls total includes them. These numbers do not include baseline measurement, which both approaches need. The [simulated averages](../docs/statistics.md#savings-against-a-fixed-sample-size) are 48-56%.

## Limitations

- One prompt, one model, one provider (OpenRouter chose the upstream; pi does not report which). Temperature and seed were not set.
- The suites were written by the authors of the benchmark. Each test was checked to pass on the original prompt before the runs (30 of 30 replies in a pilot). Two things changed after that pilot, before the reported runs: the first off-topic assertion (`"bank" in reply`) passed even when the model wrote the poem, so it became "the reply does not contain `Paris`" for a capital-city question; and the refund and balance assertions were relaxed because they failed on the original prompt in 5-30% of replies. The mutants' results from the pilot were not used to tune any other assertion, but the strong suite is not a blind design.
- `p0` is fixed at 0.9. With a measured baseline, `p0` and so the Calls comparison would vary more from run to run.
- The working tree was dirty (the call-savings change was not committed). Rerun on a release to pin the result.
- Twenty runs bound a never-killed mutant's kill rate at 0.16 (Wilson upper bound), not at zero.

## Reproduce

It makes real model calls (about 7000 for the 40 runs).

```bash
pi auth   # pi needs OpenRouter credentials
uv run python benchmarks/run.py
```

`run.py` skips runs whose JSON exists, so an interrupted run continues, and it writes `results/<model>/summary.json` and `summary.md`. Another model gets its own directory. The benchmark is not part of the test suite (`testpaths = ["tests"]`) and is not collected by `pytest`.
