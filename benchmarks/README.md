# Benchmark: three real models, two eval suites

One system prompt, three real models, two eval suites that use the same inputs. Clastogen mutates the prompt and runs each suite 20 times per model. The question: does Clastogen tell a weak eval suite from a strong one, and how stable is its verdict when the model's output is random?

These are three measurements: one prompt, three models, one provider. Read the [limitations](#limitations) before you quote a number.

## Setup

- **Models:** `openrouter/deepseek/deepseek-v4.1-flash`, `openrouter/anthropic/claude-haiku-5.5` and `openrouter/openai/gpt-6-luna`, called through pi 1.1.0 with `--thinking off`. pi gets no tools, extensions, context files, skills, prompt templates or MCP servers, so only the system prompt and the user message reach the model (see the harness below).
- **Model harness:** pi 1.1.0 in print mode, one process per model call. The prompt and the user message are the only input, and pi gets no stdin. The arguments, from [`bank_agent.py`](bank_agent.py):

  ```bash
  pi -p --no-session --no-tools --no-extensions --no-context-files --no-skills --no-prompt-templates --no-mcp \
     --offline --thinking off --model <model> --system-prompt "<prompt or mutant>" "<user message>"
  ```

  `-p` prints the reply and exits. The `--no-*` flags remove everything but the model: no session history, tools, extensions, `AGENTS.md` and `CLAUDE.md` context files, skills, prompt templates or MCP servers. `--offline` only disables pi's startup network operations; the model call still goes to OpenRouter. `--thinking off` turns reasoning off. `--system-prompt` replaces pi's own system prompt with the banking prompt, which is where Clastogen's mutants go. Temperature and seed are not set.
- **Eval harness:** pytest with the Clastogen plugin, one process per run, started by [`run.py`](run.py): `pytest --clastogen --clastogen-json=<run>.json -p no:cacheprovider benchmarks/test_<suite>.py`. Each suite has `@pytest.mark.clastogen(target="bank_agent:SYSTEM_PROMPT", p0=0.9, max_mutants=20)`. `run.py` runs four such processes at a time (`WORKERS = 4`).
- **Prompt:** a banking agent with three rules (verify identity before a balance, no refund over $50 without a manager, only banking topics). Clastogen makes 10 mutants from it.
- **Weak suite** ([`test_weak.py`](test_weak.py)): checks that the agent replied. Assertions such as "the reply is not empty" or "the reply mentions the order".
- **Strong suite** ([`test_strong.py`](test_strong.py)): same inputs, but each test asserts the decision that one rule demands.
- **Two views of the suites.** The main result uses the two tests that pass on the original prompt for all three models (balance and refund). The third test, the off-topic one, is reported separately: see [the off-topic test](#the-off-topic-test).
- **Settings:** `p0=0.9` fixed in the marker (no baseline measurement), `alpha=0.05`, `beta=0.10`, serial runs without xdist, 20 runs per suite and model.
- **Fixed-N:** the `Calls` comparison is a fixed-N test with the same error rates (18 calls per test and mutant). It is hypothetical: it was not run.

## Results

Each model has a summary written by `run.py`: [DeepSeek](results/deepseek_deepseek-v4.1-flash/summary.md), [Haiku](results/anthropic_claude-haiku-5.5/summary.md) and [Luna](results/openai_gpt-6-luna/summary.md). They show the score, the calls against the fixed-N test, and the share of runs that killed each mutant with its Wilson 95% interval. The raw JSON and logs of the runs are git-ignored; `run.py` writes them next to the summary.

Balance and refund tests only, 20 runs each:

| Model | Suite | Mutation Score % (min-max) | Calls | Saving % (min-max) | INCONCLUSIVE executions |
| --- | --- | --- | --- | --- | --- |
| DeepSeek | weak | 0.0 (0.0-0.0) | 121 | 66.4 (65.6-66.7) | 0/400 |
| DeepSeek | strong | 42.0 (20.0-50.0) | 119.2 | 64.0 (56.4-68.5) | 8/367 |
| Haiku | weak | 0.0 (0.0-0.0) | 120 | 66.7 (66.7-66.7) | 0/400 |
| Haiku | strong | 38.5 (30.0-50.0) | 121.5 | 64.2 (59.1-70.4) | 4/377 |
| Luna | weak | 0.0 (0.0-0.0) | 120 | 66.7 (66.7-66.7) | 0/400 |
| Luna | strong | 56.0 (50.0-60.0) | 102.6 | 68.3 (65.7-71.6) | 3/360 |

Kills of the strong suite out of 20 runs, per mutant (the four mutants that no model's strong suite ever killed are left out):

| Mutant | DeepSeek | Haiku | Luna |
| --- | --- | --- | --- |
| Delete the refund rule | 20 | 20 | 20 |
| Delete the identity rule | 20 | 20 | 20 |
| Refund threshold $50 to $500 | 19 | 14 | 20 |
| "ALWAYS approve" refunds | 12 | 20 | 20 |
| "NEVER verify" identity | 13 | 3 | 20 |
| "RARELY approve" refunds | 0 | 0 | 12 |

### The off-topic test

The third strong test asks "What is the capital of France?" and asserts that the reply does not contain `Paris`. It also needs the original prompt to pass, and the models differ on that:

| Model | Passed on the original prompt |
| --- | --- |
| DeepSeek | 20 of 20 runs |
| Haiku | 14 of 20 runs |
| Luna | 7 of 20 runs |

This is a measurement of the model, not of the eval: Haiku and especially Luna often answer an off-topic question under a prompt that says "only banking topics". Clastogen does not mutate a test that fails on the original prompt, so the off-topic test was not mutated in 6 of Haiku's 20 runs and in 13 of Luna's.

When the test did pass on the original prompt, it killed every topic mutant in all 14 of Haiku's and all 7 of Luna's such runs, and "USUALLY verify identity" too. A test with a true pass rate far below 0.9 breaks the SPRT's assumption `p0=0.9`, so these kills are noise, not evidence about the mutants. The summaries show this as the "with the off-topic test" column and keep the 2-test view as the main result.

## What the numbers say

1. **The weak suite finds nothing.** It killed none of the 10 mutants in any of the 60 runs, with any model, including "NEVER verify identity" and "ALWAYS approve refunds". Its tests pass with a broken prompt.
2. **The strong suite kills the mutants that change the model's behavior.** Clastogen separates the two suites with every model. Three mutants die in 19 or 20 of 20 runs with every model, five with Luna.
3. **The same mutant does not behave the same with every model.** "ALWAYS approve refunds" dies in 12 of 20 runs with DeepSeek and in 20 of 20 with Haiku; "NEVER verify identity" dies in 13 of 20 with DeepSeek, in 3 of 20 with Haiku and in 20 of 20 with Luna; "RARELY approve refunds" dies in 12 of 20 runs with Luna and never with the other two. A mutation score belongs to a prompt and a model, not to a prompt alone.
4. **Some verdicts are not stable.** Two mutants with Haiku, three with DeepSeek and one with Luna are killed in some runs and survive in others, so one run of the strong suite scores anywhere from 20% to 50% (DeepSeek), 30% to 50% (Haiku) or 50% to 60% (Luna). The SPRT makes its error guarantees at `p0=0.9` and `p1=0.6`; a mutant whose pass rate lies between the two gets either verdict. If one run decides a merge, run it more than once or report an interval.
5. **Four mutants were never killed by the balance and refund tests, with any model.** Three of them change the topic rule, which these tests do not examine. The fourth is "USUALLY verify identity". In a small manual check with DeepSeek (3 replies per mutant, not saved and not part of the results), the model still asked for identity verification, still required a manager, and still refused off-topic requests under "USUALLY verify", "RARELY approve" and "SHOULD only discuss banking". They may be equivalent mutants for that model, which is what [suppressions](../docs/guide.md#suppress-mutants) are for. Treat that as a hypothesis: the check was not recorded, and it was not repeated with Haiku or Luna. Luna's 12 kills of "RARELY approve refunds" show that such a mutant can be equivalent for one model and not for another.
6. **The saving is 56-72% against a fixed-N test.** Most of it is arithmetic: a survivor costs 6 calls against 18, so the weak suite, where every mutant survives, saves about 66.7% in every run. The strong suite saves a little less because borderline mutants need more calls. In 15 of the 1104 test-and-mutant executions of the strong suites (8 with DeepSeek, 4 with Haiku, 3 with Luna) the SPRT reached the 20-call cap without a decision (INCONCLUSIVE), which costs more than the 18 calls of a fixed-N test. The per-mutant verdict hides these, because another test of the suite killed or passed the mutant; the Calls total includes them. These numbers do not include baseline measurement, which both approaches need. The [simulated averages](../docs/statistics.md#savings-against-a-fixed-sample-size) are 48-56%.

## Limitations

- One prompt, three models, one provider (OpenRouter chose the upstream; pi does not report which). Temperature and seed were not set.
- The suites were written by the authors of the benchmark and piloted on DeepSeek. Each test was checked to pass on the original prompt before the runs (30 of 30 replies in a pilot). That bias favors DeepSeek: the suite was fitted to its replies, and Haiku's off-topic test fails on the original prompt in some replies. Two things changed after the pilot, before the reported runs: the first off-topic assertion (`"bank" in reply`) passed even when the model wrote the poem, so it became "the reply does not contain `Paris`" for a capital-city question; and the refund and balance assertions were relaxed because they failed on the original prompt in 5-30% of replies. The mutants' results from the pilot were not used to tune any other assertion, but the strong suite is not a blind design. The pilot and the manual check were not saved, so their figures (30 of 30, 5-30%, 3 replies) cannot be reproduced from this repository; only the 20-run results below `results/` can.
- `p0` is fixed at 0.9. A test with a lower true pass rate on the original prompt, such as the off-topic test with Haiku and Luna, breaks the SPRT's assumption and kills mutants by noise. With a measured baseline, Clastogen would not mutate such a test, and `p0` and so the Calls comparison would vary more from run to run.
- All runs used a dirty tree (`git_dirty` in `summary.json`). DeepSeek ran on commit `68da7ba`, before the `benchmarks/` directory was committed, so that commit does not pin the suites. Haiku and Luna ran on `35d3fdb` with uncommitted changes to `run.py` and `bank_agent.py` (`stdin`, the model variable). The committed suites and prompt are the files Haiku and Luna ran; that DeepSeek ran the same text was not verified. The summaries were recomputed from the raw runs with the current `run.py`.
- pi accepted `--thinking off` for Luna, but whether that disables its reasoning is unknown.
- Twenty runs bound a never-killed mutant's kill rate at 0.16 (Wilson upper bound), not at zero.

## Reproduce

It makes real model calls (about 7000 for the 40 runs).

```bash
pi auth   # pi needs OpenRouter credentials
uv run python benchmarks/run.py
BENCHMARK_MODEL=openrouter/anthropic/claude-haiku-5.5 uv run python benchmarks/run.py   # another model
```

`run.py` skips runs whose JSON exists, so an interrupted run continues, and it writes `results/<model>/summary.json` and `summary.md`. Each model gets its own results directory. The benchmark is not part of the test suite (`testpaths = ["tests"]`) and is not collected by `pytest`.
