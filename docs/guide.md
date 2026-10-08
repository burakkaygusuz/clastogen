# Clastogen guide

## How to read the report

- **Statuses:**
  - `KILLED`: a test failed with the mutant.
  - `SURVIVED`: the tests continued to pass with the mutant.
  - `INCONCLUSIVE`: the SPRT got to `max_steps` before a decision.
  - `ERROR`: the evaluation itself failed. The JSON `error` field gives the cause. One exception in any trial stops the SPRT for that mutant, so configure retries in your LLM client for transient API errors.
  - `SKIPPED`: the test called `pytest.skip` during a trial.
  - `SUPPRESSED`: you suppressed the mutant. See [Suppress mutants](#suppress-mutants).
- **Mutation Score** = `KILLED / (KILLED + SURVIVED + INCONCLUSIVE)`. The score does not include `ERROR`, `SKIPPED` and `SUPPRESSED` mutants. If more than one test examines a mutant, the strongest result is the result for that mutant. The order is `KILLED` > `SURVIVED` > `INCONCLUSIVE` > `ERROR` > `SKIPPED` > `SUPPRESSED`.
- **Errors fail the run:** A mutant with the result `ERROR` is not in the score. Thus any `ERROR` mutant makes the run fail, also without `--clastogen-fail-under`. A baseline error gives `ERROR` to the mutants of its test, and a `pytest.skip` during the baseline gives `SKIPPED`. If a different test kills such a mutant, the result is `KILLED`.
- **Baselines:** Before mutation, Clastogen runs each marked test 9 more times without a mutant (10 runs in total). It calculates p0 with the Laplace estimate `(s + 1) / (n + 2)`. If the raw pass rate of a test is less than 80%, the test is flaky and Clastogen does not mutate it. The terminal summary shows the pass counts on one "Baselines" line. Use `-v` to show p0 for each test.
- **Flags:**
  - `--clastogen-fail-under MIN_SCORE`: the run fails if the score is less than `MIN_SCORE`.
  - `--clastogen-json PATH`: writes the results to a JSON file. The file contains `mutation_score`, `total_mutants`, `counts` (all six statuses), `results` (one merged record for each mutant), `executions` (the record of each test for each mutant, which the HTML kill matrix uses) and `baselines`. For mutants that the SPRT did not run (`ERROR`, `SKIPPED`, `SUPPRESSED`), `sample_count` and `llr` are `null`.
  - `--clastogen-html PATH`: writes the same data to one HTML file.
  - `--clastogen-md PATH`: writes a short Markdown summary: the score and one table of mutants. Use it for GitHub job summaries and pull request comments. See [CI](../README.md#ci).

## Run options

```bash
# Show a log for each step
pytest --clastogen -s --log-cli-level=INFO
```

### Marker options

```python
@pytest.mark.clastogen(
    target="my_app.agent:SYSTEM_PROMPT",  # Required: 'module:VAR' or 'module:Class.ATTR'
    max_mutants=5,  # Maximum number of mutants (default: 5)
    delta=0.30,  # Minimum pass-rate decrease to find (default: 0.30)
    p0=0.90,  # Known baseline pass rate (default: measured in 10 baseline runs; if you set it, Clastogen does not do these runs)
    max_steps=20,  # Maximum number of runs for each mutant (default: 20)
    alpha=0.05,  # SPRT type I error rate: chance to kill an unchanged mutant (default: 0.05)
    beta=0.10,  # SPRT type II error rate: chance to miss a real decrease (default: 0.10)
)
```

### Read the prompt at call time

For each trial, Clastogen replaces the module attribute (or class attribute) with the mutant. After the trial, it puts back the original value. It also replaces module-level aliases that have the same name and refer to the same string object. An example is `from my_app.agent import SYSTEM_PROMPT` at the top of a different module.

Thus, your code must read the prompt when it calls the model. Clastogen does not change copies that your code made before the trial. Mutants of these copies always show `SURVIVED`.

For each trial, Clastogen removes and builds again the function-scoped fixtures. Thus, a fixture such as `bot = {"system": agent.PROMPT}` gets the mutant. Clastogen builds module-scoped and session-scoped fixtures only one time, so they keep the original prompt. Values that your code captures at import time also keep the original prompt. Mutants of these values also show `SURVIVED`.

All trials run in the call phase of the original test. Hooks of other plugins apply one time to the test and all its trials. For example, a `pytest-timeout` limit applies to the total time of the test and its trials.

Incorrect: the code captures the prompt one time at import time. It does not read the changed attribute.

```python
# my_app/agent.py
SYSTEM_PROMPT = "You must never approve refunds over $50."
# A copy made at import. F-strings, default args and clients made at import have the same problem.
CONFIG = {"system": SYSTEM_PROMPT}


def ask(question: str) -> str:
    return client.chat(system=CONFIG["system"], user=question)
```

Correct: the code reads the module attribute at each call. A `from my_app.agent import SYSTEM_PROMPT` in the function body also works.

```python
# my_app/agent.py
SYSTEM_PROMPT = "You must never approve refunds over $50."


def ask(question: str) -> str:
    return client.chat(system=SYSTEM_PROMPT, user=question)
```

### Parallel runs (pytest-xdist)

You can use `pytest --clastogen -n 4`. Each worker sends its records with the test report. The controller merges the records into one summary. Each process measures its own baselines and keeps its own list of killed mutants. Thus, a worker can run a mutant again after a different worker killed it. Parallel runs do more work in total, but they take less time.

## Suppress mutants

Sometimes a mutant survives because the foundation model obeys the rule from its training. This is an equivalent mutant. Suppress equivalent mutants, so that they do not change your score. Clastogen shows suppressed mutants as `SUPPRESSED`, and the Mutation Score does not include them. The mutant ID is the 12-character hash in the summary. An example is `7d1281020f96`, the inverted refund rule in the [Demo](../README.md#demo).

Put the IDs in `.clastogen/suppressions.toml` in your pytest `rootdir`. Each entry must have a `reason`:

```toml
[[suppressions]]
mutant_id = "7d1281020f96"
reason = "Model refuses large refunds regardless of the prompt"
```

If the file is not correct, a `--clastogen` run stops with a usage error. Runs without `--clastogen` do not read the file.

## Known limitations

1. **Mutation operators:** Clastogen finds rules by their words. A sentence is a candidate only if it contains a keyword such as `must`, `never`, `always`, `avoid`, `only`, `may not` or `require`. Clastogen does not mutate rules without one of these keywords, for example "Escalate to a human if …". Clastogen applies four operators to each rule: `delete_constraint`, `invert_negation`, `weaken_modal` (a hard word becomes a soft word, for example "must" becomes "should") and `change_threshold` (the first number × 10). Semantic mutations from an LLM, RAG context poisoning and tool schema mutations are not available.
2. **Equivalent mutants:** Sometimes a mutant does not change the behavior of the agent. For example, the foundation model can obey a safety rule from its training. Clastogen does not prove semantic equivalence. Use suppressions (`.clastogen/suppressions.toml`) for these mutants.
3. **Effect size:** The SPRT looks for an absolute decrease of `delta` (default 0.30) from a 10-run Laplace baseline. Thus, the maximum p0 is 11/12 ≈ 0.917. A small regression such as 0.95 → 0.88 needs approximately 60-70 runs for each mutant. With `max_steps=20`, the result is `INCONCLUSIVE` or `SURVIVED`. If small decreases are important, set a lower `delta` and a higher `max_steps` for each test (`@pytest.mark.clastogen(delta=0.10, max_steps=100)`). This increases the API cost.
4. **Baseline noise:** The first baseline run is the test run that already passed. Also, p0 is a point estimate. Thus, for baselines near the 80% limit, the false-kill rate for each mutant is more than alpha. These are the rates for an unchanged mutant (`delta=0.30`, `max_steps=20`, accepted baselines only): a true baseline of 0.90 gives 2.2%, 0.85 gives 5.0%, 0.80 gives 8.6% and 0.75 gives 13.2%. A higher limit does not fix this problem. With a 90% limit, 0.85 still gives 7.0%. Make the test more stable, or set p0 explicitly.
5. **Differential execution:** After a test kills a mutant, other tests do not run that mutant again. But Clastogen does not make a static dependency graph before execution.
