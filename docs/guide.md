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
  - `--clastogen-fail-under MIN_SCORE`: the run fails if the score is less than `MIN_SCORE`. It checks only the score, not the constraint coverage; read `constraint_coverage.percent` in the JSON for that.
  - `--clastogen-json PATH`: writes the results to a JSON file. The file contains `mutation_score`, `total_mutants`, `counts` (all six statuses), `results` (one merged record for each mutant), `executions` (the record of each test for each mutant, which the HTML kill matrix uses), `baselines`, `by_operator`, `constraint_coverage`, `calls` and `fixed_calls`. For mutants that the SPRT did not run (`ERROR`, `SKIPPED`, `SUPPRESSED`), `sample_count` and `llr` are `null`.
  - `--clastogen-oracle`: runs the probe tests on every mutant to label which mutants change behavior. See [Probes](#probes-which-survivors-matter). The JSON then has a `behavior` section.
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
    oracle_samples=None,  # Make this test a probe that runs this many times (at least 5; default: not a probe)
)
```

### Probes: which survivors matter

A survivor is not always a blind spot. The model can keep its behavior without a rule, so the mutant changes nothing and no test could kill it. A **probe** is a test that tells the two cases apart. It is a normal pytest test that checks one behavior of the agent and passes on the original prompt:

```python
@pytest.mark.clastogen(target="my_app.agent:SYSTEM_PROMPT", oracle_samples=30)
def test_probe_no_discount_codes():
    assert "DISCOUNT" not in call_agent("Can you give me a discount code?")
```

With `pytest --clastogen --clastogen-oracle`, each probe runs `oracle_samples` times on the original prompt and as many times on every mutant of its target (also mutants another test killed). A mutant **changed behavior** when the probe's pass rate drops by at least `delta` and the drop is significant at `alpha` (one-sided Fisher exact test). Without the flag, a probe is only a normal test, and a probe never gets a `KILLED` or `SURVIVED` verdict of its own. Use the same `max_mutants` as the tests under check, so that both see the same mutants.

The report crosses the verdict with the label:

|                              | Behavior changed | No behavior change found               |
| ---------------------------- | ---------------- | -------------------------------------- |
| **Killed**                   | caught           | killed without a behavior change       |
| **Survived or inconclusive** | blind spot       | survived with no behavior change found |

A mutant that no probe measured is `unlabelled`. The terminal line splits the counts into defects (blind spots, killed without a behavior change) and not proven (survived with no behavior change found, unlabelled): a probe can miss a change, so "no behavior change found" never means equivalent. The JSON `behavior.labels` maps each measured mutant ID to `changed` or `no_change_found`. The line `Behavior-changing mutants killed: X of Y` is the share of caught mutants among the caught and blind spots. It stands next to the Mutation Score and does not replace it. `--clastogen-fail-under` still checks the score.

- A probe judges a mutant alone, and a mutant changed behavior when **any** probe regressed. A new probe can add a label, and it never removes one. But every probe and every mutant is one more chance of a false "changed" (alpha is not corrected for the number of mutants): raise `delta` or lower `alpha` when you have many probes. The merge is monotone, but model output is random: a re-run can measure a probe differently.
- "No behavior change found" is not a proof of equivalence. It means that no probe found a drop of `delta` or more. A rule that no probe covers can change behavior unseen.
- The cost is probes × mutants × `oracle_samples` calls, plus `oracle_samples` baseline calls for each probe. Suppressed mutants are skipped. A probe whose baseline pass rate is below 80%, or whose trial is skipped or errors, is not measured. A mutant whose probe trials error (for example, the injected prompt fails) is not measured either. Such mutants stay `unlabelled`.
- A probe always runs exactly `oracle_samples` times. It does not stop early, so a label does not depend on when you stop.
- Choose `oracle_samples` with this table. It gives the chance of the label `changed` for a mutant whose true pass rate drops from 0.9 (`delta=0.30`, `alpha=0.05`, computed with the same Fisher test as the plugin). With a baseline of 0.9, a mutant with no change gets the label in less than 1% of runs. With a baseline of 0.8, it can reach 1.5%.

  | `oracle_samples` | true drop 0.3 | true drop 0.5 | true drop 0.7 |
  | ---------------- | ------------- | ------------- | ------------- |
  | 10               | 29%           | 66%           | 94%           |
  | 20               | 58%           | 95%           | 100%          |
  | 30               | 56%           | 98%           | 100%          |

  The probe finds a drop of exactly `delta` in only 29% to 58% of runs. Thus, "no behavior change found" says little about small drops. Use at least 20 samples.

- Do not write a second test for a behavior that a test already checks. Put the assertion in a helper and call it from two tests: one plain test with the marker for the SPRT verdict, and one probe with `oracle_samples`.

### Read the prompt at call time

For each mutant, Clastogen replaces the module attribute (or class attribute) with the mutant. After all trials of that mutant, it puts back the original value. It also replaces module-level aliases that have the same name and refer to the same string object. An example is `from my_app.agent import SYSTEM_PROMPT` at the top of a different module.

Thus, your code must read the prompt when it calls the model. Clastogen does not change copies that your code made before the trial, such as a default argument (`def ask(msg, system=SYSTEM_PROMPT)`), a closure or a list that holds the string. Mutants of these copies always show `SURVIVED`.

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

You can use `pytest --clastogen -n 4`. Each worker sends its records with the test report. The controller merges the records into one summary. Each process measures its own baselines and keeps its own list of killed mutants. Thus, a worker can run a mutant again after a different worker killed it. The verdicts and the Mutation Score do not change, but the `Calls` line counts these extra runs, so it is higher than in a serial run. Parallel runs do more work in total, but they take less time.

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

1. **Mutation operators:** Clastogen finds rules by their words, not by meaning. A sentence is a candidate if it has a keyword (`must`, `never`, `always`, `avoid`, `only`, `require`, `if`, `unless`, `at most`, `within`, `without`, `escalate`, `refuse`, `reject`, `decline`, `approve` and similar) or any digit. Role sentences ("You are a …") are not candidates. A rule with none of these, for example "Respond in the customer's language", is not mutated. The keywords are English, so in a prompt in another language only the sentences with a digit are candidates. These operators apply to each candidate; an operator that finds nothing to change in a rule gives no mutant:

   | Operator                           | Change                                                                                                                                   |
   | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
   | `change_threshold`                 | the first number × 10                                                                                                                    |
   | `drop_exception`                   | removes the first "unless …" or "except …" clause, up to the next comma or semicolon                                                     |
   | `flip_comparator`                  | "at most" ↔ "at least", "no more than" → "at least", "fewer than" ↔ "more than", "maximum" ↔ "minimum"                                   |
   | `scope_swap`                       | "all", "every", "each" → "some"; "both" → "either"; "any" → "every"                                                                      |
   | `delete_constraint`                | removes the rule                                                                                                                         |
   | `invert_negation`                  | "never" ↔ "always", "must" → "must not" and similar                                                                                      |
   | `weaken_modal`                     | a hard word becomes a soft word, for example "must" becomes "should"                                                                     |
   | `delete_section`, `delete_example` | removes a whole Markdown heading section (one that holds a rule) or an `<example>` block or "Examples" section, with its heading or tags |

   Every rule and every block gets one mutant before any gets a second, in prompt order, and `max_mutants` caps the total; with more candidates than the budget, evenly spaced ones are used. Rules without a measured mutant (no mutant, or only `ERROR`, `SKIPPED` and `SUPPRESSED` ones) are listed as untested constraints. Deleting a whole section is often an easy kill, so read the score next to the per-operator rates. Semantic mutations from an LLM, RAG context poisoning and tool schema mutations are not available.

2. **Mutants that do not change behavior:** Sometimes a mutant does not change the behavior of the agent. For example, the foundation model can obey a safety rule from its training. Clastogen does not prove semantic equivalence. Write [probes](#probes-which-survivors-matter) to see which survivors change behavior, and use suppressions (`.clastogen/suppressions.toml`) for the mutants that do not. To judge a low score, read the `By operator` line (a score that comes only from `weaken_modal` misses is more likely equivalent mutants than one from `delete_constraint` misses) and the `Constraint Coverage` line (it counts the rules that at least one mutant killed; `survived` are rules that were measured but never killed, including `INCONCLUSIVE` ones; a rule without a measured mutant is `untested`, so set `max_mutants` at least as high as the number of rules when coverage matters. A killed `delete_section` mutant does not credit the rules inside the section, and the line does not say which test killed a rule). The same numbers are in the JSON as `by_operator` and `constraint_coverage`.
3. **Effect size:** The SPRT looks for an absolute decrease of `delta` (default 0.30) from a 10-run Laplace baseline. Thus, the maximum p0 is 11/12 ≈ 0.917. A small regression such as 0.95 → 0.88 needs approximately 60-70 runs for each mutant. With `max_steps=20`, the result is `INCONCLUSIVE` or `SURVIVED`. If small decreases are important, set a lower `delta` and a higher `max_steps` for each test (`@pytest.mark.clastogen(delta=0.10, max_steps=100)`). This increases the API cost.
4. **Baseline noise:** The first baseline run is the test run that already passed. Also, p0 is a point estimate. Thus, for baselines near the 80% limit, the false-kill rate for each mutant is more than alpha. These are the rates for an unchanged mutant (`delta=0.30`, `max_steps=20`, accepted baselines only): a true baseline of 0.90 gives 2.2%, 0.85 gives 5.0%, 0.80 gives 8.6% and 0.75 gives 13.2%. A higher limit does not fix this problem. With a 90% limit, 0.85 still gives 7.0%. The terminal, Markdown and HTML reports flag every test with a raw baseline rate below 90% as a noisy baseline, and the JSON `baselines` have a `noisy` field. Make the test more stable, or set p0 explicitly.
5. **Differential execution:** After a test kills a mutant, other tests do not run that mutant again. But Clastogen does not make a static dependency graph before execution.
