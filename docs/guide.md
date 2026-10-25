# Clastogen guide

## How to read the report

- **Statuses:**
  - `KILLED`: a test failed with the mutant.
  - `SURVIVED`: the tests continued to pass with the mutant.
  - `INCONCLUSIVE`: the SPRT got to `max_steps` before a decision.
  - `ERROR`: the evaluation itself failed, or the canary failed (see Canary). The JSON `error` field gives the cause. One exception in any trial stops the SPRT for that mutant, so configure retries in your LLM client for transient API errors.
  - `SKIPPED`: the test called `pytest.skip` during a trial, or its baseline was flaky (`baseline: unstable`).
  - `UNMEASURED`: `--clastogen-max-calls` had no budget left for this mutant. See [Trial budget](#trial-budget).
  - `SUPPRESSED`: you suppressed the mutant. See [Suppress mutants](#suppress-mutants).
- **Mutation Score** = `KILLED / (KILLED + SURVIVED + INCONCLUSIVE)`. The score does not include `ERROR`, `SKIPPED`, `UNMEASURED` and `SUPPRESSED` mutants. If more than one test examines a mutant, the strongest result is the result for that mutant. The order is `KILLED` > `SURVIVED` > `INCONCLUSIVE` > `ERROR` > `SKIPPED` > `UNMEASURED` > `SUPPRESSED`. The score line also shows a Wilson 95% interval; see [Statistics](statistics.md#statistical-caveats).
- **Errors fail the run:** A mutant with the result `ERROR` is not in the score. Thus any `ERROR` mutant makes the run fail, also without `--clastogen-fail-under`. A baseline error gives `ERROR` to the mutants of its test, and a failed canary (see below) gives it to the survivors of its test. A `pytest.skip` during the baseline gives `SKIPPED`. If a different test kills such a mutant, the result is `KILLED`.
- **Baselines:** Before mutation, Clastogen runs each marked test 9 more times without a mutant (10 runs in total). It calculates p0 with the Laplace estimate `(s + 1) / (n + 2)`. If the raw pass rate of a test is less than 80%, the test is flaky and Clastogen does not mutate it, and its mutants are `SKIPPED` with the reason `baseline: unstable`. The terminal summary shows the pass counts on one "Baselines" line. Use `-v` to show p0 for each test.
- **Canary:** After the mutants, when the test killed none of them and left some `SURVIVED` or `INCONCLUSIVE`, Clastogen runs the test 3 times with the target prompt replaced by an empty string. A test that passes all 3 times does not depend on the prompt at call time (for example, it uses a copy made at import), or the model satisfies it with its default behavior, so its survivors say nothing. They become `ERROR` with the reason `canary: test passes with an empty prompt; read the prompt at call time, or assert on behavior that the prompt causes`. `ERROR` is a deliberate choice: it fails the run, so a weak test cannot look like a pass. A kill of the same mutant by another test still takes precedence. A test that killed a mutant is not checked: the kill proves that it depends on the prompt. A crash or skip on the empty prompt is neutral: it is not a pass, and it gives no warning. The canary trials are not in the score and not in the `Calls:` line.
- **Flags:**
  - `--clastogen-fail-under MIN_SCORE`: the run fails if the score is less than `MIN_SCORE`, or if any mutant is `UNMEASURED`. It checks only the score, not the constraint coverage; read `constraint_coverage.percent` in the JSON for that.
  - `--clastogen-max-calls N`: runs at most N trials per process, see [Trial budget](#trial-budget).
  - `--clastogen-json PATH`: writes the results to a JSON file. The file contains `mutation_score`, `mutation_score_ci95`, `total_mutants`, `counts` (all seven statuses), `results` (one merged record for each mutant), `executions` (the record of each test for each mutant, which the HTML kill matrix uses), `baselines`, `by_operator`, `constraint_coverage` (the totals, `merge_rule`, and `rules`: one row per rule with `status`, `killed`, `mutants` and `killed_by`), `unmutated_sentences`, `calls` and `fixed_calls`. For mutants that the SPRT did not run (`ERROR`, `SKIPPED`, `UNMEASURED`, `SUPPRESSED`), `sample_count` and `llr` are `null`.
  - Error fields hold only the exception class name (for example `unexpected execution error: RuntimeError`). The full exception text and traceback go to the log, which `-rA` or `-rP` prints. Mutated prompt text still appears in the reports, so review them before you publish them.
  - `--clastogen-oracle`: runs the probe tests on every mutant to label which mutants change behavior. See [Probes](#probes-which-survivors-matter). The JSON then has a `behavior` section.
  - `--clastogen-html PATH`: writes the same data to one HTML file.
  - `--clastogen-md PATH`: writes a short Markdown summary: the score and one table of mutants. Use it for GitHub job summaries and pull request comments. See [CI](../README.md#ci).

## Run options

```bash
# Show a log for each step
pytest --clastogen -s --log-cli-level=INFO

# Preview the rules, mutants and worst-case test calls without running any test
pytest --clastogen --collect-only
```

The preview runs no test and no model call. `At most N test calls` counts the 3 canary runs (they run only when nothing is killed), the 9 baseline runs (only without `p0`) and up to `max_steps` runs per mutant. A real run uses fewer calls when a mutant is killed early. With `--clastogen-oracle`, each probe adds `oracle_samples * (1 + mutants)` trials; without it, probes add none. The `Calls:` line of the summary counts only the mutant trials, so it excludes the 9 baseline runs and the 3 canary runs of each test and is lower than the real number of trials.

### Marker options

```python
@pytest.mark.clastogen(
    target="my_app.agent:SYSTEM_PROMPT",  # Required: 'module:VAR' or 'module:Class.ATTR'
    max_mutants=5,  # Maximum number of built-in mutants (default: 5); custom mutants are extra
    delta=0.30,  # Minimum pass-rate decrease to find (default: 0.30)
    p0=0.90,  # Known baseline pass rate (default: measured in 10 baseline runs; if you set it, Clastogen does not do these runs)
    max_steps=20,  # Maximum number of runs for each mutant (default: 20)
    alpha=0.05,  # SPRT type I error rate for the target, divided by the number of tests T of that target (default: 0.05)
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

With `pytest --clastogen --clastogen-oracle`, each probe runs `oracle_samples` times on the original prompt and as many times on every mutant of its target (also mutants another test killed). A mutant **changed behavior** when the probe's pass rate drops by at least `delta` and the drop is significant at `alpha` (one-sided Fisher exact test). If not, the probe computes a one-sided upper bound of the drop (`1 - alpha` confidence, Newcombe's method). When the bound is at least `delta`, the probe is too weak to rule out a drop of `delta`, and the mutant is **suspected**. When the bound is below `delta`, the probe found **no behavior change**: on its own inputs, it ruled out a drop of `delta` or more. Without the flag, a probe is only a normal test, and a probe never gets a `KILLED` or `SURVIVED` verdict of its own. Use the same `max_mutants` as the tests under check, so that both see the same mutants.

The report crosses the verdict with the label:

|                              | Behavior changed | No behavior change found               |
| ---------------------------- | ---------------- | -------------------------------------- |
| **Killed**                   | caught           | killed without a behavior change       |
| **Survived or inconclusive** | blind spot       | survived with no behavior change found |

A mutant that no probe measured is `unlabelled`. A `suspected` mutant is in none of the four cells, whether the tests killed it or not. The terminal line splits the counts into defects (blind spots, killed without a behavior change), not proven (survived with no behavior change found, unlabelled) and suspected: a probe can miss a change, so "no behavior change found" never means equivalent. The JSON `behavior.labels` maps each measured mutant ID to `changed`, `suspected` or `no_change_found`. To resolve a suspected survivor, run more `oracle_samples` or sharpen the probe. The line `Behavior-changing mutants killed: X of Y` is the share of caught mutants among the caught and blind spots. It stands next to the Mutation Score and does not replace it. `--clastogen-fail-under` still checks the score.

- A probe judges a mutant alone, and a mutant gets the strongest label of any probe (`changed`, then `suspected`, then `no_change_found`). A new probe can raise a label, and it never lowers one. But every probe and every mutant is one more chance of a false "changed" (alpha is not corrected for the number of mutants): raise `delta` or lower `alpha` when you have many probes. The merge is monotone, but model output is random: a re-run can measure a probe differently.
- "No behavior change found" is not a proof of equivalence. It means that each probe that measured the mutant ruled out a drop of `delta` or more on its own inputs. Smaller drops, rules that no probe covers, and inputs that no probe sends can change behavior unseen. `suspected` measures the power of a probe on its inputs, not how well the inputs cover the prompt.
- A probe that checks only that the agent does not do something cannot see a deleted section. Without the rules, the agent often stays cautious, and the probe still passes. For a rule that allows an action, add a second probe where the right reply is to act.
- The cost is probes × mutants × `oracle_samples` calls, plus `oracle_samples` baseline calls for each probe. Suppressed mutants are skipped. A probe whose baseline pass rate is below 80%, or whose trial is skipped or errors, is not measured. A mutant whose probe trials error (for example, the injected prompt fails) is not measured either. Such mutants stay `unlabelled`.
- A probe always runs exactly `oracle_samples` times. It does not stop early, so a label does not depend on when you stop.
- Choose `oracle_samples` with these tables (`delta=0.30`, `alpha=0.05`, computed with the same tests as the plugin). The first gives the chance of the label `changed` for a mutant whose true pass rate drops from 0.9. With a baseline of 0.9, a mutant with no change gets the label in less than 1% of runs. With a baseline of 0.8, it can reach 1.5%.

  | `oracle_samples` | true drop 0.3 | true drop 0.5 | true drop 0.7 |
  | ---------------- | ------------- | ------------- | ------------- |
  | 10               | 29%           | 66%           | 94%           |
  | 20               | 58%           | 95%           | 100%          |
  | 30               | 56%           | 98%           | 100%          |

  A drop of exactly `delta` is called `changed` in only 29% to 58% of runs, and most of the other runs end as `suspected`, not as "no behavior change found". The second table gives the chance of the label `suspected`:

  | `oracle_samples` | no change (baseline 0.9) | true drop 0.3 | no change (baseline 0.8) |
  | ---------------- | ------------------------ | ------------- | ------------------------ |
  | 10               | 34%                      | 64%           | 42%                      |
  | 20               | 10%                      | 37%           | 22%                      |
  | 30               | 3%                       | 38%           | 9%                       |

  The bound is an approximation. Use at least 20 samples.

- Do not write a second test for a behavior that a test already checks. Put the assertion in a helper and call it from two tests: one plain test with the marker for the SPRT verdict, and one probe with `oracle_samples`.

### Read the prompt at call time

For each mutant, Clastogen replaces the module attribute (or class attribute) with the mutant. After all trials of that mutant, it puts back the original value. It also replaces module-level aliases that have the same name and refer to the same string object. An example is `from my_app.agent import SYSTEM_PROMPT` at the top of a different module.

Thus, your code should read the prompt when it calls the model. A second path covers some copies. When `httpx` or `httpx2` is installed, Clastogen also rewrites the original prompt text in the **system positions** of every outgoing httpx request (OpenAI `messages` or Responses `input` items with role `system` or `developer`, OpenAI Responses `instructions`, Anthropic top-level `system` as a string or as text blocks, Gemini `systemInstruction.parts`, Cohere v1 `preamble`), for sync and async clients. So the SDKs that use httpx (`openai`, `anthropic`) get the mutant even when your code copied the prompt at import time or an SDK object holds it, with no code change. It rewrites only where the original text occurs verbatim, and leaves every other request alone. It does not cover: clients that do not use httpx (`urllib`, `requests`, `aiohttp`), gRPC, streamed request bodies (`client.stream()` with a JSON body is rewritten), signed requests (an `Authorization: AWS4-HMAC-SHA256` header, e.g. Bedrock, is skipped because the signature covers the body), and prompts that your code assembles from several strings so that the original text never appears whole. Tested with real SDK objects on `httpx.MockTransport`: OpenAI (Chat and Responses) and Anthropic. Gemini and Cohere bodies are tested as raw requests, not through their SDKs, and the other httpx SDKs (Groq, Together, Ollama, Mistral…) are untested. The case-study benchmark uses the `pi` CLI, so it gives no evidence for it. httpx is not a Clastogen dependency: without it the path is a no-op.

The attribute replacement does not change copies that your code made before the trial, such as a default argument (`def ask(msg, system=SYSTEM_PROMPT)`), a closure or a list that holds the string. Mutants of these copies show `SURVIVED`; when the test killed no mutant, they show `ERROR` (see Canary).

For each trial, Clastogen removes and builds again the function-scoped fixtures. Thus, a fixture such as `bot = {"system": agent.PROMPT}` gets the mutant. Clastogen builds module-scoped and session-scoped fixtures only one time, so they keep the original prompt. Values that your code captures at import time also keep the original prompt. Mutants of these values also show `SURVIVED`; when the test killed no mutant, they show `ERROR` (see Canary).

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

### Alpha is divided by the number of tests

A mutant is killed if any test kills it, so with T tests on one target the chance of a false kill grows with T. Clastogen divides `alpha` by T for every SPRT of that target (Bonferroni). T is the number of tests with the same `target` selected in this session: marked tests without `oracle_samples`, counted after `-k` and `--lf` deselection, so a filtered run has a smaller T. Under pytest-xdist each test counts once; with `--dist each` every worker runs every test again, so T is multiplied by the worker count. Skipped, failing and flaky-baseline tests still count, because the count is taken at selection: that is safe and costs some power. The JSON report has T as `tests_per_target`. The cost is power: more mutants end `INCONCLUSIVE` as T grows (for a drop of exactly `delta` from 0.90: 18% at T=1, 41% at T=12, 53% at T=50 in the [simulation](statistics.md#scenario-3c-t-tests-on-one-target-bonferroni-split)). `max_steps` is unchanged; raise it or use fewer, stronger tests per target.

### Parallel runs (pytest-xdist)

You can use `pytest --clastogen -n 4`. Each worker sends its records with the test report. The controller merges the records into one summary. Each process measures its own baselines and keeps its own list of killed mutants. Thus, a worker can run a mutant again after a different worker killed it. The verdicts and the Mutation Score do not change, but the `Calls` line counts these extra runs, so it is higher than in a serial run. Parallel runs do more work in total, but they take less time.

## Trial budget

`--clastogen-max-calls N` caps the trials that a run makes. The original test call is not counted. A unit starts only when its worst case fits in what is left. The setup unit is the 9 baseline runs (only without `p0`) and one mutant's `max_steps` runs, so a test starts only when one mutant can still be funded after the baseline. A mutant unit is `max_steps` runs, the canary unit is 3 runs and is skipped when it does not fit, and a probe unit is `oracle_samples` runs. A unit that does not fit leaves its mutants `UNMEASURED`; a probe that does not fit records nothing for its mutants. `--collect-only` shows the uncapped worst case.
The budget is shared first-come across all tests in the run: a probe that runs first can spend the whole cap and leave a later mutation test `UNMEASURED`.

With xdist, the cap is split statically: each worker gets `N // workers`, and it must be at least the number of workers. An idle worker's share is unused, so a run with one busy test on two workers can spend at most half of `N`. A per-worker share too small for one unit measures nothing, and the run still exits 0 unless `--clastogen-fail-under` is set.

## Suppress mutants

Sometimes a mutant survives because the foundation model obeys the rule from its training. This is an equivalent mutant. Suppress equivalent mutants, so that they do not change your score. Clastogen shows suppressed mutants as `SUPPRESSED`, and the Mutation Score does not include them. The mutant ID is the 12-character hash in the summary. An example is `bb49e3ca17a0`, the inverted refund rule in the [Demo](../README.md#demo).

Put the IDs in `.clastogen/suppressions.toml` in your pytest `rootdir`. Each entry must have a `reason`:

```toml
[[suppressions]]
mutant_id = "bb49e3ca17a0"
reason = "Model refuses large refunds regardless of the prompt"
```

If the file is not correct, a `--clastogen` run stops with a usage error. Runs without `--clastogen` do not read the file.

An ID that matches no mutant in the run (a stale ID, or a mutant outside `max_mutants` or the tests you ran) is listed in a warning; the exit code does not change.

## Custom mutants

Rule discovery reads keywords and digits, so it misses rules written without them (for example "No drive-by refactors") and rules in other languages. Add your own mutants for them in `.clastogen/mutants.toml` in your pytest `rootdir`. You can write them by hand, or let your coding agent write them (see the Clastogen skill in `skills/clastogen/`); Clastogen itself calls no LLM, and the file is plain data, so a run stays deterministic.

```toml
[[mutants]]
target = "app:SYSTEM_PROMPT"                           # the same target string as in the marker
original = "Asla kişisel veriyi (TC, kart no) sorma"   # must occur verbatim in the prompt
mutated = "Kişisel veriyi (TC, kart no) sorabilirsin"  # "" deletes the snippet
description = "allows asking for personal data"
```

- Every field is required and must be a string; `mutated` may be empty. An unknown key, or an unknown top-level table such as `[[mutant]]`, is an error (the suppressions file is checked the same way).
- `original` must occur exactly once in the prompt (also not once in an `<example>` block and once outside it); the prompt of the mutant is the prompt with that text replaced by `mutated`. Take a longer snippet if the text repeats.
- `target` must be the same string as the `target` of the marker (for example `app:SYSTEM_PROMPT`). A target that no marker uses is not an error, only a warning in the terminal summary.
- A custom mutant has the operator `custom`, and its ID hashes the target, the operator and both texts, so [suppressions](#suppress-mutants) work on it. It is measured like any mutant, by every marked test of its target, and it counts in the score.
- `max_mutants` caps only the built-in mutants. Custom mutants are always added, so a marked test measures the custom mutants plus up to `max_mutants` built-in ones, and the call estimate of `--collect-only` includes them. If two mutants make the same prompt, only one is kept, and a custom mutant wins over a built-in one. `--clastogen-oracle` includes custom mutants too.
- An entry whose `original` does not occur exactly once in the current prompt, or whose `mutated` equals `original`, stops the run with a usage error that names the entry. With pytest-xdist, every marked test of that target fails with the same message instead. If the text looks right, check the Unicode normalization (NFC) and the line endings of the file and of the prompt.
- In the report, a sentence that contains a custom `original` is a rule with its own `k of n mutants killed` line, and it is not in `Unmutated sentences`. A trailing `.`, `!`, `?`, `;` or `:` of the `original` is ignored for this match, since report sentences have none. Write one sentence per entry: a multi-sentence `original` replaces text but credits no rule, and the terminal report lists it under "Custom mutants that match no single sentence". A sentence inside an `<example>` block credits no rule either. A short `original` that is part of several sentences credits all of them.
- A malformed file stops a `--clastogen` run with a usage error. Runs without `--clastogen` do not read it.

## Known limitations

1. **Mutation operators:** Clastogen finds rules by their words, not by meaning. A sentence is a candidate if it has a keyword (`must`, `never`, `always`, `avoid`, `only`, `require`, `if`, `unless`, `at most`, `within`, `without`, `escalate`, `refuse`, `reject`, `decline`, `approve` and similar) or any digit. Role sentences ("You are a …") are not candidates. A rule with none of these, for example "Respond in the customer's language", is not mutated. The keywords are English, so in a prompt in another language only the sentences with a digit are candidates; write [custom mutants](#custom-mutants) for the rest. These operators apply to each candidate; an operator that finds nothing to change in a rule gives no mutant:

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

   Every rule and every block gets one mutant before any gets a second, in prompt order, and `max_mutants` caps the total of built-in mutants; with more candidates than the budget, evenly spaced ones are used. Rules without a measured mutant (no mutant, or only `ERROR`, `SKIPPED` and `SUPPRESSED` ones) are listed as untested constraints. Deleting a whole section is often an easy kill, so read the score next to the per-operator rates. Clastogen does not generate semantic mutations itself; add them as [custom mutants](#custom-mutants). RAG context poisoning and tool schema mutations are not available.

2. **Mutants that do not change behavior:** Sometimes a mutant does not change the behavior of the agent. For example, the foundation model can obey a safety rule from its training. Clastogen does not prove semantic equivalence. Write [probes](#probes-which-survivors-matter) to see which survivors change behavior, and use suppressions (`.clastogen/suppressions.toml`) for the mutants that do not. To judge a low score, read the `By operator` line (a score that comes only from `weaken_modal` misses is more likely equivalent mutants than one from `delete_constraint` misses) and the `Constraint Coverage` line (it counts the rules that at least one mutant killed; `survived` are rules that were measured but never killed, including `INCONCLUSIVE` ones; a rule without a measured mutant is `untested`, so set `max_mutants` at least as high as the number of rules when coverage matters. A killed `delete_section` mutant does not credit the rules inside the section). Under the line, the `Rules` rows (weakest first, the terminal shows 10 unless `-v`) give `k of n mutants killed` for each rule and the test that killed one (`killed_by`, the lowest test ID among the killers; a serial run skips a mutant a test already killed and a parallel run does not, so the name can differ between them). A mutant is killed if any test kills it, and `n` counts the mutants that were measured. The `Unmutated sentences` list the sentences that rule discovery skipped (no obligation word and no digit, outside `<example>` blocks); no rule-level operator made a mutant for them (a `delete_section` or `delete_example` mutant can still remove them), so read it to see whether a rule is written in a way the discovery cannot find. The same numbers are in the JSON as `by_operator`, `constraint_coverage` and `unmutated_sentences`.
3. **Effect size:** The SPRT looks for an absolute decrease of `delta` (default 0.30) from a 10-run Laplace baseline. Thus, the maximum p0 is 11/12 ≈ 0.917. A small regression such as 0.95 → 0.88 needs approximately 60-70 runs for each mutant. With `max_steps=20`, the result is `INCONCLUSIVE` or `SURVIVED`. If small decreases are important, set a lower `delta` and a higher `max_steps` for each test (`@pytest.mark.clastogen(delta=0.10, max_steps=100)`). This increases the API cost.
4. **Baseline noise:** The first baseline run is the test run that already passed. Also, p0 is a point estimate. Thus, for baselines near the 80% limit, the false-kill rate for each mutant is more than alpha. These are the rates for an unchanged mutant with one test on the target (more tests: [alpha split](#alpha-is-divided-by-the-number-of-tests); `delta=0.30`, `max_steps=20`, accepted baselines only): a true baseline of 0.90 gives 2.2%, 0.85 gives 5.0%, 0.80 gives 8.6% and 0.75 gives 13.2%. A higher limit does not fix this problem. With a 90% limit, 0.85 still gives 7.0%. The terminal, Markdown and HTML reports flag every test with a raw baseline rate below 90% as a noisy baseline, and the JSON `baselines` have a `noisy` field. Make the test more stable, or set p0 explicitly.
5. **Differential execution:** After a test kills a mutant, other tests do not run that mutant again. But Clastogen does not make a static dependency graph before execution.
6. **Test types:** Only plain pytest test functions are supported; a `unittest.TestCase` test gets `ERROR`. Soft-assert plugins such as pytest-check record failures instead of raising, so every trial passes and the mutants are reported `SURVIVED`, and a check that fails during a trial is charged to the real test, so a passing test can be reported `FAILED`. Do not mark such tests.
