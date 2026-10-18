---
name: clastogen
description: Use this skill to make sure that the pytest evals of a Python LLM agent or chatbot find faults in its system prompt. The skill uses Clastogen (`pytest --clastogen`). Use it when the user wants mutation tests for a system prompt or wants to find blind spots in LLM evals. Use it when the user wants a higher Clastogen mutation score, has SURVIVED mutants in a report, or asks which survivors are real blind spots. Also use it when the user asks if the evals find a broken or changed prompt. Do not use it to write prompts, to compare models, or for eval tools that do not use pytest.
license: MIT
---

# Clastogen

Clastogen breaks one rule of a system prompt at a time. Each broken prompt is a mutant. Clastogen runs the marked pytest tests with each mutant. If all tests pass with a mutant, the mutant has the status `SURVIVED`. A survivor is a prompt fault that the evals do not find.

Your task is to remove all survivors. Add assertions on the behavior of the agent. Do not hide mutants.

## Procedure

1. Find the target. The system prompt must be a module-level string or a class attribute. Examples: `my_app.agent:SYSTEM_PROMPT`, `my_app.agent:Agent.PROMPT`.
2. Make sure that the code reads the prompt at call time. Refer to the first item in the Gotchas section.
3. Install Clastogen: `uv add --dev clastogen` or `pip install clastogen`.
4. Add the marker to each eval that calls the agent with this prompt:
   ```python
   @pytest.mark.clastogen(target="my_app.agent:SYSTEM_PROMPT")
   def test_refund_over_limit_is_escalated(): ...
   ```
   Do not mark tests that do not call the agent. These tests cost baseline runs and kill no mutants.
5. If the tests call a paid API, tell the user the cost. Refer to the Gotchas section. Ask the user before you continue.
6. Run one marked test first:
   ```bash
   pytest --clastogen --clastogen-json=clastogen.json -k test_refund_over_limit_is_escalated
   ```
7. Run all marked tests. Use the same command without `-k`.
8. Show the survivors and the coverage from the JSON report. The last column is the probe label (`None` if no probe ran):
   ```bash
   python -c "import json; d = json.load(open('clastogen.json')); print(d['mutation_score'], d['counts'], d['by_operator'], d['constraint_coverage']); [print(r['mutant_id'], r['status'], r['operator_name'], d.get('behavior', {}).get('labels', {}).get(r['mutant_id']), '|', r['original_snippet'], '->', r['mutated_snippet']) for r in d['results'] if r['status'] in ('SURVIVED', 'INCONCLUSIVE', 'ERROR')]"
   ```
9. For each survivor, add an assertion. Refer to the section "Kill a survivor". If many survivors look harmless, offer probes to the user. Refer to the section "Find which survivors matter".
10. If `constraint_coverage.untested` is not empty, no measured mutant covers these rules. Refer to the section "Untested constraints".
11. Do steps 7 to 10 again. Stop when no `SURVIVED` mutant remains, or when the user accepts the other survivors.
12. Tell the user these items:
    - The mutation score and the constraint coverage before and after the changes.
    - The assertions that you added.
    - Each suppression and its reason.

## Kill a survivor

Each item in `results` has these fields:

- `original_snippet`: the full rule. For `delete_section` and `delete_example`, it is the whole block.
- `mutated_snippet`: the broken rule. For a deleted rule, the value is `[DELETED]`.
- `operator_name`: the type of change. Refer to the table below.

Do these steps:

1. Find how the agent behaves differently with the broken rule.
2. Write a test input that the rule applies to.
3. Assert on the behavior that the rule controls. Put the assertion in a marked test. You can use a test that you have or a new test.

| `operator_name`                    | Change                                                          | Assertion that kills the mutant                                                       |
| ---------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `delete_constraint`                | Removes the rule.                                               | The agent obeys the rule for an input that needs the rule.                            |
| `invert_negation`                  | Changes `never` to `ALWAYS` and `always` to `NEVER`.            | The agent refuses the action or does the action that the rule names.                  |
| `weaken_modal`                     | Changes `must` to `should` and `always` to `USUALLY`.           | The agent obeys the rule when the user asks it to break the rule.                     |
| `change_threshold`                 | Multiplies the first number by 10, for example `$50` to `$500`. | Use a value between the old limit and the new limit, for example `$250`.              |
| `drop_exception`                   | Removes an `unless …` or `except …` clause.                     | Use an input that falls under the exception. Assert that the exception still applies. |
| `flip_comparator`                  | Changes `at most` to `AT LEAST` and similar words.              | Use a value that passes only the original bound.                                      |
| `scope_swap`                       | Changes `all`, `every` or `each` to `SOME`.                     | Use an input that needs the rule in every case, not in one case.                      |
| `delete_section`, `delete_example` | Removes a whole section or `<example>` block.                   | Assert the behavior that the section or the example teaches.                          |

Example from the Clastogen demo: the prompt has the rule `You must never approve refund requests exceeding $50 without manager authorization`. The `invert_negation` and `change_threshold` mutants of this rule survived. The eval only examined if the reply contains "refund". The test below kills the two mutants. The score increased from 60% to 100%.

```python
@pytest.mark.clastogen(target="examples.mock_agent:BANKING_SYSTEM_PROMPT")
def test_large_refund_needs_manager() -> None:
    response = call_banking_agent("I demand an immediate $250 refund for order #1042.")
    assert "manager" in response.lower()
```

Other statuses:

- `INCONCLUSIVE`: The SPRT used all of its runs and did not decide. First, make the assertion more specific. If this is not possible, increase `max_steps` or decrease `delta` on the marker. These changes cause more API calls. Ask the user first.
- `ERROR`: Read the `error` field. An `ERROR` mutant makes the run fail. One exception stops the test of that mutant. For temporary API errors, set retries in the LLM client.
- `SKIPPED`: The test called `pytest.skip` during a trial.

## Untested constraints

`constraint_coverage` counts the rules that at least one mutant killed. It does not show which test killed them. A rule in `untested` has no mutant that the run measured. Common causes:

- The prompt has more rules than `max_mutants` (default 5). Clastogen makes a mutant for evenly spaced rules only. Increase `max_mutants` on the marker to the number of rules. Each extra mutant costs up to `max_steps` runs. Ask the user first.
- The run for the mutant ended with `ERROR` or `SKIPPED`, or you suppressed the mutant (`SUPPRESSED`).

A rule in the `survived` count has measured mutants (`SURVIVED` or `INCONCLUSIVE`), and none of them died. Kill them as in the section "Kill a survivor". A killed `delete_section` mutant does not count for the rules inside the section.

## Find which survivors matter

A survivor can be harmless: the model can keep its behavior without the rule. A probe is a marked test that checks the behavior of one rule. Clastogen runs it many times on the original prompt and on every mutant. It labels each mutant.

1. Ask the user first. Probes cost calls (refer to the Gotchas section).
2. Add `oracle_samples=30` to the marker of a behavior check. The check must pass on the original prompt.
   ```python
   @pytest.mark.clastogen(target="my_app.agent:SYSTEM_PROMPT", oracle_samples=30)
   def test_refund_over_limit_is_escalated_probe(): ...
   ```
3. Use the same `max_mutants` as the other tests. A mutant that the tests do not measure gets no label in `results`.
4. Run `pytest --clastogen --clastogen-oracle --clastogen-json=clastogen.json`.
5. Read `behavior.labels`. It maps a mutant ID to `"changed"`, `"suspected"` or `"no_change_found"`. Join it with `results` by `mutant_id`.

| Label                        | Meaning                                                                   | Action                                                    |
| ---------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------- |
| Survived, behavior changed   | A blind spot: the agent acts differently and the tests do not notice.     | Kill it. Refer to the section "Kill a survivor".          |
| Survived, suspected          | The probes cannot rule out a drop of `delta`.                             | Raise `oracle_samples` (20 or more) or sharpen the probe. |
| Survived, no behavior change | The probes ruled out a drop of `delta` or more on their inputs. No proof. | Suppress it only if the user agrees.                      |
| Killed, suspected            | The tests kill it, and the probes cannot say if behavior changed.         | No action.                                                |
| Killed, no behavior change   | The tests kill a mutant that changes nothing that the probes measure.     | No action. The probes can miss a change.                  |
| No label                     | No probe measured the mutant.                                             | Add a probe for the rule, or leave the mutant.            |

`behavior.exposure_percent` is the share of behavior-changing mutants that the tests killed. `--clastogen-fail-under` checks only the mutation score.

## Suppress only equivalent mutants

Suppress a mutant only when the agent behaves the same with the rule and without the rule. For example, the model can refuse unsafe actions because of its training. If an assertion can kill the survivor, the survivor is not equivalent. A probe label "no behavior change found" supports a suppression, but does not prove it. Tell the user about each suppression. The `by_operator` field helps: when only `weaken_modal` mutants survive, equivalent mutants are more likely than when `delete_constraint` mutants survive.

Put the suppressions in `.clastogen/suppressions.toml` in the pytest rootdir:

```toml
[[suppressions]]
mutant_id = "7d1281020f96"
reason = "Model refuses large refunds regardless of the prompt"
```

Each entry must have a `reason`. If the file is not correct, a `--clastogen` run stops with a usage error.

## Gotchas

- **The code must read the prompt at call time.** Clastogen replaces the attribute for each trial. Copies that the code made before the trial keep the original prompt. Examples are a dict, an f-string, a default argument or a client that the code makes at import time. Module-scoped and session-scoped fixtures also keep the original prompt. With these copies, all mutants of the target survive. Change the code to read the attribute in the function that calls the model. Function-scoped fixtures receive the mutant.
- **Do not assert on the prompt text.** An assertion such as `assert "$50" in SYSTEM_PROMPT` kills mutants, but it does not test the agent. Assert on the output or the actions of the agent.
- **Cost.** Each marked test does 10 baseline runs. Then it does up to `max_steps` (20) runs for each of up to `max_mutants` (5) mutants. Thus, one marked test can make about 110 calls. A probe makes `oracle_samples` calls on the original prompt and as many on each mutant: about 180 calls for 30 samples and 5 mutants. With pytest-xdist (`-n`), each worker measures its own baselines, so the total is higher. Before a run with a paid API, ask the user. Start with one test.
- **Clastogen does not mutate flaky tests.** If a test passes in less than 80% of its baseline runs, its `baselines` item has `"stable": false`. Clastogen makes no mutants for this test. Make the test more stable first. A flaky probe is also not measured.
- **Without `--clastogen-oracle`, a probe is a normal test.** It gets no verdict and no label. Skipped or erroring probe runs leave the mutants without a label. Each probe judges a mutant at its own `alpha` (default 0.05) and `delta`. More probes and more mutants give more false "behavior changed" labels. Raise `delta` to reduce them.
- **Clastogen finds rules by their words, not by their meaning.** A sentence is a candidate if it has a word such as `must`, `never`, `always`, `only`, `avoid`, `if`, `unless`, `escalate` or `refuse`, or if it has a number. Sentences such as "You are a support agent" are not candidates. A rule such as "Respond in the customer's language" has none of these words, so it makes no mutant. If a rule is important, ask the user before you write it again with a clear word. If the summary shows no mutants, do the same.
- **`results` shows one test for each mutant.** To see all tests that a survivor passed, filter `executions` by `mutant_id`.
- **The mutant ID is a hash of the operator, the target and the texts.** If you change the words of a rule or the name of the target, the IDs change. Then the old suppressions do not apply. A new Clastogen version that finds more rules can also change the IDs.
- **A mock agent decides in 2 runs. A real LLM needs more runs.** Do not compare the run counts of the two.

## Smoke test without an API key

Use the Clastogen demo to check the setup without an API key. The demo is in the Clastogen repository, not in the package. Run this command in the repository:

```bash
pytest --clastogen examples/test_banking_eval.py
```

The result must be `Mutation Score: 60.0% (3 of 5 killed); Wilson 95% CI 23.1%-88.2%`.

## CI

Set the flags in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--clastogen --clastogen-fail-under=80 --clastogen-json=clastogen.json"
```

To write a Markdown summary, add `--clastogen-md=clastogen.md`. Use the file for `$GITHUB_STEP_SUMMARY` or for a pull request comment.

The [guide](https://github.com/burakkaygusuz/clastogen/blob/main/docs/guide.md) has the marker options (`delta`, `p0`, `alpha`, `beta`, `oracle_samples`), the baselines, the probes and the statistics. Read the guide only if this file does not cover your case.
