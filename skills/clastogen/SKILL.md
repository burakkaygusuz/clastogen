---
name: clastogen
description: Use this skill to make sure that the pytest evals of a Python LLM agent or chatbot find faults in its system prompt. The skill uses Clastogen (`pytest --clastogen`). Use it when the user wants mutation tests for a system prompt or wants to find blind spots in LLM evals. Use it when the user wants a higher Clastogen mutation score or has SURVIVED mutants in a report. Also use it when the user asks if the evals find a broken or changed prompt. Do not use it to write prompts, to compare models, or for eval tools that do not use pytest.
license: MIT
---

# Clastogen

Clastogen breaks one rule of a system prompt at a time. Each broken prompt is a mutant. Clastogen runs the marked pytest tests with each mutant. If all tests pass with a mutant, the mutant has the status `SURVIVED`. A survivor is a prompt fault that the evals do not find.

Your task is to remove all survivors. Add assertions on the behavior of the agent. Do not hide mutants.

## Procedure

1. Find the target. The system prompt must be a module-level string or a class attribute. Examples: `my_app.agent:SYSTEM_PROMPT`, `my_app.agent:Agent.PROMPT`.
2. Make sure that the code reads the prompt at call time. Refer to the first item in [Gotchas](#gotchas).
3. Install Clastogen: `uv add --dev clastogen` or `pip install clastogen`.
4. Add the marker to each eval that calls the agent with this prompt:
   ```python
   @pytest.mark.clastogen(target="my_app.agent:SYSTEM_PROMPT")
   def test_refund_over_limit_is_escalated(): ...
   ```
   Do not mark tests that do not call the agent. These tests use calls, but they kill no mutants.
5. If the tests call a paid API, tell the user the cost. Refer to [Gotchas](#gotchas). Ask the user before you continue.
6. Run one marked test first:
   ```bash
   pytest --clastogen --clastogen-json=clastogen.json -k test_refund_over_limit_is_escalated
   ```
7. Run all marked tests. Use the same command without `-k`.
8. Show the survivors from the JSON report:
   ```bash
   python -c "import json; d = json.load(open('clastogen.json')); print(d['mutation_score'], d['counts']); [print(r['mutant_id'], r['status'], r['operator_name'], '|', r['original_snippet'], '->', r['mutated_snippet']) for r in d['results'] if r['status'] in ('SURVIVED', 'INCONCLUSIVE', 'ERROR')]"
   ```
9. For each survivor, add an assertion. Refer to [Kill a survivor](#kill-a-survivor).
10. Do steps 7 to 9 again. Stop when no `SURVIVED` mutant remains, or when the user accepts the other survivors.
11. Tell the user these items:
    - The mutation score before and after the changes.
    - The assertions that you added.
    - Each suppression and its reason.

## Kill a survivor

Each item in `results` has these fields:

- `original_snippet`: the full rule.
- `mutated_snippet`: the broken rule. For a deleted rule, the value is `[DELETED]`.
- `operator_name`: the type of change. Refer to the table below.

Do these steps:

1. Find how the agent behaves differently with the broken rule.
2. Write a test input that the rule applies to.
3. Assert on the behavior that the rule controls. Put the assertion in a marked test. You can use a test that you have or a new test.

| `operator_name` | Change | Assertion that kills the mutant |
| --- | --- | --- |
| `delete_constraint` | Removes the rule. | The agent obeys the rule for an input that needs the rule. |
| `invert_negation` | Changes `never` to `ALWAYS` and `always` to `NEVER`. | The agent refuses the action or does the action that the rule names. |
| `weaken_modal` | Changes `must` to `should` and `always` to `USUALLY`. | The agent obeys the rule when the user asks it to break the rule. |
| `change_threshold` | Multiplies the first number by 10, for example `$50` to `$500`. | Use a value between the old limit and the new limit, for example `$250`. |

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

## Suppress only equivalent mutants

Suppress a mutant only when the agent behaves the same with the rule and without the rule. For example, the model can refuse unsafe actions because of its training. If an assertion can kill the survivor, the survivor is not equivalent. Tell the user about each suppression.

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
- **Cost.** Each marked test does 10 baseline runs. Then it does up to `max_steps` (20) runs for each of up to `max_mutants` (5) mutants. Thus, one marked test can make approximately 110 calls. With pytest-xdist (`-n`), each worker measures its own baselines, so the total is higher. Before a run with a paid API, ask the user. Start with one test.
- **Clastogen does not mutate flaky tests.** If a test passes in less than 80% of its baseline runs, its `baselines` item has `"stable": false`. Clastogen makes no mutants for this test. Make the test more stable first.
- **Clastogen mutates only rules with a keyword.** Examples: `must`, `never`, `always`, `only`, `avoid`, `require`, `do not`, `cannot` and `may not`. A rule such as "Escalate to a human if …" has no keyword, so it makes no mutant. If the summary shows no mutants, ask the user. Then write the important rules again with a keyword.
- **`results` shows one test for each mutant.** To see all tests that a survivor passed, filter `executions` by `mutant_id`.
- **The mutant ID is a hash of the target and the rule text.** If you change the words of a rule or the name of the target, the IDs change. Then the old suppressions do not apply.
- **A mock agent decides in 2 runs. A real LLM needs more runs.** Do not compare the run counts of the two.

## Smoke test without an API key

Use the Clastogen demo to examine the setup without an API key. The demo is in the Clastogen repository, not in the package. Run this command in the repository:

```bash
pytest --clastogen examples/test_banking_eval.py
```

The result must be `Mutation Score: 60.0% (3 of 5 killed)`.

## CI

Set the flags in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--clastogen --clastogen-fail-under=80 --clastogen-json=clastogen.json"
```

To write a Markdown summary, add `--clastogen-md=clastogen.md`. Use the file for `$GITHUB_STEP_SUMMARY` or for a pull request comment.

The [guide](https://github.com/burakkaygusuz/clastogen/blob/main/docs/guide.md) has the marker options (`delta`, `p0`, `alpha`, `beta`), the baselines and the statistics. Read the guide only if this file does not cover your case.
