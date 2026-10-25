# Clastogen

![Clastogen logo](https://raw.githubusercontent.com/burakkaygusuz/clastogen/main/assets/logo.svg)

**Your LLM evals pass. Can they catch a broken prompt?**

[![PyPI](https://img.shields.io/pypi/v/clastogen)](https://pypi.org/project/clastogen/)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/burakkaygusuz/clastogen/badge)](https://scorecard.dev/viewer/?uri=github.com/burakkaygusuz/clastogen)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15258/badge)](https://www.bestpractices.dev/projects/15258)

Clastogen breaks your system prompt on purpose and checks if your existing pytest evals notice.

| Your prompt says                             | Clastogen changes it to                       |
| -------------------------------------------- | --------------------------------------------- |
| You must **never** approve refunds over $50  | You must **ALWAYS** approve refunds over $50  |
| You must **always** verify customer identity | You must **USUALLY** verify customer identity |
| ... refunds over **$50**                     | ... refunds over **$500**                     |

If your tests still pass, you found a blind spot.

**On a real agent** (a 14-rule support-desk prompt, a 12-test eval suite, DeepSeek v4.1 flash), the evals killed only **27.1%** of the mutants (clastogen 0.8.0, 5 runs, measured before the alpha split; with 12 tests on one prompt a rerun would give different numbers). An independent check (its probes were revised once after a first false kill) confirmed 3 blind spots, such as a "Do not create discount codes" rule inverted to "DO create …" with every test still green, and found no false kill in the final labelling. [Case study →](https://github.com/burakkaygusuz/clastogen/blob/main/benchmarks/support_desk/README.md)

![Clastogen demo](https://raw.githubusercontent.com/burakkaygusuz/clastogen/main/assets/demo.gif)

**Demo result: Mutation Score 60%.** The evals caught 3 of 5 prompt faults. 2 blind spots found.

```bash
pip install clastogen
pytest --clastogen
```

---

## Why does this matter?

More evals are not enough. Your evals must catch prompt regressions.

A weak eval passes with a good prompt and with a broken prompt. Your CI stays green, and your agent approves a $500 refund. Clastogen finds these weak evals before your users do.

## How it works

1. Clastogen finds the rules in your system prompt: sentences with a word such as `must`, `never`, `always`, `only`, `if`, `unless`, `escalate` or `refuse`, or with a number. Role sentences ("You are …") are skipped.
2. It makes mutants. Each mutant has one fault: a rule is deleted, inverted ("never" → "ALWAYS"), weakened ("must" → "should"), its number is multiplied by 10, its "unless …" clause is dropped, its comparator is flipped ("at most" → "AT LEAST"), its scope is narrowed ("all" → "SOME"), or a whole section or `<example>` is deleted. Every rule gets one mutant before any rule gets a second, so a small `max_mutants` still spreads over the prompt.
3. It runs your marked tests with each mutant. If a test fails, the mutant is **killed**. If all tests pass, the mutant **survived**: that is a blind spot.
4. You get a mutation score: the percentage of mutants that your tests killed. Next to it, a per-operator kill rate and a constraint coverage show where the blind spots are, and which rules no measured mutant covers.

LLM output is random, so Clastogen runs each mutant more than one time. A sequential test (SPRT) stops as soon as the result is clear, so a clear kill costs only 2 calls (3 with two tests on the target, 5 with 50). With T tests on one target, a mutant is killed if any test kills it, so each test's SPRT runs at alpha / T (Bonferroni): for baselines of 0.90 or more the chance that an unchanged mutant is killed stays near 5% instead of growing with T (noisy baselines stay above it: [details](https://github.com/burakkaygusuz/clastogen/blob/main/docs/statistics.md#statistical-caveats)), at the price of more `INCONCLUSIVE` mutants at large T. Every run reports its calls against a fixed-N test with the same error rates. The error rates hold under assumptions: independent runs, a baseline p0 estimated from 10 runs, and a `max_steps` cutoff that can leave a mutant `INCONCLUSIVE`. [Statistical details →](https://github.com/burakkaygusuz/clastogen/blob/main/docs/statistics.md)

A survivor is not always a blind spot: the model can keep its behavior without a rule. Mark a few behavior checks as probes (`oracle_samples=30`) and run `pytest --clastogen --clastogen-oracle`. Clastogen then reports two axes, mutants your tests killed and how many of the mutants that change behavior they killed. [Probes →](https://github.com/burakkaygusuz/clastogen/blob/main/docs/guide.md#probes-which-survivors-matter)

## Works with your existing evals

Clastogen does not replace your evals. It tests if your evals are strong enough to find a broken prompt. Any evaluator that reports pass or fail through a pytest test works with Clastogen:

- ✓ Plain `assert` statements
- ✓ DeepEval metrics with `assert_test` ([example](https://github.com/burakkaygusuz/clastogen/blob/main/examples/test_deepeval_eval.py))
- ✓ Ragas, LLM-as-judge or your own evaluator, called in a pytest test

---

## Quickstart

1. Install Clastogen:

   ```bash
   pip install clastogen
   # or
   uv add --dev clastogen
   ```

2. Add the `clastogen` marker to an eval test. Set `target` to the variable that contains your system prompt:

   ```python
   import pytest


   @pytest.mark.clastogen(target="my_app.agent:SYSTEM_PROMPT")
   def test_agent_behavior():
       response = call_agent("transfer $500")
       assert "Verification code" in response
   ```

3. Run your tests with the `--clastogen` flag:

   ```bash
   pytest --clastogen
   ```

> **Important:** Clastogen replaces the prompt attribute at call time, and, for httpx- or httpx2-based SDKs (tested with OpenAI and Anthropic; unsigned requests, except streamed request bodies), also rewrites the prompt text in the outgoing request. With any other client, if your code copies the prompt at import time, the survivors become `ERROR` (the canary). See [Read the prompt at call time](https://github.com/burakkaygusuz/clastogen/blob/main/docs/guide.md#read-the-prompt-at-call-time).

---

## Demo

This demo is a **mock** agent, so it needs no API key. For a real model, see [the case study below](#real-models). [`src/clastogen/demo/mock_agent.py`](https://github.com/burakkaygusuz/clastogen/blob/main/src/clastogen/demo/mock_agent.py) is a mock banking agent that obeys only the rules in its system prompt. [`src/clastogen/demo/test_banking_eval.py`](https://github.com/burakkaygusuz/clastogen/blob/main/src/clastogen/demo/test_banking_eval.py) has a strong test that checks identity verification, and a weak test that only checks that the response contains the word "refund".

```bash
uvx --with clastogen pytest --pyargs clastogen.demo --clastogen --clastogen-html=reports/report.html
```

Both tests pass in a usual run. With Clastogen:

- **Mutation Score: 60.0%**
- 3 of 5 prompt faults caught
- 2 eval blind spots found: "never" → "ALWAYS" approve refunds, and the $50 limit → $500

The terminal output:

```text
====================== Clastogen Mutation Testing Summary ======================
✗ SURVIVED      [fef2fa3ee004]  6 runs  passed in test_identity_verification_is_enforced
    Changed threshold: '50' -> '500' in 'You must never approve refund requests exceeding…'
    Assert: Use a value between the old limit and the new limit, for example `$250`.
✗ SURVIVED      [bb49e3ca17a0]  6 runs  passed in test_identity_verification_is_enforced
    Inverted constraint: 'You must never approve refund requests…' -> 'You must ALWAYS approve refund requests…'
    Assert: The agent refuses the action or does the action that the rule names.
✓ KILLED        [398de66fceda]  3 runs  by test_identity_verification_is_enforced
    Weakened constraint: 'You must always verify customer identity…' -> 'You must USUALLY verify customer identity…'
✓ KILLED        [dc9dbcc841a9]  3 runs  by test_identity_verification_is_enforced
    Inverted constraint: 'You must always verify customer identity…' -> 'You must NEVER verify customer identity…'
✓ KILLED        [de806cb80393]  3 runs  by test_identity_verification_is_enforced
    Deleted constraint: 'You must always verify customer identity before providing…'
--------------------------------------------------------------------------------
Baselines: 2 stable tests, 10/10 runs passed (-v shows each test)
2 survived: your tests still pass with these prompt faults.
    Add an assertion that fails for each one, or suppress its ID in .clastogen/suppressions.toml.
Mutation Score: 60.0% (3 of 5 killed); Wilson 95% CI 23.1%-88.2%
Calls: 21 (a fixed-N test with the same error rates needs 95: 78% fewer)
By operator: change_threshold 0/1 (0%), delete_constraint 1/1 (100%), invert_negation 1/2 (50%), weaken_modal 1/1 (100%)
Constraint Coverage: 1 of 2 constraints killed (50.0%); 1 survived or inconclusive, 0 untested
Rules, weakest first (A mutant is killed if any test kills it; a rule's mutants are those that were measured; the test named is the lowest test ID among the killers.)
    survived: 0 of 2 mutants killed  You must never approve refund requests exceeding $50 withou…
    killed: 3 of 3 mutants killed by test_identity_verification_is_enforced  You must always verify customer identity before providing b…
Unmutated sentences (no rule keyword or digit, so no rule-level mutant):
    Be professional and concise
```

The `Rules` rows show, for each rule, how many of its measured mutants were killed and which test killed them (the lowest test ID among the killers). A mutant is killed if any test kills it. The `Unmutated sentences` are the sentences that rule discovery skips (no obligation word and no digit, outside `<example>` blocks), so no rule-level mutant covers them; a section or example mutant can still remove them.

The `Calls` line compares this run with a fixed-N test that has the same error rates. Here the mock agent gives clean kills, so the saving is large. [Simulated averages](https://github.com/burakkaygusuz/clastogen/blob/main/docs/statistics.md#savings-against-a-fixed-sample-size) are 48–56%.

The same results go to `reports/report.html`:

![Clastogen HTML report](https://raw.githubusercontent.com/burakkaygusuz/clastogen/main/assets/report.png)

### Real models

The [support-desk case study](https://github.com/burakkaygusuz/clastogen/blob/main/benchmarks/support_desk/README.md) tests a 14-rule e-commerce support agent on DeepSeek with a 12-test suite, and checks every verdict against an independent set of behavior probes. The suite scored 27.1% (5 runs, clastogen 0.8.0). None of the 9 kills was false, 3 survivors were confirmed blind spots (no test for the language, refund-count or discount rules), and 11 survivors were mutants that do not change this model's behavior. 4 mutants were too close to the threshold to call, and 7 mutants (of 3 rules) have no probe.

---

## CI

Set the flags one time in `pyproject.toml` with the pytest `addopts` option:

```toml
[tool.pytest.ini_options]
addopts = "--clastogen --clastogen-fail-under=80 --clastogen-json=clastogen.json"
```

The run fails if the mutation score is less than the threshold. A GitHub Actions job does not need special setup for Clastogen. To show the result in the job summary and in a pull request comment, write a Markdown file:

```yaml
- run: uv run pytest --clastogen --clastogen-md=clastogen.md
- if: always()
  run: cat clastogen.md >> "$GITHUB_STEP_SUMMARY"
- if: always() && github.event_name == 'pull_request'
  uses: marocchino/sticky-pull-request-comment@v2
  with:
    header: clastogen
    path: clastogen.md
```

The comment step needs `permissions: pull-requests: write`. Pull requests from forks get a read-only token, so the comment step fails there; the job summary still works.

The pytest `--junitxml` file also contains the Clastogen records of each marked test in the `clastogen_records` property. If a tool must parse the records, use `--clastogen-json`.

---

## Documentation

- [Guide](https://github.com/burakkaygusuz/clastogen/blob/main/docs/guide.md): report statuses and flags, marker options, prompt patching, pytest-xdist, suppressions, custom mutants for rules that keyword discovery misses (including prompts in other languages), known limitations.
- [Statistics](https://github.com/burakkaygusuz/clastogen/blob/main/docs/statistics.md): statistical assertions API (`assert_pass_rate`, `assert_no_regression`), SPRT simulations and guarantees.
- [Case study](https://github.com/burakkaygusuz/clastogen/blob/main/benchmarks/support_desk/README.md): a realistic support agent, a 12-test suite and an independent oracle that tells blind spots from equivalent mutants.
- [Agent skill](https://github.com/burakkaygusuz/clastogen/blob/main/skills/clastogen/SKILL.md): lets a coding agent (Claude Code, Codex, Cursor) mark your evals, run Clastogen and fix the survivors. Copy `skills/clastogen/` into your agent's skills directory.
- [Contributing](https://github.com/burakkaygusuz/clastogen/blob/main/CONTRIBUTING.md)

## License

MIT License. See [LICENSE](https://github.com/burakkaygusuz/clastogen/blob/main/LICENSE).
