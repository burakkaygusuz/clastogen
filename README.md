# Clastogen

![Clastogen logo](https://raw.githubusercontent.com/burakkaygusuz/clastogen/main/assets/logo.svg)

**Your LLM evals pass. Can they catch a broken prompt?**

[![PyPI](https://img.shields.io/pypi/v/clastogen)](https://pypi.org/project/clastogen/)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/burakkaygusuz/clastogen/badge)](https://scorecard.dev/viewer/?uri=github.com/burakkaygusuz/clastogen)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15258/badge)](https://www.bestpractices.dev/projects/15258)

Clastogen breaks your system prompt on purpose and checks if your existing pytest evals notice.

| Your prompt says | Clastogen changes it to |
| --- | --- |
| You must **never** approve refunds over $50 | You must **ALWAYS** approve refunds over $50 |
| You must **always** verify customer identity | You must **USUALLY** verify customer identity |
| ... refunds over **$50** | ... refunds over **$500** |

If your tests still pass, you found a blind spot.

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

1. Clastogen finds the rules in your system prompt: sentences with words such as `must`, `never`, `always` or `only`.
2. It makes mutants. Each mutant has one fault: a rule is deleted, inverted ("never" → "ALWAYS"), weakened ("must" → "should"), or its number is multiplied by 10.
3. It runs your marked tests with each mutant. If a test fails, the mutant is **killed**. If all tests pass, the mutant **survived**: that is a blind spot.
4. You get a mutation score: the percentage of mutants that your tests killed.

LLM output is random, so Clastogen runs each mutant more than one time. A sequential test (SPRT) stops as soon as the result is clear, so a clear kill costs only 2 calls. [Statistical details →](https://github.com/burakkaygusuz/clastogen/blob/main/docs/statistics.md)

## Works with your existing evals

Clastogen does not replace your evals. It tests if your evals are strong enough to find a broken prompt. Any evaluator that reports pass or fail through a pytest test works with Clastogen:

- ✓ Plain `assert` statements
- ✓ DeepEval metrics with `assert_test` ([example](https://github.com/burakkaygusuz/clastogen/blob/main/examples/test_deepeval_eval.py))
- ✓ Ragas, LLM-as-judge or your own evaluator, called in a pytest test

| Tool | The question that it answers |
| --- | --- |
| promptfoo | Which prompt and model give better output? Is my app safe from attacks? |
| DeepEval, Ragas | Is the output relevant, correct and faithful to the context? |
| Inspect AI | How well does a model or agent do a task? |
| muteval | Does my eval suite catch a degraded prompt, RAG context, tool output or model? |
| **Clastogen** | **If my system prompt breaks, do my tests fail?** |

muteval is the closest alternative. Clastogen focuses on system-prompt mutations and runs directly inside pytest.

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

> **Important:** Clastogen replaces the prompt at call time. If your code copies the prompt at import time, all mutants survive. See [Read the prompt at call time](https://github.com/burakkaygusuz/clastogen/blob/main/docs/guide.md#read-the-prompt-at-call-time).

---

## Demo

You do not need an API key for this demo. [`examples/mock_agent.py`](https://github.com/burakkaygusuz/clastogen/blob/main/examples/mock_agent.py) is a mock banking agent that obeys only the rules in its system prompt. [`examples/test_banking_eval.py`](https://github.com/burakkaygusuz/clastogen/blob/main/examples/test_banking_eval.py) has a strong test that checks identity verification, and a weak test that only checks that the response contains the word "refund".

```bash
uv run pytest --clastogen --clastogen-html=reports/report.html examples/test_banking_eval.py
```

Both tests pass in a usual run. With Clastogen:

- **Mutation Score: 60.0%**
- 3 of 5 prompt faults caught
- 2 eval blind spots found: "never" → "ALWAYS" approve refunds, and the $50 limit → $500

The terminal output:

```text
====================== Clastogen Mutation Testing Summary ======================
✗ SURVIVED      [7d1281020f96]  6 runs  passed in test_identity_verification_is_enforced, test_refund_handling_superficial_eval
    Inverted constraint: 'You must never approve refund requests…' -> 'You must ALWAYS approve refund requests…'
✗ SURVIVED      [add892bdcea2]  6 runs  passed in test_identity_verification_is_enforced, test_refund_handling_superficial_eval
    Changed threshold: '50' -> '500' in 'You must never approve refund requests exceeding…'
✓ KILLED        [45e77dddc073]  2 runs  by test_identity_verification_is_enforced
    Inverted constraint: 'You must always verify customer identity…' -> 'You must NEVER verify customer identity…'
✓ KILLED        [55063f25e7dd]  2 runs  by test_identity_verification_is_enforced
    Deleted constraint: 'You must always verify customer identity before providing…'
✓ KILLED        [7e685fba7c3e]  2 runs  by test_identity_verification_is_enforced
    Weakened constraint: 'You must always verify customer identity…' -> 'You must USUALLY verify customer identity…'
--------------------------------------------------------------------------------
2 survived: your tests still pass with these prompt faults.
Mutation Score: 60.0% (3 of 5 killed)
```

The same results go to `reports/report.html`:

![Clastogen HTML report](https://raw.githubusercontent.com/burakkaygusuz/clastogen/main/assets/report.png)

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

- [Guide](https://github.com/burakkaygusuz/clastogen/blob/main/docs/guide.md): report statuses and flags, marker options, prompt patching, pytest-xdist, suppressions, known limitations.
- [Statistics](https://github.com/burakkaygusuz/clastogen/blob/main/docs/statistics.md): statistical assertions API (`assert_pass_rate`, `assert_no_regression`), SPRT simulations and guarantees.
- [Agent skill](https://github.com/burakkaygusuz/clastogen/blob/main/skills/clastogen/SKILL.md): lets a coding agent (Claude Code, Codex, Cursor) mark your evals, run Clastogen and fix the survivors. Copy `skills/clastogen/` into your agent's skills directory.
- [Contributing](https://github.com/burakkaygusuz/clastogen/blob/main/CONTRIBUTING.md)

## License

MIT License. See [LICENSE](https://github.com/burakkaygusuz/clastogen/blob/main/LICENSE).
