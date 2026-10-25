# Pre-registration template

Copy this file to `benchmarks/<study>/PREREGISTRATION.md` and fill it in. Commit it **before the first paid model call**.
A result is published only if its pre-registration commit is older than its first run file. Delete the hints in italics.

## Question

_One sentence, answerable with yes or no. One claim per pre-registration._

## Claim this supports

_The sentence of the README, a PR or a release note that this result backs. A feature with no claim needs no study._

## Material

- **Prompts**: _where each comes from (public URL and commit, or the owner), and its licence. Prompt texts that may not be redistributed stay out of the repo; give a hash instead._
- **Evals**: _who wrote them, from what source, and whether the author saw any mutant. Evals written by the person who wrote the mutants or probes are marked "not independent" in the result._
- **Oracle**: _the probes that judge each survivor (blind spot or equivalent), who wrote them, how many samples. A result without oracle labels is published as "unvalidated"._
- **Mutants**: _built-in operators, custom mutants (who wrote them), `max_mutants`._

## Setup

- **Model and provider pin**: _e.g. `openrouter/deepseek/deepseek-v4.1-flash`, `deepinfra/fp8`._
- **Clastogen commit**: _hash. The working tree is clean._
- **Settings that differ from the defaults** (`max_steps`, `delta`, `alpha`, `p0`): _list them, or "none"._
- **Runs**: _number of serial runs, and the oracle labelling count._

## Budget

- **Expected calls and cost**: _from `pytest --clastogen --collect-only`, with the price per call and where it comes from._
- **Hard ceiling**: _a number in dollars. The study stops there and reports what it has._

## Measures

_Each measure with its exact definition: the Mutation Score, the confusion table against the oracle, the kill share per mutant, and the denominators. Say what counts as borderline or unlabelled and that it stays out of the denominator._

## Decision rule

_A mechanical rule applied to the published summary files, for example "operator X is removed if ... in both runs". Say what each outcome means for the claim, including the outcome that refutes it. No criterion may be added after the first run._

## Validity checks

- Baselines are stable in every run (no rejected or noisy baseline), or the affected tests are named.
- Every mutant ID in the oracle appears in the runs, and the reverse.
- _Anything else that would make the result meaningless if it failed._

## Publication

- **Raw files**: _where `run_NN.*` and the oracle labels are stored (committed, or a release asset with a hash)._
- **Deviations**: any change after the first run goes in a dated "Deviations" section of the result with the reason, never in this file.
- **Failed and invalid runs** are listed in the result, with why they are invalid.
- **Limitations**: _one model, one provider, N prompts, who wrote the evals, and what the result cannot say._
