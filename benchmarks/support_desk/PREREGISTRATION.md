# Pre-registration: the `weaken_modal` decision rule

Written before the second and third model runs (B and C) so the outcome cannot shape the rule.

## Question

Does the `weaken_modal` operator produce mutants that are almost never distinguishable from the original prompt? If so, it is removed.

## Runs

- **A**: the existing case study (`results/deepseek_deepseek-v4.1-flash`). Reported separately; it does not count toward the decision.
- **B and C**: two further models, run on the same commit as A (`9e20fb196`) with the same provider pin and a clean working tree.

## Definitions

- **Labelled** = caught + blind spot + equivalent + false kill, counted over `weaken_modal` mutants only.
- **Equivalent** is probe-defined: no drop of 0.30 or more. It is model-dependent.
- Borderline and unlabelled mutants are not in the denominator.

## Rule

`weaken_modal` is removed if, in **both** B and C:

1. equivalent / labelled is at least 0.70,
2. blind spot is 0, and
3. labelled is at least 4.

Otherwise the operator stays. The rule is applied mechanically to the published `summary.md` files; no other criteria are added afterwards.
