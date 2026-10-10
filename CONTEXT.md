# Clastogen Domain Model & Architecture Context

This document records the foundational domain terms, seams, and invariants of **Clastogen**, an LLM and Agent mutation testing and statistical assertion framework.

## Core Domain Terms

### 1. Mutant

A content-addressed, defective variant of an agent prompt.

- **Identifier (`id`)**: 12-character SHA-256 hash computed deterministically from `(target_symbol, operator_name, original_snippet, mutated_snippet)`.
- **Invariance**: An identical modification to the same target always yields the exact same `Mutant.id`, enabling cross-test deduplication and persistent suppression tracking (`.clastogen/suppressions.toml`).

### 2. Sequential Probability Ratio Test (SPRT)

Wald's sequential hypothesis testing algorithm used to evaluate non-deterministic LLM behavior.

- **Null Hypothesis ($H_0$)**: The mutant survives; the agent satisfies the requirement with pass rate $p_0$.
- **Early Stopping**: Dynamically decides `KILLED` or `SURVIVED` based on Wald's sequential boundaries. Severe defects are halted in as few as 2 trials (3 when p0 is set to 0.90), while borderline cases and flaky baselines cap at $N_{\max}$ (ASN ~7–18 depending on baseline stability) and report `INCONCLUSIVE` transparently in the indifference zone.

### 3. Statistical Assertion

A probabilistic test assertion evaluating Bernoulli trial sequences.

- Calculates **Wilson score confidence intervals** ($[w^-, w^+]$) and exact binomial probabilities using the standard library.
- Distinguishes genuine prompt regressions from stochastic test flakiness.
