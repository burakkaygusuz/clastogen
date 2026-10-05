# Clastogen Domain Model & Architecture Context

This document records the foundational domain terms, seams, and invariants of **Clastogen**, an LLM and Agent mutation testing and statistical assertion framework.

## Core Domain Terms

### 1. Mutant

A content-addressed, defective variant of an agent prompt, model constraint, or tool definition.

- **Identifier (`id`)**: 12-character SHA-256 hash computed deterministically from `(target_symbol, operator_name, original_snippet, mutated_snippet)`.
- **Invariance**: An identical modification to the same target always yields the exact same `Mutant.id`, enabling cross-test deduplication and persistent suppression tracking (`.clastogen/suppressions.toml`).

### 2. Sequential Probability Ratio Test (SPRT)

Wald's sequential hypothesis testing algorithm used to evaluate non-deterministic LLM behavior.

- **Null Hypothesis ($H_0$)**: The mutant survives; the agent satisfies the requirement with pass rate $p_0$.
- **Early Stopping**: Dynamically decides `KILLED` or `SURVIVED` based on Wald's sequential boundaries. Severe defects are halted in as few as 3 trials, while borderline cases and flaky baselines cap at $N_{\max}$ (ASN ~7–18 depending on baseline stability) and report `INCONCLUSIVE` transparently in the indifference zone.

### 3. Statistical Assertion

A probabilistic test assertion evaluating Bernoulli trial sequences.

- Calculates **Wilson score confidence intervals** ($[w^-, w^+]$) and exact binomial probabilities using the standard library.
- Distinguishes genuine prompt regressions from stochastic test flakiness.

### 4. Cassette & ReplayEngine _(Planned: v0.3)_

A deterministic record/replay module storing serialized prompt-response pairs and tool invocations.

- Replays recorded responses during local and CI regression runs to guarantee reproducibility and eliminate API token costs.
- Freezes tool outputs while allowing mutant reasoning traces to be verified.

### 5. ToolMutator _(Planned: v0.3)_

An AST and schema mutation operator that targets agent tool definitions (e.g. Model Context Protocol / MCP tool specifications, JSON Schemas, function signatures).

- Injects tool description omissions, parameter type shifts, or ambiguous instructions to test agent fault tolerance.

### 6. LLMAdapter _(Planned: v0.3)_

A deep module providing a uniform `complete()` interface over disparate LLM backends (LiteLLM, direct HTTPX, Ollama).

- Encapsulates provider dispatch, exponential backoff retries (`tenacity`), and GenAI semantic convention traces (`opentelemetry`).

### 7. FrameworkAdapter _(Planned: v0.3)_

A seam adapting external agent runners (LangChain, LangGraph, CrewAI, promptfoo, OpenAI Agents SDK) to Clastogen's mutation evaluation lifecycle.
