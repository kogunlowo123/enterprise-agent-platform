# ADR 0005 — Ship a deterministic provider instead of mocking the model

**Status:** accepted · 2026-09-03

## Context

Tests, CI and local development all need a model. Calling a real one makes the suite slow,
costly, network-dependent and non-reproducible — sampling variance alone means a passing
suite fails on rerun. Mocking it means the tests exercise the mock rather than the
platform's code paths.

## Decision

Implement the full `LLMProvider` contract with a provider whose output is a pure function
of its input. It selects context sentences covering the question by content-word overlap
with prefix-based stemming, and cites the numbered source they came from. It can be told to
fail a given number of times, which is how resilience paths get tested without breaking a
real provider on purpose.

`Settings` refuses to start with it in staging or production.

## Why this is not a mock

Routing, budgeting, guardrails, orchestration, audit and evaluation all execute exactly as
they do against a vendor. Nothing is patched or stubbed. And because the answer is a real
function of the retrieved context, an evaluation run against it measures *retrieval quality
and prompt assembly* — a genuinely useful signal, and one that cannot flake.

## Consequences

- The whole suite runs offline, in under two seconds, with no credentials.
- The evaluation gate can block a merge without spending money.
- It has no semantics: it cannot tell that "physician" and "doctor" are related, so
  semantic quality must be measured against a real model, separately and off the merge
  path.
- Its behaviour is a design surface. Restricting it to numbered source blocks, and making
  "no sources" a refusal, were both changes driven by evaluation results — recorded in
  docs/EVALUATION.md.