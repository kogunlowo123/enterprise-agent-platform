# ADR 0001 — Organise the platform as six operational planes

**Status:** accepted · 2026-09-03

## Context

An agent platform accumulates concerns quickly: authentication, tenancy, guardrails,
policy, retrieval, embeddings, model selection, cost, tools, memory, tracing. Grouped by
technical layer (`services/`, `models/`, `utils/`) they become a flat namespace where
nothing says which module may depend on which, and within a quarter every module imports
every other one.

## Decision

Organise by the *question* each group answers, not by technical kind:

`platform` · `identity` · `secops` · `netops` · `dataops` · `llmops` · `appops`

Dependencies point downward only. `identity` imports `platform` and nothing else; `appops`
sits at the top and may import from anywhere below it.

## Alternatives rejected

**Layered (api / service / repository).** Says nothing about which concern owns what. The
guardrail pipeline and the vector store are both "service", which is not a useful fact.

**Feature modules (rag/, tools/, chat/).** Duplicates cross-cutting concerns: every feature
grows its own auth check, and they drift.

**A single package.** Viable at this size and indefensible at three times it. The boundary
is cheaper to draw now than to retrofit.

## Consequences

- A cyclic import between planes is a design error the interpreter reports for you.
- Ownership is obvious: injection detection is SecOps, chunking is DataOps.
- Mapping onto how enterprise teams are actually organised makes the repository navigable
  by people who did not write it.
- Cost: seven directories for what could be one, and occasional debate about where
  something belongs. Memory sits in AppOps rather than DataOps because it is per-agent
  state, not organisational knowledge — that distinction took a conversation.