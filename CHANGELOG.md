# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-05

First release. The platform runs an agent turn end to end with identity, governance,
knowledge, model routing and runtime in place.

### Added

**Identity** — OIDC/JWKS verification with key-rotation handling; issuer, audience and
expiry enforcement; a required tenant claim; role inheritance with cycle tolerance;
wildcard permission matching; caller ∩ agent authority intersection.

**SecOps** — hash-chained tamper-evident audit log with HTTP verification; prompt-injection
detection across ten OWASP LLM01 pattern families with boundary weighting and Unicode
obfuscation defeat; credential and PII detection with Luhn and entropy validation;
deny-overrides policy engine with obligations.

**NetOps** — layered token-bucket rate limiting; three-state circuit breaker; full-jitter
retry; HTTPS egress allowlist with SSRF range blocking.

**DataOps** — structure-aware markdown and code chunking with citation provenance; hybrid
BM25 and vector retrieval fused by Reciprocal Rank Fusion; MMR diversification;
tenant-partitioned in-memory and pgvector stores with row-level security; GitHub connector
ingesting an enterprise repository at a pinned commit with rate-limit awareness and secret
quarantine.

**LLMOps** — model router with cross-vendor fallback, per-provider circuit breaking and
pre-dispatch budget enforcement; per-tenant cost attribution by model and agent; versioned
content-hashed prompt registry with rollback; deterministic evaluation harness gating CI on
groundedness, citation coverage, refusal correctness and injection resistance.

**AppOps** — the eight-stage agent lifecycle; JSON-Schema-validated tool dispatch under the
full permission chain with human-approval obligations; MCP gateway with tool namespacing,
conservative mutation inference and untrusted-output containment; memory scoped by tenant
and principal.

**API** — agent invocation, GitHub ingestion, retrieval-only search, audit verification,
cost attribution, policy inspection, and separate liveness and readiness probes. Failures
are RFC 9457 problem documents.

**Operations** — Apache-2.0 licence; multi-stage non-root container; Helm chart with
probes, NetworkPolicy, PDB and HPA; Terraform module for pgvector on RDS; CI across three
Python versions with an evaluation gate and a container smoke test; a security workflow
running pip-audit, Bandit, CodeQL, Trivy, Gitleaks and SBOM generation weekly.

### Fixed during development

Recorded because both shipped and were caught by tests rather than by review:

- **Cross-tenant leak in the lexical index.** The vector store was correctly partitioned by
  tenant, so a shared BM25 index leaked only in hybrid mode — fusion placed another
  tenant's document into the model's context. The index is now partitioned per tenant, with
  a second filter in the retriever as defence in depth.
- **`fnmatch` does not implement `**`.** The GitHub connector's default include glob
  `**/*` silently excluded every file at repository root, so ingestion reported success
  while skipping the README. Replaced with a real path-glob matcher.
- **Wildcard permissions intersected to nothing.** `knowledge:read` ∩ `knowledge:*` is
  empty under set semantics; the intersection is now computed with wildcard awareness.
- **Evaluation gate failed on unmeasured thresholds.** A declared threshold that no case
  exercised scored zero and failed the suite. Unmeasured thresholds are now reported by
  name rather than counted as breaches.
- **Citation coverage scored correctly-cited answers as zero.** Sentence splitting
  separated `"... UTC. [1]"` into a bare citation fragment plus an apparently uncited
  sentence.

[0.1.0]: https://github.com/kogunlowo123/enterprise-agent-platform/releases/tag/v0.1.0