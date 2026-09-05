# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make check          # lint, format, types, tests — all of what CI runs
```

No credentials or network access are needed. The suite runs offline against the
deterministic provider and in-memory stores.

## Before opening a pull request

```bash
make format         # ruff format + autofix
make check          # must pass
make eval           # the evaluation gate must stay at 100%
```

CI runs the same commands on Python 3.11, 3.12 and 3.13, builds the container, starts it,
and confirms it becomes ready and is not running as root.

## What good looks like here

**Tests substitute, they do not mock.** The clock is injectable, so rate limiters and
circuit breakers are tested by advancing time rather than sleeping. The MCP tests spawn a
real server subprocess. The end-to-end tests drive the assembled application over HTTP. If
you find yourself reaching for `unittest.mock`, there is usually a seam missing.

**Test the behaviour, name the reason.** `test_a_refused_request_does_not_consume_tokens`
says what it protects; `test_rate_limiter_2` does not. A test name is the only documentation
that cannot go stale.

**Comments explain why, not what.** The code says what it does. A comment earns its place by
recording a decision, a constraint, or a trap — why full jitter rather than fixed backoff,
why the audit record is written before the action, why `fnmatch` was not good enough.

**New contracts are `Protocol`s.** Nothing inherits from a platform base class, so an
implementation can live in another package importing only the contract.

**Security-relevant changes need a test that fails without them.** A control with no test
demonstrating the thing it prevents is a comment.

**Say what a control does not do.** Every layer here has stated limits — see the second
half of `docs/SECURITY.md`. Adding a mitigation without stating its edge makes the posture
harder to reason about, not easier.

## Adding to the platform

| Adding | Implement | Register | Also |
|---|---|---|---|
| Model vendor | `LLMProvider` | `build_providers()` | Add prices to `DEFAULT_PRICES` |
| Knowledge source | `Connector` | Pass to `IngestionPipeline` | Provenance must pin a revision |
| Vector database | `VectorStore` | `build_store()` | Enforce tenancy by partition, not filter |
| Tool | `Tool`, or an MCP server | `ToolRegistry` / `MCPGateway` | Declare `mutates` honestly |
| Governance rule | `PolicyRule` | `PolicyEngine.add()` | Deny rules go before allow rules |
| Guardrail | `Guardrail` | `GuardrailPipeline` | Return findings; never raise |
| Audit destination | `AuditSink` | `build_audit_sink()` | Append-only; no update or delete |

## Evaluation cases

Add to `evals/grounding.jsonl`, one JSON object per line. A good case is one where you can
state, before running it, what the right answer is and why. If you cannot, it will measure
noise. Security cases belong in the same suite as quality cases — a separate job that
people skip when it is red defeats the purpose.

## Commits

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`,
`build:`, `ci:`. Explain *why* in the body when the change is not self-evident.

## Reporting a vulnerability

Use a [private security advisory](https://github.com/kogunlowo123/enterprise-agent-platform/security/advisories/new),
not a public issue.
