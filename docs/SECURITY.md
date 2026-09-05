# Security

## Reporting a vulnerability

Open a [private security advisory](https://github.com/kogunlowo123/enterprise-agent-platform/security/advisories/new).
Please do not open a public issue for anything exploitable.

This is a reference implementation, not a hosted service. There is no bug bounty, and no
SLA on a fix.

---

## Controls

### Identity and authorization

| Control | Where | Test |
|---|---|---|
| Asymmetric token verification against IdP JWKS, with rotation handling | `identity/tokens.py` | `test_identity.py::TestTokenVerification` |
| Issuer, audience and expiry all enforced; a missing tenant claim is rejected rather than defaulted | `identity/tokens.py` | ↑ |
| Effective authority = caller grants ∩ agent grants | `identity/models.py` | `TestPermissionIntersection` |
| Token scopes narrow role permissions, never widen them | `identity/rbac.py` | ↑ |
| Cross-tenant binding refused at context construction | `identity/rbac.py` | `TestTenantIsolation` |
| An agent's forbidden tools are refused even when the caller holds the permission | `appops/tools/registry.py` | `test_appops.py::TestDispatch` |

A defaulted tenant claim is a cross-tenant data leak waiting for the first misconfigured
client, which is why a token without one is rejected outright.

### Prompt injection

Ten pattern families drawn from OWASP LLM01, scored with probabilistic-OR aggregation so
four independent weak signals read as "almost certainly" rather than saturating on one.

Two properties matter more than raw accuracy:

**Boundary awareness.** `"ignore previous instructions"` typed into a chat box is usually a
user being sloppy about their own conversation. The same string arriving inside a retrieved
wiki page or a tool response is *indirect* injection — content no human chose to send.
Retrieved and tool boundaries carry a 1.6× multiplier, which is why authority-spoofing
clears the threshold from a document but not from a person.

**Obfuscation defeat.** Text is NFKC-normalised and stripped of zero-width and
bidirectional control characters before matching, so `Ｉｇｎｏｒｅ` and `i​gnore` both
reach the patterns as `ignore` — and the presence of those characters is itself a finding.

**Indirect injection is stopped at ingestion, not retrieval.** A poisoned document is a
payload that sits dormant in the vector store until a query happens to match it. Checking
at retrieval means the payload is already inside the trust boundary and the check must run
on every query forever. Checking at ingestion means it never gets in.

### Sensitive data

Credentials (AWS, GitHub, OpenAI, Anthropic, Slack, private keys, JWTs) are treated as
critical and stop the turn: a live key that reaches a model provider must be rotated
regardless of what the provider does with it.

Personal data (email, phone, national identifiers, payment cards) is **redacted rather than
blocked** — refusing a support conversation because it mentions a customer's email would
make the platform unusable, while replacing the value keeps it out of the provider's logs.

Detectors that can be checked arithmetically are: payment cards run through Luhn, so a
sixteen-digit order number is not flagged. Generic `secret = "..."` assignments are filtered
by Shannon entropy, so `password = "changemechangeme"` is not reported.

Audit evidence masks the value it reports (`AKI**************PLE`), so the log records the
hit without reproducing the secret.

### Tool and agent safety

The full chain before anything executes, in this order:

1. The tool exists.
2. The agent's design permits it (`forbidden_tools`).
3. The caller ∩ agent intersection covers `required_permission`.
4. Policy allows it, and any obligations are discharged.
5. Arguments validate against the declared JSON Schema.
6. Only then does anything run.

Permission is checked **before** argument validation so a denied caller cannot use
validation error messages to probe a tool's shape. Invented arguments are rejected rather
than dropped: a model that invents an argument has misunderstood the tool, and silently
dropping it turns an error into a wrong result.

Mutating tools attract a human-approval obligation, raised as an exception rather than
returned as a flag — ignoring a returned flag is one missing branch, ignoring a raised
exception takes deliberate effort.

An agent principal is refused any mutating tool, so chained delegation must terminate at a
human-authorised caller.

Tool arguments are recorded in the audit log **by key only**. Values routinely carry
customer data, and the audit log has longer retention than most data agreements allow.

### MCP

MCP servers are written by people who have never heard of this platform, so the gateway
treats them as untrusted by default:

- Tools are namespaced `server.tool`, so an audit record says which server ran.
- Permissions are inferred conservatively from the tool name and description. A server that
  adds `delete_everything` in a later release is refused by default rather than silently
  granted.
- An explicit `allow_tools` list means a server growing new tools imports nothing new until
  someone decides it should.
- **Untrusted server output passes through the guardrail pipeline** before reaching the
  model. This is the mitigation for tool poisoning, where a benign-looking server returns a
  payload aimed at the model rather than the user.
- Server subprocesses receive only the environment explicitly configured for them, so an
  MCP server cannot read the platform's model API keys.

### Network

- HTTPS-only egress against an allowlist. A denylist assumes you can enumerate every bad
  destination.
- Hostname resolution is checked against blocked ranges, so an allowlisted name resolving
  to `169.254.169.254`, loopback or RFC 1918 space is refused. This is the DNS-rebinding
  and SSRF mitigation.
- The Helm chart ships a matching `NetworkPolicy`. The application check is fast and
  explains itself in an audit record; the network policy holds when the application is
  compromised.

### Configuration

`Settings` refuses to start in staging or production with:

- HS256 development signing, or a leftover dev signing key
- An in-memory vector store or audit sink
- The deterministic embedder
- Injection enforcement disabled
- Prompt-content export enabled

Every problem is reported at once, so one failed deploy gives the full list. Secrets are
typed `SecretStr`, so a stray `repr()` in a log line or traceback prints `**********`.

### Supply chain

`pip-audit` on dependencies, Trivy on the container, Bandit and CodeQL on source, Gitleaks
across full history, and a CycloneDX SBOM published as a build artefact. The security
workflow also runs weekly, because a dependency that was clean at merge time does not stay
clean.

---

## What these controls do not do

Stating this plainly matters more than the control list, because a security posture nobody
can see the edges of is one nobody can reason about.

**Pattern matching does not stop a determined attacker.** The injection detector catches
unsophisticated and opportunistic attempts. It will not catch someone who paraphrases,
encodes, translates or splits a payload across documents. It is one layer, and the layers
carrying real weight are the permission intersection, least-privilege tool grants, human
approval on writes, and egress control — all of which hold when the detector misses, which
it will.

**Lexical groundedness is not entailment.** The metric measures word overlap between the
answer and its sources. It is a reliable floor — a score near zero means the answer has
essentially no relationship to its sources — but a high score is not proof of correctness,
and a correct paraphrase is marked down.

**Mutation inference is a heuristic.** A read-only tool named `run_query` is classified as
mutating and requires the write permission. That is the correct way to be wrong, but it is
still wrong.

**The audit chain proves integrity, not availability.** An attacker with write access to
the sink can truncate it. Detecting truncation requires WORM storage and off-cluster export
of the head hash; the platform provides the hash and the endpoint, the deployment provides
the retention policy.

**Rate limits and budgets are per-process.** Correct for one instance, wrong behind a load
balancer where N replicas each grant the full limit. Both take a pluggable store and Redis
is the production swap.

**Tenant isolation depends on the store.** In-memory partitioning is enforced in code; the
pgvector backend adds row-level security so that the application role physically cannot
read another tenant's rows even if a query is written wrongly. Defence in depth, because
the application-layer predicate is one typo from absent — and that typo happened during
development in the lexical index, which is documented in the README.

**No formal verification, no penetration test.** The controls are tested; they have not
been adversarially assessed by anyone other than the author.
