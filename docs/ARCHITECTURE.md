# Architecture

## The shape of the problem

An agent that works in a demo and an agent that works in an enterprise differ in ways that
have almost nothing to do with the model. The demo needs a prompt and an API key. The
enterprise version needs to answer *who is asking*, *on whose behalf*, *with what data*,
*at what cost*, *under which policy*, and *how would we prove any of that afterwards* —
and it needs to answer them on every single turn, not once at design time.

Those are six different questions with six different rates of change. The platform is
organised around them.

![Request lifecycle](architecture/agent-flow.svg)

## The six planes

Each plane owns one question and depends only on the planes below it in the diagram. The
dependency direction is enforced by the import graph: `identity` imports from `platform`
and nothing else; `secops` imports `identity`; `appops` sits at the top and imports from
everything. A cycle here would mean two planes are really one.

### Identity — who is acting

Two identities are in play on every turn, and conflating them is the root of most agent
security incidents.

The **principal** is the human or service that asked for the work, established by verifying
a bearer token against the identity provider's published keys. The **agent identity** is
what the runtime is permitted to do while carrying out that request.

The effective authority is the **intersection**:

```python
effective = caller_grants ∩ agent_grants
```

Not the union, and not either one alone. An agent cannot be used to widen its caller's
reach, and a highly-privileged caller cannot accidentally hand an agent more authority than
the agent was designed to hold. `AgentIdentity.forbidden_tools` is enforced even when the
caller holds the permission, because that constraint belongs to the agent's design rather
than to the caller's authority.

The intersection is computed with wildcard awareness rather than as a plain set operation.
A caller holding `knowledge:read` and an agent granted `knowledge:*` intersect to nothing
under set semantics, while the correct answer is `knowledge:read`.

### SecOps — should this happen at all

RBAC answers "may this caller do this kind of thing". Policy answers "given everything we
know about this specific request, should it happen anyway". Different shapes: permissions
are static grants, policies are contextual rules reading tenant residency, data
classification, the chosen model and the estimated cost.

Rules are **deny-overrides** and ordered, with the first matching deny winning. A
governance engine where adding a rule can *widen* access is one nobody can reason about.
Rules are data — `PolicyRule` objects an administrator can load from configuration — and
every decision returns the id of the rule that produced it, so a refusal traces back to a
specific line.

An allow can carry **obligations**: `require_human_approval`, `redact_pii`,
`disable_prompt_capture`. An allow with an undischarged obligation is a deny, and it is the
orchestrator's job to discharge them.

The **audit log** is a hash chain. Each record stores the SHA-256 of its canonical content
combined with the hash of the record before it, so editing, deleting or reordering any
record breaks verification from that point onward. This proves integrity, not availability:
an attacker with write access to the sink can still truncate it, which is why production
binds the sink to storage with an object-lock policy and exports the head hash off-cluster.

### NetOps — reachability and blast radius

Rate limiting is a **token bucket**, because it is the only common algorithm expressing
both a sustained rate and a burst allowance in two numbers a capacity planner can reason
about. Buckets are layered per principal *and* per tenant, and both are checked before
either is consumed, so a request rejected by the tenant limit does not silently burn the
principal's allowance.

Provider calls run under a **circuit breaker** that admits exactly one probe on recovery,
and retry with **full jitter** — `random(0, base × 2ⁿ)` rather than fixed exponential
backoff, because fixed backoff synchronises every client that failed at the same instant
into retrying at the same instant.

**Egress is allowlist-only.** A denylist assumes you can enumerate every bad destination.
An allowlisted hostname that resolves into a private, loopback or link-local range is still
refused, which is what stops DNS rebinding reaching the cloud metadata endpoint.

### DataOps — what the organisation knows

Chunking is **structure-aware**. Fixed-size chunking is the single largest source of bad
retrieval in RAG systems that otherwise look correct: it cuts tables in half, separates
headings from the paragraphs they introduce, and severs function signatures from bodies.
Markdown splits at heading boundaries and carries the heading path onto every chunk, which
is what makes a citation legible to the person checking the answer.

Retrieval is **hybrid**. Pure vector search is good at paraphrase and bad at rare exact
tokens — ask for `ERR_4021` and the embedding returns a cloud of similar-looking
identifiers. Pure lexical search has the mirror failure. Both run, and the rankings are
combined with **Reciprocal Rank Fusion**, which combines by *rank* rather than by score
because a BM25 score of 14.2 and a cosine similarity of 0.81 are not comparable quantities.

The fused list is diversified with **MMR**, because the top five results of a good
retriever are frequently five near-copies of the same paragraph.

Both the vector store and the lexical index are **partitioned by tenant** — not filtered
after the fact. A store that searches globally and drops foreign results leaks the moment
someone adds a code path that forgets the filter.

### LLMOps — which model, at what cost

Applications request a **route** (`fast`, `balanced`, `deep`), not a model. Swapping which
model serves `balanced`, or failing over to a second vendor during an outage, becomes a
configuration change rather than a deploy across every consuming service.

Each route is an ordered chain, and fallbacks **cross vendors deliberately** — a chain of
three models from one provider does nothing when that provider is the thing that is down.
A candidate is skipped without being attempted when its circuit is open or its estimated
cost would breach the tenant's budget.

Budget is checked before the call and recorded after it against the *actual* usage the
provider reported. Estimating and then trusting the estimate is how platforms discover a
20% accounting drift a month later.

Prompts are **versioned and content-hashed**. A prompt is production configuration with the
blast radius of code, routinely managed with neither the review nor the rollback that code
gets. Every response records the hash of the prompt that produced it.

### AppOps — how the turn runs

```
Receive → Resolve → Retrieve → Reason → Audit → Execute → Validate → Learn
```

The sequence is the security design as much as the control flow:

| Stage | What happens | Why here |
|---|---|---|
| Receive | Guardrails on the user's input | Before anything else touches it |
| Resolve | Authorization, policy, budget | Everything that can refuse cheaply refuses before a token is spent |
| Retrieve | Grounding material, tenant-scoped | The model sees sources, not the corpus |
| Reason | Model call through the router | With retrieved context and a filtered tool manifest |
| Audit | The decision is recorded | *Before* the action — an action audited afterwards is one whose record can be lost by the crash it caused |
| Execute | Tool dispatch | Under the full permission chain, with approval obligations enforced |
| Validate | Output guardrails, grounding check | Redaction, citation coverage, low-overlap warnings |
| Learn | Outcome to episodic memory | So a repeatedly failing tool call stops being retried |

The orchestrator controls the workflow. The model reasons *inside* it and never about it:
it cannot skip validation, widen its own permissions, or decide this turn does not need
auditing.

## The composition root

`bootstrap.py` builds every dependency and injects it downwards. No module below it reaches
for a global, which is why the whole tree is testable by substitution rather than by
patching — the test suite hands the same components a manual clock, a deterministic
provider and an in-memory store, and exercises the production code paths unchanged.

The provider set is assembled from whichever credentials are present, and routes are pruned
to candidates that are actually configured. A deployment with only an Anthropic key gets an
Anthropic-only router rather than a startup failure.

## Extension points

| To add | Implement | Register |
|---|---|---|
| A model vendor | `LLMProvider` | `build_providers()` |
| A knowledge source | `Connector` | Pass to `IngestionPipeline.ingest()` |
| A vector database | `VectorStore` | `build_store()` |
| An embedding model | `Embedder` | `build_embedder()` |
| A tool | `Tool`, or an MCP server | `ToolRegistry` / `MCPGateway.register_server()` |
| A governance rule | `PolicyRule` | `PolicyEngine.add()` |
| A guardrail | `Guardrail` | `GuardrailPipeline(guardrails=...)` |
| An audit destination | `AuditSink` | `build_audit_sink()` |

Every one of these is a `Protocol`, satisfied structurally. Nothing inherits from a
platform base class, so an implementation can live in another package without importing
anything but the contract.

## What is deliberately not here

- **A UI.** The platform is an API. A console belongs in a separate deployable.
- **Model fine-tuning.** Routing and retrieval solve more problems, more cheaply.
- **A workflow engine.** The orchestrator runs one turn. Multi-day processes belong in
  Temporal or Step Functions, calling this.
- **Distributed rate limiting and budgets.** The algorithms are correct; the state is
  in-process, which is right for one instance and wrong behind a load balancer. Both take a
  pluggable store, and Redis is the production swap. Stated rather than hidden.
