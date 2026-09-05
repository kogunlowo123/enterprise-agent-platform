# ADR 0002 — Effective authority is the intersection of caller and agent grants

**Status:** accepted · 2026-09-03

## Context

An agent acts on a user's behalf. Two identities are involved, and the platform must decide
what the combination is permitted to do.

The obvious readings are both wrong:

- **Caller's permissions alone.** An agent designed to answer questions could then delete
  production data whenever an administrator happens to use it.
- **Agent's permissions alone.** The agent becomes a privilege-escalation device: any user
  who can invoke it inherits whatever it was granted.

## Decision

```
effective = caller_grants ∩ agent_grants
```

An agent can never widen its caller's reach, and a privileged caller cannot hand an agent
more authority than it was designed to hold.

`AgentIdentity.forbidden_tools` is enforced *above* this: a tool on that list is refused
even when the intersection would permit it, because the constraint belongs to the agent's
design rather than the caller's authority.

## Implementation note

Plain set intersection is wrong, and this shipped broken until a test caught it. A caller
holding `knowledge:read` and an agent granted `knowledge:*` intersect to the empty set
literally, while the correct answer is `knowledge:read`. The intersection is computed with
wildcard awareness: each side is kept where the other side covers it under the matching
rules.

## Consequences

- Refusals distinguish "you lack this" from "the agent lacks this", which is a materially
  more useful error.
- An agent's manifest only advertises tools that will actually succeed, so the model is not
  taught to attempt things that get refused.
- Cost: two grants to reason about instead of one, and the counter-intuitive result that
  granting an agent `*` gives it nothing beyond its caller.