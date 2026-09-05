# ADR 0003 — Hash-chain the audit log

**Status:** accepted · 2026-09-03

## Context

The platform makes decisions with consequences — refusing a request, spending money,
calling a tool that changes a system of record. An audit trail that can be edited after the
fact is a liability rather than a control: it provides the appearance of accountability
without the substance.

## Decision

Each record stores the SHA-256 of its own canonical content combined with the hash of the
record before it. Verification recomputes the whole chain.

The canonical form has sorted keys and fixed separators, so two processes hashing the same
logical record always agree.

## Alternatives rejected

**Trust the storage layer.** Append-only at the database level protects against the
application, not against someone with database access — usually the same person you most
need the trail to constrain.

**Sign each record.** Stronger, and it needs a key, a rotation story and an HSM. The chain
gets the tamper-evidence property with no key management, and signing can be layered on
later without changing the record format.

**Ship to a SIEM and rely on it.** Correct, and orthogonal: it protects records once they
arrive. This protects them before, during and after.

## Consequences

- Editing, deleting or reordering any record breaks verification from that point onward,
  and `/v1/audit/verify` names the sequence number where it broke.
- The chain resumes across restarts by reading its own head.
- Records must be written in order by a single writer per chain. Horizontal scale needs
  either a chain per instance, reconciled later, or a single writer — an accepted
  limitation, and the reason the sink is an interface.
- **It proves integrity, not availability.** Someone with write access to the sink can
  still truncate it. Detecting truncation needs WORM retention and off-cluster export of
  the head hash; the platform provides the hash, the deployment provides the policy.