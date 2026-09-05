"""Ambient request context.

A single agent turn touches the gateway, the policy engine, the retriever, two model
providers and an audit sink. Threading a correlation id and a tenant id through every one
of those call signatures would be noise, and forgetting to thread it is how audit trails
develop holes. It lives in a :class:`~contextvars.ContextVar` instead, set once at the
edge and read wherever it is needed.

The context is read-only once bound. Mutating it mid-request would let a downstream
component quietly change which tenant an action is attributed to.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from eap.identity.models import Principal

_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford base32, no look-alike glyphs


def new_id(prefix: str, *, length: int = 16) -> str:
    """Generate a sortable-enough, URL-safe, prefixed identifier.

    Prefixed because an id that leaks into a log or a bug report should say what it is:
    ``run_7h2k...`` is self-describing where a bare UUID is not.
    """
    body = "".join(secrets.choice(_ALPHABET) for _ in range(length))
    return f"{prefix}_{body}"


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything ambient about the work currently in flight."""

    correlation_id: str
    principal: Principal | None = None
    run_id: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def tenant_id(self) -> str | None:
        return self.principal.tenant_id if self.principal else None

    def with_principal(self, principal: Principal) -> RequestContext:
        return RequestContext(
            correlation_id=self.correlation_id,
            principal=principal,
            run_id=self.run_id,
            attributes=dict(self.attributes),
        )

    def with_run(self, run_id: str) -> RequestContext:
        return RequestContext(
            correlation_id=self.correlation_id,
            principal=self.principal,
            run_id=run_id,
            attributes=dict(self.attributes),
        )

    def log_fields(self) -> dict[str, str]:
        """The subset that belongs on every log line."""
        fields = {"correlation_id": self.correlation_id}
        if self.run_id:
            fields["run_id"] = self.run_id
        if self.principal:
            fields["tenant_id"] = self.principal.tenant_id
            fields["principal_id"] = self.principal.subject
        return fields | self.attributes


_CURRENT: ContextVar[RequestContext | None] = ContextVar("eap_request_context", default=None)


def current_context() -> RequestContext | None:
    return _CURRENT.get()


def require_context() -> RequestContext:
    ctx = _CURRENT.get()
    if ctx is None:
        raise RuntimeError(
            "no request context bound; work that must be attributed to a caller has to run "
            "inside bind_context()"
        )
    return ctx


@contextmanager
def bind_context(ctx: RequestContext) -> Iterator[RequestContext]:
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)
