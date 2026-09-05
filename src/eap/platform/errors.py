"""Error taxonomy.

Every failure the platform can produce is one of these. Each carries an HTTP status and
a stable machine-readable ``code`` so that callers can branch on the failure without
parsing prose, and so that dashboards can count failure classes rather than strings.

The taxonomy is deliberately small. A caller only needs to know four things: was this my
fault (4xx), was it a policy decision (403 with a reason), should I retry (429/503), or is
the platform broken (500).
"""

from __future__ import annotations

from typing import Any


class PlatformError(Exception):
    """Base class for every error the platform raises deliberately.

    Anything that is *not* a ``PlatformError`` escaping to the edge is a bug, and the edge
    handler reports it as an unattributed internal error without leaking the detail.
    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, /, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_problem(self, *, correlation_id: str | None = None) -> dict[str, Any]:
        """Render as an RFC 9457 problem document."""
        problem: dict[str, Any] = {
            "type": f"https://docs.eap.dev/errors/{self.code}",
            "title": self.code.replace("_", " "),
            "status": self.status_code,
            "detail": self.message,
        }
        if self.details:
            problem["details"] = self.details
        if correlation_id:
            problem["correlation_id"] = correlation_id
        return problem


class ConfigurationError(PlatformError):
    """The platform is misconfigured. Fails at startup, never at request time."""

    status_code = 500
    code = "configuration_error"


class ValidationError(PlatformError):
    status_code = 422
    code = "validation_error"


class AuthenticationError(PlatformError):
    """No usable credential was presented, or the credential did not verify."""

    status_code = 401
    code = "authentication_failed"


class AuthorizationError(PlatformError):
    """The caller is known but lacks the permission for this action."""

    status_code = 403
    code = "authorization_denied"

    def __init__(self, message: str, /, *, permission: str | None = None, **details: Any) -> None:
        if permission:
            details["required_permission"] = permission
        super().__init__(message, **details)


class TenantIsolationError(AuthorizationError):
    """A request tried to reach data belonging to a different tenant.

    Separated from the general authorization failure because this one is a security event,
    not a permission gap: it means something already went wrong upstream.
    """

    code = "tenant_isolation_violation"


class PolicyViolation(PlatformError):
    """A governance policy refused the action.

    Distinct from ``AuthorizationError``: the caller may well have the permission, but a
    rule (data residency, model allowlist, budget, content policy) says no anyway.
    """

    status_code = 403
    code = "policy_violation"

    def __init__(self, message: str, /, *, rule: str, **details: Any) -> None:
        super().__init__(message, rule=rule, **details)
        self.rule = rule


class GuardrailTripped(PolicyViolation):
    """Input or output was blocked by a guardrail (injection, PII, secret leakage)."""

    code = "guardrail_tripped"


class BudgetExceeded(PolicyViolation):
    """The tenant has spent its allocation for the current window."""

    status_code = 429
    code = "budget_exceeded"


class RateLimited(PlatformError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, /, *, retry_after_seconds: float, **details: Any) -> None:
        super().__init__(message, retry_after_seconds=round(retry_after_seconds, 3), **details)
        self.retry_after_seconds = retry_after_seconds


class NotFoundError(PlatformError):
    status_code = 404
    code = "not_found"


class ToolExecutionError(PlatformError):
    """A tool was invoked correctly but failed while running."""

    status_code = 502
    code = "tool_execution_failed"


class ProviderError(PlatformError):
    """An upstream model provider failed."""

    status_code = 502
    code = "provider_error"

    def __init__(self, message: str, /, *, provider: str, retryable: bool = True, **d: Any) -> None:
        super().__init__(message, provider=provider, retryable=retryable, **d)
        self.provider = provider
        self.retryable = retryable


class NoProviderAvailable(ProviderError):
    """Every candidate in the routing chain was exhausted."""

    status_code = 503
    code = "no_provider_available"

    def __init__(self, message: str, /, *, attempted: list[str]) -> None:
        PlatformError.__init__(self, message, attempted=attempted)
        self.provider = ",".join(attempted)
        self.retryable = True
        self.attempted = attempted


class CircuitOpen(PlatformError):
    """A dependency is being shed because it is failing."""

    status_code = 503
    code = "circuit_open"


class TimeoutExceeded(PlatformError):
    status_code = 504
    code = "timeout"
