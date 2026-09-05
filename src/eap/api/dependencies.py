"""Request-scoped dependencies.

The authentication chain in one place: bearer token → verified principal → tenant →
resolved permissions → rate limit → bound request context. Routes declare
``ctx: SecurityContext = Depends(require_security_context)`` and receive a caller whose
authority is already settled, so no route can accidentally serve an unauthenticated
request by forgetting a check.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from eap.bootstrap import Platform
from eap.identity.models import Principal, SecurityContext, Tenant
from eap.platform.context import RequestContext, bind_context, new_id
from eap.platform.errors import AuthenticationError
from eap.secops.audit import AuditAction, Outcome

bearer_scheme = HTTPBearer(auto_error=False)


def get_platform(request: Request) -> Platform:
    platform: Platform | None = getattr(request.app.state, "platform", None)
    if platform is None:  # pragma: no cover - only reachable if the lifespan did not run
        raise RuntimeError("platform was not assembled; application startup did not complete")
    return platform


PlatformDep = Annotated[Platform, Depends(get_platform)]


def get_correlation_id(request: Request) -> str:
    """Honour an inbound correlation id so a trace spans the whole call graph."""
    return request.headers.get("x-correlation-id") or new_id("req")


def resolve_tenant(platform: Platform, principal: Principal) -> Tenant:
    """Look up the tenant named by the token.

    Backed by a directory in a real deployment. The token's tenant claim is authoritative
    for *which* tenant, never for the tenant's properties: residency and suspension come
    from the platform's own record, or a token minted by a compromised client could declare
    itself resident anywhere it liked.
    """
    return Tenant(id=principal.tenant_id, name=principal.tenant_id, data_residency="us")


async def require_security_context(
    request: Request,
    platform: PlatformDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> SecurityContext:
    correlation_id = get_correlation_id(request)

    if credentials is None or not credentials.credentials:
        raise AuthenticationError("this endpoint requires a bearer token")

    try:
        principal = platform.verifier.verify(credentials.credentials)
    except AuthenticationError as exc:
        platform.audit.record(
            AuditAction.AUTH_FAILED,
            outcome=Outcome.DENIED,
            tenant_id="unknown",
            actor="anonymous",
            correlation_id=correlation_id,
            reason=exc.message,
            path=request.url.path,
        )
        raise

    tenant = resolve_tenant(platform, principal)
    ctx = platform.authorizer.build_context(principal, tenant)

    platform.rate_limiter.enforce(principal_id=principal.subject, tenant_id=tenant.id, cost=1.0)

    platform.audit.record(
        AuditAction.AUTH_SUCCEEDED,
        outcome=Outcome.ALLOWED,
        tenant_id=tenant.id,
        actor=str(principal),
        correlation_id=correlation_id,
        path=request.url.path,
    )

    request.state.correlation_id = correlation_id
    request.state.security = ctx
    return ctx


SecurityDep = Annotated[SecurityContext, Depends(require_security_context)]
CorrelationDep = Annotated[str, Depends(get_correlation_id)]


def context_for(ctx: SecurityContext, correlation_id: str) -> RequestContext:
    return RequestContext(correlation_id=correlation_id, principal=ctx.principal)


__all__ = [
    "CorrelationDep",
    "PlatformDep",
    "SecurityDep",
    "bind_context",
    "context_for",
    "get_platform",
    "require_security_context",
]
