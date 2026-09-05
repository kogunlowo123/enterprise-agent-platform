"""Health, readiness and the operational surface.

Liveness and readiness answer different questions and must not share an implementation.
Liveness asks "is this process wedged" — if it returns false the orchestrator restarts the
pod, so it must not depend on anything external. A liveness probe that checks the database
turns a database blip into a cluster-wide restart storm.

Readiness asks "can this replica serve traffic right now", and *should* check dependencies,
because the correct response to a broken dependency is to leave the load balancer pool
rather than to serve errors.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from eap.api.dependencies import PlatformDep, SecurityDep
from eap.identity.rbac import PERM_AUDIT_READ, PERM_TENANT_ADMIN

router = APIRouter(tags=["operations"])


class LivenessResponse(BaseModel):
    status: str
    version: str


@router.get("/healthz", response_model=LivenessResponse)
async def liveness(platform: PlatformDep) -> LivenessResponse:
    """Process is alive. No dependency is consulted, deliberately."""
    return LivenessResponse(status="ok", version=platform.settings.service_version)


@router.get("/readyz")
async def readiness(platform: PlatformDep, response: Response) -> dict[str, Any]:
    """Every dependency this replica needs in order to serve a request."""
    checks: dict[str, Any] = {}

    checks["configuration"] = {"ok": True, "environment": platform.settings.environment}
    checks["model_router"] = {
        "ok": bool(platform.router.routes),
        "routes": sorted(platform.router.routes),
    }
    checks["knowledge_store"] = {"ok": platform.store is not None}

    audit_state = platform.audit.verify()
    checks["audit_chain"] = {
        "ok": audit_state.valid,
        "records": audit_state.records_checked,
        "head": audit_state.head_hash[:16],
        "broken_at": audit_state.broken_at,
    }

    checks["mcp_servers"] = await platform.mcp.health()

    ready = all(check.get("ok", True) for check in checks.values() if isinstance(check, dict))
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "checks": checks}


class AuditVerificationResponse(BaseModel):
    valid: bool
    records_checked: int
    head_hash: str
    broken_at: int | None = None
    reason: str | None = None


@router.get("/v1/audit/verify", response_model=AuditVerificationResponse)
async def verify_audit(platform: PlatformDep, ctx: SecurityDep) -> AuditVerificationResponse:
    """Recompute the audit hash chain and report whether it is intact."""
    platform.authorizer.require(ctx, PERM_AUDIT_READ)
    result = platform.audit.verify()
    return AuditVerificationResponse(
        valid=result.valid,
        records_checked=result.records_checked,
        head_hash=result.head_hash,
        broken_at=result.broken_at,
        reason=result.reason,
    )


@router.get("/v1/cost/attribution")
async def cost_attribution(platform: PlatformDep, ctx: SecurityDep) -> dict[str, object]:
    """Today's spend for the caller's tenant, split by model and agent."""
    platform.authorizer.require(ctx, PERM_TENANT_ADMIN)
    return platform.cost.attribution(ctx.tenant.id)


@router.get("/v1/policy/rules")
async def list_policy_rules(platform: PlatformDep, ctx: SecurityDep) -> dict[str, object]:
    """The governance rules in force. Auditors ask for this before they ask for anything else."""
    platform.authorizer.require(ctx, PERM_AUDIT_READ)
    return {
        "rules": [
            {
                "id": rule.id,
                "effect": str(rule.effect),
                "description": rule.description,
                "obligations": list(rule.obligations),
            }
            for rule in platform.policy.rules
        ]
    }
