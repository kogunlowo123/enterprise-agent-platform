"""Shared fixtures.

Every fixture builds a real object. Nothing here is a mock: the clock is manual, the
provider is deterministic, the store is in-memory, and all three implement the same
contracts their production counterparts do. Tests therefore exercise the actual code paths
rather than an approximation of them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from eap.appops.memory import MemoryManager
from eap.appops.orchestrator import Orchestrator
from eap.appops.tools.registry import ToolDispatcher, ToolRegistry
from eap.dataops.chunking import Chunker, ChunkingConfig
from eap.dataops.embeddings import DeterministicEmbedder
from eap.dataops.ingest import IngestionPipeline
from eap.dataops.retrieval import HybridRetriever, LexicalIndex
from eap.dataops.vectorstore import InMemoryVectorStore
from eap.identity.models import AgentIdentity, Principal, PrincipalType, Tenant
from eap.identity.rbac import (
    PERM_AGENT_INVOKE,
    PERM_KNOWLEDGE_READ,
    PERM_KNOWLEDGE_WRITE,
    PERM_MODEL_INVOKE,
    PERM_TOOL_EXECUTE,
    PERM_TOOL_EXECUTE_WRITE,
    Authorizer,
)
from eap.llmops.cost import CostTracker
from eap.llmops.prompts import default_registry
from eap.llmops.providers.local import DeterministicProvider
from eap.llmops.router import Candidate, ModelRouter, Route
from eap.platform.clock import ManualClock
from eap.secops.audit import AuditLog, InMemoryAuditSink
from eap.secops.guardrails.pipeline import GuardrailPipeline
from eap.secops.policy import PolicyEngine

DEV_KEY = "test-signing-key-not-used-anywhere-real"
ISSUER = "https://issuer.test/eap"
AUDIENCE = "eap-gateway"


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(start=datetime(2026, 3, 1, 12, 0, tzinfo=UTC))


@pytest.fixture
def audit_sink() -> InMemoryAuditSink:
    return InMemoryAuditSink()


@pytest.fixture
def audit(audit_sink: InMemoryAuditSink, clock: ManualClock) -> AuditLog:
    return AuditLog(audit_sink, clock=clock)


@pytest.fixture
def tenant() -> Tenant:
    return Tenant(id="acme", name="Acme Corp", data_residency="us")


@pytest.fixture
def other_tenant() -> Tenant:
    return Tenant(id="globex", name="Globex", data_residency="eu")


@pytest.fixture
def authorizer() -> Authorizer:
    return Authorizer()


@pytest.fixture
def agent() -> AgentIdentity:
    return AgentIdentity(
        id="test-agent",
        name="Test Agent",
        mission="Answer questions from the corpus, with citations.",
        owner="platform-team",
        non_goals=("giving legal advice",),
        granted_permissions=frozenset(
            {PERM_AGENT_INVOKE, PERM_KNOWLEDGE_READ, PERM_MODEL_INVOKE, PERM_TOOL_EXECUTE}
        ),
        forbidden_tools=frozenset({"delete_everything"}),
        max_tool_calls_per_turn=3,
    )


@pytest.fixture
def user_principal() -> Principal:
    return Principal(
        subject="alice",
        tenant_id="acme",
        principal_type=PrincipalType.USER,
        roles=frozenset({"agent.operator"}),
        email="alice@acme.test",
    )


@pytest.fixture
def security(authorizer: Authorizer, user_principal: Principal, tenant: Tenant, agent):
    return authorizer.build_context(user_principal, tenant, agent=agent)


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder(dimensions=128)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def lexical() -> LexicalIndex:
    return LexicalIndex()


@pytest.fixture
def retriever(store, embedder, lexical) -> HybridRetriever:
    return HybridRetriever(store=store, embedder=embedder, lexical_index=lexical)


@pytest.fixture
def guardrails() -> GuardrailPipeline:
    return GuardrailPipeline()


@pytest.fixture
def policy() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture
def provider() -> DeterministicProvider:
    return DeterministicProvider()


@pytest.fixture
def cost(clock: ManualClock) -> CostTracker:
    return CostTracker(daily_budget_usd=10.0, clock=clock)


@pytest.fixture
def router(provider: DeterministicProvider, cost: CostTracker) -> ModelRouter:
    return ModelRouter(
        providers={"local": provider},
        routes={
            "balanced": Route(
                name="balanced",
                candidates=(Candidate("local", "local-deterministic", reason="only provider"),),
            )
        },
        cost=cost,
        timeout_seconds=5.0,
    )


@pytest.fixture
def ingestion(store, embedder, guardrails, audit) -> IngestionPipeline:
    return IngestionPipeline(
        store=store,
        embedder=embedder,
        chunker=Chunker(ChunkingConfig(max_tokens=128, overlap_tokens=16, min_tokens=8)),
        guardrails=guardrails,
        audit=audit,
    )


@pytest.fixture
def tools() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def dispatcher(tools, authorizer, policy, audit) -> ToolDispatcher:
    return ToolDispatcher(registry=tools, authorizer=authorizer, policy=policy, audit=audit)


@pytest.fixture
def memory(clock: ManualClock) -> MemoryManager:
    return MemoryManager(clock=clock)


@pytest.fixture
def orchestrator(
    agent, router, retriever, guardrails, policy, authorizer, audit, memory, tools, dispatcher
) -> Orchestrator:
    return Orchestrator(
        agent=agent,
        router=router,
        retriever=retriever,
        guardrails=guardrails,
        policy=policy,
        authorizer=authorizer,
        audit=audit,
        prompts=default_registry(),
        memory=memory,
        tools=tools,
        dispatcher=dispatcher,
        organisation="Acme Corp",
    )


def mint_token(
    *,
    subject: str = "alice",
    tenant: str = "acme",
    roles: tuple[str, ...] = ("agent.operator",),
    principal_type: str = "user",
    expires_in_minutes: int = 30,
    audience: str = AUDIENCE,
    issuer: str = ISSUER,
    key: str = DEV_KEY,
    scopes: tuple[str, ...] | None = None,
) -> str:
    """Mint a token the verifier will accept, or deliberately will not."""
    now = datetime.now(UTC)
    claims = {
        "sub": subject,
        "tenant": tenant,
        "typ": principal_type,
        "roles": list(roles),
        "iss": issuer,
        "aud": audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_in_minutes)).timestamp()),
    }
    if scopes is not None:
        claims["scope"] = " ".join(scopes)
    return jwt.encode(claims, key, algorithm="HS256")


OPERATOR_PERMISSIONS = frozenset(
    {
        PERM_AGENT_INVOKE,
        PERM_KNOWLEDGE_READ,
        PERM_KNOWLEDGE_WRITE,
        PERM_MODEL_INVOKE,
        PERM_TOOL_EXECUTE,
        PERM_TOOL_EXECUTE_WRITE,
    }
)
