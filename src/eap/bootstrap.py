"""Composition root.

Every dependency is constructed here and injected downwards. No module reaches for a global,
and nothing below this file knows how it was configured — which is what makes the whole tree
testable by substitution rather than by monkeypatching.

The provider set is assembled from whichever credentials are present. A deployment with only
an Anthropic key gets an Anthropic-only router rather than a startup failure, and routes fall
back through whatever is actually available.
"""

from __future__ import annotations

from dataclasses import dataclass

from eap.appops.mcp.gateway import MCPGateway
from eap.appops.memory import MemoryManager
from eap.appops.orchestrator import Orchestrator
from eap.appops.tools.registry import ToolDispatcher, ToolRegistry
from eap.dataops.chunking import Chunker, ChunkingConfig
from eap.dataops.embeddings import Embedder, build_embedder
from eap.dataops.ingest import IngestionPipeline
from eap.dataops.retrieval import HybridRetriever, LexicalIndex
from eap.dataops.vectorstore import InMemoryVectorStore, PgVectorStore, VectorStore
from eap.identity.models import AgentIdentity
from eap.identity.rbac import (
    PERM_AGENT_INVOKE,
    PERM_KNOWLEDGE_READ,
    PERM_MODEL_INVOKE,
    PERM_TOOL_EXECUTE,
    Authorizer,
    RoleRegistry,
)
from eap.identity.tokens import TokenVerifier
from eap.llmops.cost import CostTracker
from eap.llmops.prompts import PromptRegistry, default_registry
from eap.llmops.providers.anthropic import AnthropicProvider
from eap.llmops.providers.base import LLMProvider
from eap.llmops.providers.local import DeterministicProvider
from eap.llmops.providers.openai import OpenAIProvider
from eap.llmops.router import Candidate, ModelRouter, Route, default_routes
from eap.netops.egress import EgressGuard
from eap.netops.ratelimit import LayeredRateLimiter, RateLimiter
from eap.netops.resilience import RetryPolicy
from eap.platform.clock import SYSTEM_CLOCK, Clock
from eap.platform.config import Settings, get_settings
from eap.platform.telemetry import configure_telemetry, get_logger
from eap.secops.audit import AuditLog, build_audit_sink
from eap.secops.guardrails.base import Severity
from eap.secops.guardrails.pipeline import GuardrailPipeline
from eap.secops.policy import PolicyEngine

log = get_logger(__name__)

DEFAULT_AGENT = AgentIdentity(
    id="knowledge-assistant",
    name="Knowledge Assistant",
    mission=(
        "Answer questions about this organisation's documented policies, procedures and "
        "engineering practice, grounded strictly in retrieved sources, with a citation for "
        "every claim."
    ),
    owner="platform-team",
    non_goals=(
        "giving legal, medical or financial advice",
        "acting on systems of record without a human in the loop",
        "answering from general knowledge when the corpus is silent",
    ),
    granted_permissions=frozenset(
        {PERM_AGENT_INVOKE, PERM_KNOWLEDGE_READ, PERM_MODEL_INVOKE, PERM_TOOL_EXECUTE}
    ),
    forbidden_tools=frozenset(),
    max_tool_calls_per_turn=4,
)


@dataclass(slots=True)
class Platform:
    """The assembled platform. One instance per process."""

    settings: Settings
    verifier: TokenVerifier
    authorizer: Authorizer
    audit: AuditLog
    policy: PolicyEngine
    guardrails: GuardrailPipeline
    rate_limiter: LayeredRateLimiter
    egress: EgressGuard
    embedder: Embedder
    store: VectorStore
    lexical: LexicalIndex
    retriever: HybridRetriever
    ingestion: IngestionPipeline
    cost: CostTracker
    router: ModelRouter
    prompts: PromptRegistry
    tools: ToolRegistry
    dispatcher: ToolDispatcher
    mcp: MCPGateway
    memory: MemoryManager
    orchestrator: Orchestrator

    async def reindex_lexical(self, tenant_id: str) -> int:
        """Rebuild the BM25 index for a tenant from what is in the vector store.

        The lexical index is derived state. It is rebuilt after ingestion rather than
        maintained incrementally, because BM25's IDF term depends on corpus-wide document
        frequencies: incremental updates leave the scores subtly wrong in a way that never
        surfaces as an error, only as gradually worse results.
        """
        if not isinstance(self.store, InMemoryVectorStore):
            return 0
        documents = await self.store.all_documents(tenant_id)
        self.lexical.build(tenant_id, documents)
        return len(documents)


def build_providers(settings: Settings) -> dict[str, LLMProvider]:
    providers: dict[str, LLMProvider] = {}

    anthropic_key = settings.llmops.anthropic_api_key
    if anthropic_key and anthropic_key.get_secret_value():
        providers["anthropic"] = AnthropicProvider(
            api_key=anthropic_key.get_secret_value(),
            timeout=settings.netops.provider_timeout_seconds,
        )

    openai_key = settings.llmops.openai_api_key
    if openai_key and openai_key.get_secret_value():
        providers["openai"] = OpenAIProvider(
            api_key=openai_key.get_secret_value(),
            timeout=settings.netops.provider_timeout_seconds,
        )

    if not providers:
        # No credentials configured. Rather than refusing to start, serve the deterministic
        # provider so that the platform is exercisable end to end offline. Settings blocks
        # this combination in staging and prod.
        log.warning(
            "bootstrap.no_model_credentials",
            detail="serving the deterministic local provider; not suitable for production",
        )
        providers["local"] = DeterministicProvider()
    return providers


def build_routes(providers: dict[str, LLMProvider]) -> dict[str, Route]:
    """Drop candidates whose provider is not configured, and never leave a route empty."""
    available = set(providers)
    routes: dict[str, Route] = {}

    for name, route in default_routes().items():
        candidates = tuple(c for c in route.candidates if c.provider in available)
        if not candidates:
            fallback_provider = next(iter(available))
            fallback_model = next(iter(providers[fallback_provider].supported_models))
            candidates = (
                Candidate(fallback_provider, fallback_model, reason="only configured provider"),
            )
        routes[name] = Route(name=name, candidates=candidates, description=route.description)
    return routes


def build_store(settings: Settings, pool: object | None = None) -> VectorStore:
    if settings.dataops.vector_store == "pgvector":
        if pool is None:
            raise ValueError("pgvector requires a connection pool")
        return PgVectorStore(pool, dimensions=settings.dataops.embedding_dimensions)
    return InMemoryVectorStore()


def build_platform(
    settings: Settings | None = None,
    *,
    clock: Clock = SYSTEM_CLOCK,
    pool: object | None = None,
    agent: AgentIdentity | None = None,
) -> Platform:
    settings = settings or get_settings()
    configure_telemetry(settings.observability, service_version=settings.service_version)

    audit = AuditLog(
        build_audit_sink(settings.secops.audit_sink, file_path=settings.secops.audit_file_path),
        clock=clock,
    )
    guardrails = GuardrailPipeline(
        block_on_injection=settings.secops.block_on_injection,
        injection_threshold=settings.secops.injection_threshold,
        block_severity=Severity.CRITICAL,
        redact=settings.secops.redact_pii_in_prompts,
    )

    embedder = build_embedder(
        settings.dataops.embedding_provider,
        dimensions=settings.dataops.embedding_dimensions,
        model=settings.dataops.embedding_model,
        api_key=(
            settings.llmops.openai_api_key.get_secret_value()
            if settings.llmops.openai_api_key
            else None
        ),
    )
    store = build_store(settings, pool)
    lexical = LexicalIndex()
    retriever = HybridRetriever(store=store, embedder=embedder, lexical_index=lexical)

    ingestion = IngestionPipeline(
        store=store,
        embedder=embedder,
        chunker=Chunker(
            ChunkingConfig(
                max_tokens=settings.dataops.chunk_max_tokens,
                overlap_tokens=settings.dataops.chunk_overlap_tokens,
            )
        ),
        guardrails=guardrails,
        audit=audit,
    )

    providers = build_providers(settings)
    cost = CostTracker(daily_budget_usd=settings.llmops.daily_tenant_budget_usd, clock=clock)
    router = ModelRouter(
        providers=providers,
        routes=build_routes(providers),
        cost=cost,
        retry=RetryPolicy(max_attempts=settings.netops.max_retries + 1),
        timeout_seconds=settings.netops.provider_timeout_seconds,
    )

    authorizer = Authorizer(RoleRegistry())
    policy = PolicyEngine()
    tools = ToolRegistry()
    dispatcher = ToolDispatcher(registry=tools, authorizer=authorizer, policy=policy, audit=audit)
    mcp = MCPGateway(registry=tools, guardrails=guardrails)
    memory = MemoryManager(clock=clock)
    prompts = default_registry()

    orchestrator = Orchestrator(
        agent=agent or DEFAULT_AGENT,
        router=router,
        retriever=retriever,
        guardrails=guardrails,
        policy=policy,
        authorizer=authorizer,
        audit=audit,
        prompts=prompts,
        memory=memory,
        tools=tools,
        dispatcher=dispatcher,
    )

    rate_limiter = LayeredRateLimiter(
        principal=RateLimiter(
            rate_per_minute=settings.netops.rate_limit_per_minute,
            burst=settings.netops.rate_limit_burst,
            clock=clock,
        ),
        tenant=RateLimiter(
            rate_per_minute=settings.netops.rate_limit_per_minute * 10,
            burst=settings.netops.rate_limit_burst * 10,
            clock=clock,
        ),
    )

    log.info(
        "bootstrap.completed",
        environment=settings.environment,
        providers=sorted(providers),
        vector_store=settings.dataops.vector_store,
        embedding_provider=settings.dataops.embedding_provider,
    )

    return Platform(
        settings=settings,
        verifier=TokenVerifier(settings.identity),
        authorizer=authorizer,
        audit=audit,
        policy=policy,
        guardrails=guardrails,
        rate_limiter=rate_limiter,
        egress=EgressGuard(settings.netops.egress_allowlist),
        embedder=embedder,
        store=store,
        lexical=lexical,
        retriever=retriever,
        ingestion=ingestion,
        cost=cost,
        router=router,
        prompts=prompts,
        tools=tools,
        dispatcher=dispatcher,
        mcp=mcp,
        memory=memory,
        orchestrator=orchestrator,
    )
