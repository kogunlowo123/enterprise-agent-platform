"""Agent invocation and knowledge ingestion endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from eap.api.dependencies import CorrelationDep, PlatformDep, SecurityDep, bind_context, context_for
from eap.appops.orchestrator import AgentRequest
from eap.dataops.connectors.github import GitHubConnector, GitHubRepoConfig
from eap.identity.rbac import PERM_KNOWLEDGE_WRITE
from eap.secops.policy import DataClassification

router = APIRouter(tags=["agent"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default="default", max_length=128)
    route: Literal["fast", "balanced", "deep"] = "balanced"
    corpus: str | None = Field(default=None, max_length=64)
    classification: DataClassification = DataClassification.INTERNAL
    top_k: int = Field(default=8, ge=1, le=25)
    approval_token: str | None = None


class RetrievalSummary(BaseModel):
    strategy: str
    documents: int
    groundedness: float


class AskResponse(BaseModel):
    run_id: str
    answer: str
    citations: list[str]
    model: str
    provider: str
    route: str
    fell_back: bool
    cost_usd: float
    latency_ms: float
    retrieval: RetrievalSummary
    tools_invoked: list[str]
    warnings: list[str]
    audit_head: str


@router.post("/v1/agent/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    platform: PlatformDep,
    ctx: SecurityDep,
    correlation_id: CorrelationDep,
) -> AskResponse:
    """Run one agent turn: guardrails, authorization, retrieval, model, validation, audit."""
    with bind_context(context_for(ctx, correlation_id)):
        turn = await platform.orchestrator.run(
            AgentRequest(
                question=body.question,
                session_id=body.session_id,
                route=body.route,
                corpus=body.corpus,
                classification=body.classification,
                top_k=body.top_k,
                approval_token=body.approval_token,
            ),
            ctx=ctx,
            correlation_id=correlation_id,
        )
    return AskResponse(**turn.to_dict())


class GitHubIngestRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=120)
    ref: str = Field(default="HEAD", max_length=120)
    corpus: str = Field(default="default", max_length=64)
    include_globs: list[str] = Field(default_factory=lambda: ["**/*"])
    max_files: int = Field(default=2000, ge=1, le=20000)
    api_url: str = "https://api.github.com"


class IngestResponse(BaseModel):
    corpus: str
    commit: str | None
    sources_seen: int
    sources_quarantined: int
    chunks_written: int
    chunks_replaced: int
    quarantined: list[dict[str, str]]
    lexical_documents: int


@router.post(
    "/v1/knowledge/github",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_github(
    body: GitHubIngestRequest,
    platform: PlatformDep,
    ctx: SecurityDep,
    correlation_id: CorrelationDep,
) -> IngestResponse:
    """Ingest an existing enterprise repository into this tenant's knowledge corpus.

    Synchronous by design at this size. A repository large enough to need a job queue is
    also large enough to need scheduling, retry and progress reporting, which belong in a
    worker rather than behind a request that will time out at the load balancer.
    """
    platform.authorizer.require(ctx, PERM_KNOWLEDGE_WRITE)

    token = platform.settings.dataops.github_token
    connector = GitHubConnector(
        GitHubRepoConfig(
            owner=body.owner,
            repo=body.repo,
            ref=body.ref,
            include_globs=tuple(body.include_globs),
            max_files=body.max_files,
            api_url=body.api_url,
        ),
        token=token.get_secret_value() if token else None,
    )

    with bind_context(context_for(ctx, correlation_id)):
        try:
            report = await platform.ingestion.ingest(
                connector,
                tenant_id=ctx.tenant.id,
                corpus=body.corpus,
                correlation_id=correlation_id,
                actor=str(ctx.principal),
            )
        finally:
            await connector.aclose()

        indexed = await platform.reindex_lexical(ctx.tenant.id)

    return IngestResponse(
        corpus=report.corpus,
        commit=connector.resolved_sha,
        sources_seen=report.sources_seen,
        sources_quarantined=report.sources_quarantined,
        chunks_written=report.chunks_written,
        chunks_replaced=report.chunks_replaced,
        quarantined=[{"source": s, "reason": r} for s, r in report.quarantined],
        lexical_documents=indexed,
    )


class SearchResponse(BaseModel):
    query: str
    strategy: str
    results: list[dict[str, object]]


@router.get("/v1/knowledge/search", response_model=SearchResponse)
async def search(
    q: str,
    platform: PlatformDep,
    ctx: SecurityDep,
    top_k: int = 8,
    corpus: str | None = None,
) -> SearchResponse:
    """Retrieval without generation. The endpoint you use to debug a bad answer."""
    from eap.identity.rbac import PERM_KNOWLEDGE_READ

    platform.authorizer.require(ctx, PERM_KNOWLEDGE_READ)
    result = await platform.retriever.retrieve(
        q, tenant_id=ctx.tenant.id, top_k=min(top_k, 25), corpus=corpus
    )
    return SearchResponse(
        query=q,
        strategy=result.strategy,
        results=[
            {
                "score": round(document.score, 6),
                "retriever": document.retriever,
                "citation": document.citation,
                "excerpt": document.text[:400],
            }
            for document in result.documents
        ],
    )
