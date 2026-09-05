"""The agent runtime.

This is where every plane meets. One turn runs a fixed lifecycle, and the sequence is the
security design as much as the control flow:

    Receive → Resolve → Retrieve → Reason → Audit → Execute → Validate → Learn

* **Receive** — guardrails on the user's input, before it reaches anything else.
* **Resolve** — authorization, policy, budget. Everything that can refuse cheaply refuses
  here, before a token is spent.
* **Retrieve** — grounding material from the knowledge plane, tenant-scoped.
* **Reason** — the model call, through the router, with the retrieved context.
* **Audit** — the decision is recorded before it is acted on, not after. An action that is
  audited after execution is an action whose audit record can be lost by the crash that the
  action caused.
* **Execute** — tool dispatch, if the model asked for one, under the full permission chain.
* **Validate** — guardrails on the model's output, plus grounding checks.
* **Learn** — outcome written to episodic memory.

The orchestrator controls the workflow. The model reasons *inside* that workflow and never
about it: it cannot skip validation, widen its own permissions, or decide that this turn
does not need auditing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from eap.appops.memory import MemoryManager, RunOutcome
from eap.appops.tools.registry import ApprovalRequired, ToolDispatcher, ToolRegistry
from eap.dataops.retrieval import HybridRetriever, RetrievalResult
from eap.identity.models import AgentIdentity, SecurityContext
from eap.identity.rbac import PERM_AGENT_INVOKE, PERM_KNOWLEDGE_READ, Authorizer
from eap.llmops.evaluation import metrics
from eap.llmops.prompts import PromptRegistry
from eap.llmops.providers.base import CompletionRequest, Message, Role, ToolSchema
from eap.llmops.router import ModelRouter
from eap.platform.context import new_id
from eap.platform.errors import GuardrailTripped, PlatformError
from eap.platform.telemetry import get_logger, get_tracer
from eap.secops.audit import AuditAction, AuditLog, Outcome
from eap.secops.guardrails.pipeline import GuardrailPipeline
from eap.secops.policy import DataClassification, PolicyEngine, PolicyRequest

log = get_logger(__name__)
tracer = get_tracer(__name__)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    question: str
    session_id: str = "default"
    route: str = "balanced"
    corpus: str | None = None
    classification: DataClassification = DataClassification.INTERNAL
    top_k: int = 8
    approval_token: str | None = None
    max_output_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class AgentTurn:
    run_id: str
    answer: str
    citations: tuple[str, ...]
    model: str
    provider: str
    route: str
    fell_back: bool
    cost_usd: float
    latency_ms: float
    retrieval_strategy: str
    documents_retrieved: int
    groundedness: float
    tools_invoked: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    audit_head: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "answer": self.answer,
            "citations": list(self.citations),
            "model": self.model,
            "provider": self.provider,
            "route": self.route,
            "fell_back": self.fell_back,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
            "retrieval": {
                "strategy": self.retrieval_strategy,
                "documents": self.documents_retrieved,
                "groundedness": round(self.groundedness, 4),
            },
            "tools_invoked": list(self.tools_invoked),
            "warnings": list(self.warnings),
            "audit_head": self.audit_head,
        }


class Orchestrator:
    """Runs one agent turn end to end."""

    def __init__(
        self,
        *,
        agent: AgentIdentity,
        router: ModelRouter,
        retriever: HybridRetriever,
        guardrails: GuardrailPipeline,
        policy: PolicyEngine,
        authorizer: Authorizer,
        audit: AuditLog,
        prompts: PromptRegistry,
        memory: MemoryManager | None = None,
        tools: ToolRegistry | None = None,
        dispatcher: ToolDispatcher | None = None,
        organisation: str = "the organisation",
        min_groundedness: float = 0.35,
    ) -> None:
        self._agent = agent
        self._router = router
        self._retriever = retriever
        self._guardrails = guardrails
        self._policy = policy
        self._authorizer = authorizer
        self._audit = audit
        self._prompts = prompts
        self._memory = memory or MemoryManager()
        self._tools = tools
        self._dispatcher = dispatcher
        self._organisation = organisation
        self._min_groundedness = min_groundedness

    @property
    def agent(self) -> AgentIdentity:
        return self._agent

    async def run(
        self, request: AgentRequest, *, ctx: SecurityContext, correlation_id: str
    ) -> AgentTurn:
        run_id = new_id("run")
        started = time.perf_counter()
        warnings: list[str] = []

        with tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("agent.id", self._agent.id)
            span.set_attribute("tenant.id", ctx.tenant.id)
            span.set_attribute("run.id", run_id)

            question = self._receive(request, ctx=ctx, correlation_id=correlation_id)
            self._resolve(request, ctx=ctx, correlation_id=correlation_id, run_id=run_id)
            retrieval = await self._retrieve(
                request, question, ctx=ctx, correlation_id=correlation_id
            )

            if not retrieval.documents:
                warnings.append("no supporting documents were retrieved for this question")

            decision = await self._reason(
                request, question, retrieval, ctx=ctx, correlation_id=correlation_id, run_id=run_id
            )

            self._audit.record(
                AuditAction.MODEL_INVOKED,
                outcome=Outcome.ALLOWED,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource=f"{decision.response.provider}/{decision.response.model}",
                run_id=run_id,
                route=decision.route,
                fell_back=str(decision.fell_back),
                input_tokens=str(decision.response.usage.input_tokens),
                output_tokens=str(decision.response.usage.output_tokens),
                cost_usd=f"{decision.cost_usd:.6f}",
            )

            tools_invoked = await self._execute(
                decision, ctx=ctx, correlation_id=correlation_id, request=request, warnings=warnings
            )

            answer = self._validate(
                decision.response.text,
                retrieval=retrieval,
                ctx=ctx,
                correlation_id=correlation_id,
                warnings=warnings,
            )

            grounding = metrics.groundedness(answer, retrieval.to_context())
            if retrieval.documents and grounding.score < self._min_groundedness:
                warnings.append(
                    f"answer has low lexical overlap with its sources "
                    f"({grounding.score:.2f}); treat it as unverified"
                )

            self._learn(run_id, request, answer, tools_invoked, ctx=ctx)

            latency_ms = (time.perf_counter() - started) * 1000
            self._audit.record(
                AuditAction.AGENT_RUN_COMPLETED,
                outcome=Outcome.ALLOWED,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource=self._agent.id,
                run_id=run_id,
                citations=str(len(retrieval.citations)),
                warnings=str(len(warnings)),
            )

            span.set_attribute("run.cost_usd", decision.cost_usd)
            span.set_attribute("run.groundedness", grounding.score)

            return AgentTurn(
                run_id=run_id,
                answer=answer,
                citations=retrieval.citations,
                model=decision.response.model,
                provider=decision.response.provider,
                route=decision.route,
                fell_back=decision.fell_back,
                cost_usd=decision.cost_usd,
                latency_ms=latency_ms,
                retrieval_strategy=retrieval.strategy,
                documents_retrieved=len(retrieval.documents),
                groundedness=grounding.score,
                tools_invoked=tuple(tools_invoked),
                warnings=tuple(warnings),
                audit_head=self._audit.head_hash,
            )

    def _receive(self, request: AgentRequest, *, ctx: SecurityContext, correlation_id: str) -> str:
        decision = self._guardrails.evaluate_input(request.question)
        if not decision.allowed:
            self._audit.record(
                AuditAction.GUARDRAIL_TRIPPED,
                outcome=Outcome.DENIED,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource="user_input",
                reason=decision.reason,
                blocked_by=",".join(decision.blocked_by),
            )
            decision.raise_if_blocked()
        return decision.text

    def _resolve(
        self, request: AgentRequest, *, ctx: SecurityContext, correlation_id: str, run_id: str
    ) -> None:
        try:
            self._authorizer.require(ctx, PERM_AGENT_INVOKE)
            self._authorizer.require(ctx, PERM_KNOWLEDGE_READ)
        except PlatformError as exc:
            self._audit.record(
                AuditAction.AUTHZ_DENIED,
                outcome=Outcome.DENIED,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource=self._agent.id,
                reason=exc.message,
            )
            raise

        policy_decision = self._policy.evaluate(
            PolicyRequest(
                action="agent.invoke",
                security=ctx,
                resource=self._agent.id,
                classification=request.classification,
                attributes={"route": request.route},
            )
        )
        if not policy_decision.allowed:
            self._audit.record(
                AuditAction.POLICY_DENIED,
                outcome=Outcome.DENIED,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource=self._agent.id,
                reason=policy_decision.reason,
                rule=policy_decision.rule_id,
            )
            self._policy.enforce(
                PolicyRequest(
                    action="agent.invoke",
                    security=ctx,
                    resource=self._agent.id,
                    classification=request.classification,
                )
            )

        self._audit.record(
            AuditAction.AGENT_RUN_STARTED,
            outcome=Outcome.ALLOWED,
            tenant_id=ctx.tenant.id,
            actor=str(ctx.principal),
            correlation_id=correlation_id,
            resource=self._agent.id,
            run_id=run_id,
            route=request.route,
            classification=str(request.classification),
            obligations=",".join(policy_decision.obligations),
        )

    async def _retrieve(
        self, request: AgentRequest, question: str, *, ctx: SecurityContext, correlation_id: str
    ) -> RetrievalResult:
        result = await self._retriever.retrieve(
            question, tenant_id=ctx.tenant.id, top_k=request.top_k, corpus=request.corpus
        )
        self._audit.record(
            AuditAction.KNOWLEDGE_RETRIEVED,
            outcome=Outcome.ALLOWED,
            tenant_id=ctx.tenant.id,
            actor=str(ctx.principal),
            correlation_id=correlation_id,
            resource=request.corpus or "default",
            documents=str(len(result.documents)),
            strategy=result.strategy,
        )
        return result

    async def _reason(
        self,
        request: AgentRequest,
        question: str,
        retrieval: RetrievalResult,
        *,
        ctx: SecurityContext,
        correlation_id: str,
        run_id: str,
    ) -> Any:
        template = self._prompts.get("grounded_answer")
        system_prompt = template.render(
            agent_name=self._agent.name,
            organisation=self._organisation,
            mission=self._agent.mission,
            context=retrieval.to_context() or "(no sources were retrieved)",
            question=question,
        )

        messages = [Message(role=Role.SYSTEM, content=system_prompt)]
        for entry in self._memory.session_transcript(
            tenant_id=ctx.tenant.id,
            principal_id=ctx.principal.subject,
            session_id=request.session_id,
        ):
            messages.append(
                Message(
                    role=Role.ASSISTANT if entry.role == "assistant" else Role.USER,
                    content=entry.content,
                )
            )
        messages.append(Message(role=Role.USER, content=question))

        tools: tuple[ToolSchema, ...] = ()
        if self._tools is not None:
            tools = self._tools.schemas_for(ctx)

        return await self._router.complete(
            CompletionRequest(
                messages=messages,
                max_output_tokens=request.max_output_tokens,
                tools=tools,
            ),
            tenant_id=ctx.tenant.id,
            route=request.route,
            correlation_id=correlation_id,
            agent_id=self._agent.id,
        )

    async def _execute(
        self,
        decision: Any,
        *,
        ctx: SecurityContext,
        correlation_id: str,
        request: AgentRequest,
        warnings: list[str],
    ) -> list[str]:
        if not decision.response.wants_tool_call or self._dispatcher is None:
            return []

        invoked: list[str] = []
        calls = decision.response.tool_calls[: self._agent.max_tool_calls_per_turn]
        if len(decision.response.tool_calls) > len(calls):
            warnings.append(
                f"the model requested {len(decision.response.tool_calls)} tool calls; the "
                f"agent's per-turn ceiling of {self._agent.max_tool_calls_per_turn} applied"
            )

        for call in calls:
            try:
                await self._dispatcher.dispatch(
                    call.name,
                    call.arguments,
                    ctx=ctx,
                    correlation_id=correlation_id,
                    classification=request.classification,
                    approval_token=request.approval_token,
                )
                invoked.append(call.name)
            except ApprovalRequired as exc:
                warnings.append(f"tool '{exc.tool}' is pending human approval ({exc.request_id})")
            except PlatformError as exc:
                warnings.append(f"tool '{call.name}' was refused: {exc.message}")
        return invoked

    def _validate(
        self,
        text: str,
        *,
        retrieval: RetrievalResult,
        ctx: SecurityContext,
        correlation_id: str,
        warnings: list[str],
    ) -> str:
        decision = self._guardrails.evaluate_output(text)
        if not decision.allowed:
            self._audit.record(
                AuditAction.GUARDRAIL_TRIPPED,
                outcome=Outcome.DENIED,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource="model_output",
                reason=decision.reason,
                blocked_by=",".join(decision.blocked_by),
            )
            raise GuardrailTripped(
                "the generated answer was withheld because it failed output validation",
                rule=",".join(decision.blocked_by) or "output_guardrail",
            )

        if decision.text != text:
            warnings.append("sensitive values in the answer were redacted before it was returned")

        citation_check = metrics.citation_coverage(
            decision.text, source_count=len(retrieval.documents)
        )
        if retrieval.documents and not citation_check.passed:
            warnings.append(f"citation coverage below threshold: {citation_check.detail}")

        return decision.text

    def _learn(
        self,
        run_id: str,
        request: AgentRequest,
        answer: str,
        tools_invoked: list[str],
        *,
        ctx: SecurityContext,
    ) -> None:
        self._memory.record_turn(
            tenant_id=ctx.tenant.id,
            principal_id=ctx.principal.subject,
            session_id=request.session_id,
            role="user",
            content=request.question,
        )
        self._memory.record_turn(
            tenant_id=ctx.tenant.id,
            principal_id=ctx.principal.subject,
            session_id=request.session_id,
            role="assistant",
            content=answer,
        )
        self._memory.record_outcome(
            tenant_id=ctx.tenant.id,
            principal_id=ctx.principal.subject,
            outcome=RunOutcome(
                run_id=run_id,
                question=request.question,
                succeeded=True,
                summary=answer[:400],
                tools_used=tuple(tools_invoked),
            ),
        )
