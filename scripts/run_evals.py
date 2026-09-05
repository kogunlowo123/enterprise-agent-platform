"""Run the evaluation suites and exit non-zero if the gate fails.

Invoked by ``make eval`` and by CI. It drives the same platform code the API drives: the
guardrail pipeline inspects each case's context exactly as it would inspect a retrieved
document, and the router serves the answer through the configured provider.

With no credentials configured this runs against the deterministic provider, which makes
the result a measurement of retrieval quality, prompt assembly and the guardrail layer
rather than of a model's mood. Point ``EAP_LLMOPS_ANTHROPIC_API_KEY`` at a real key to
measure the model as well.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from eap.llmops.cost import CostTracker
from eap.llmops.evaluation.harness import Answer, EvalCase, EvalHarness, load_cases
from eap.llmops.prompts import default_registry
from eap.llmops.providers.base import CompletionRequest, Message, Role
from eap.llmops.router import ModelRouter
from eap.platform.config import get_settings
from eap.platform.telemetry import configure_telemetry
from eap.secops.guardrails.pipeline import GuardrailPipeline

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"grounding": ROOT / "evals" / "grounding.jsonl"}
REPORT_DIR = ROOT / "reports"


def build_answer_fn(router: ModelRouter, guardrails: GuardrailPipeline):  # type: ignore[no-untyped-def]
    """Answer one case through the real prompt, guardrail and routing path."""
    template = default_registry().get("grounded_answer")

    async def answer(case: EvalCase) -> Answer:
        # An injection case's context is hostile material arriving from the knowledge
        # plane. Running it through the same guardrail the ingestion pipeline uses means
        # the suite measures the deployed control, not an idealised one.
        decision = guardrails.evaluate_retrieved(case.context) if case.context else None
        if decision is not None and not decision.allowed:
            return Answer(
                text=(
                    "The supplied sources were quarantined by the platform's guardrails, so "
                    "the available sources do not contain material that answers this "
                    "question."
                )
            )

        context = decision.text if decision is not None else ""
        system = template.render(
            agent_name="Evaluation Agent",
            organisation="the organisation",
            mission="Answer strictly from the supplied sources, citing each claim.",
            context=context or "(no sources were retrieved)",
            question=case.question,
        )
        result = await router.complete(
            CompletionRequest(
                messages=[
                    Message(role=Role.SYSTEM, content=system),
                    Message(role=Role.USER, content=case.question),
                ],
                max_output_tokens=512,
            ),
            tenant_id="eval",
            route="balanced",
        )
        return Answer(
            text=result.response.text,
            latency_ms=result.response.latency_ms,
            cost_usd=result.cost_usd,
        )

    return answer


async def main() -> int:
    # Deliberately not build_platform(): evaluation needs a router and the guardrail
    # pipeline, and nothing else. Requiring an identity provider to run an eval suite would
    # mean the gate could not run in a CI job that has no IdP -- which is every CI job.
    from eap.bootstrap import build_providers, build_routes

    settings = get_settings()
    configure_telemetry(settings.observability, service_version=settings.service_version)

    providers = build_providers(settings)
    router = ModelRouter(
        providers=providers,
        routes=build_routes(providers),
        cost=CostTracker(daily_budget_usd=settings.llmops.daily_tenant_budget_usd),
        timeout_seconds=settings.netops.provider_timeout_seconds,
    )
    guardrails = GuardrailPipeline(
        block_on_injection=settings.secops.block_on_injection,
        injection_threshold=settings.secops.injection_threshold,
    )
    answer_fn = build_answer_fn(router, guardrails)

    failed = False
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in SUITES.items():
        if not path.exists():
            print(f"suite '{name}': dataset not found at {path}", file=sys.stderr)
            return 2

        cases = load_cases(path)
        report = await EvalHarness(suite=name, cases=cases).run(answer_fn)
        report.write_json(REPORT_DIR / f"eval-{name}.json")

        print(report.render())
        print()

        met, _ = report.meets_thresholds()
        if not met or report.passed != report.total:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
