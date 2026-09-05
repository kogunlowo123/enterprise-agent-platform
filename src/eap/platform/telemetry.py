"""Structured logging and tracing.

Two rules hold everywhere in the platform:

1. Logs are structured events, never formatted prose. ``log.info("retrieval.completed",
   hits=8)`` is queryable; ``log.info(f"found {n} hits")`` is not.
2. Request context is attached by the logging pipeline, not by callers. A component that
   has to remember to pass ``correlation_id=`` will eventually forget.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from eap.platform.config import ObservabilitySettings
from eap.platform.context import current_context

_configured = False


def _inject_request_context(
    _logger: Any, _name: str, event: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: fold the ambient request context into every event."""
    ctx = current_context()
    if ctx is not None:
        for key, value in ctx.log_fields().items():
            event.setdefault(key, value)
    return event


def _inject_trace_ids(
    _logger: Any, _name: str, event: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: correlate a log line with its span in the trace backend."""
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context.is_valid:
        event.setdefault("trace_id", format(span_context.trace_id, "032x"))
        event.setdefault("span_id", format(span_context.span_id, "016x"))
    return event


def configure_telemetry(settings: ObservabilitySettings, *, service_version: str) -> None:
    """Wire logging and tracing. Idempotent: safe to call from both app startup and tests."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _inject_request_context,
            _inject_trace_ids,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": service_version,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.trace_sample_ratio)),
    )
    if settings.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
        )
    trace.set_tracer_provider(provider)
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
