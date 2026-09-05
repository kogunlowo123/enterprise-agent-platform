"""FastAPI application.

The edge does four things and nothing else: assemble the platform once at startup, stamp a
correlation id on every request, translate the platform's error taxonomy into RFC 9457
problem documents, and route.

Error translation matters more than it looks. Without it, a ``PolicyViolation`` reaches the
client as a 500 with a stack trace, which is both a worse API and an information leak. With
it, every failure the platform can produce has a status code, a stable code, and no internal
detail beyond what the caller needs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from eap.api.routes import agent, operations
from eap.bootstrap import Platform, build_platform
from eap.platform.config import Settings, get_settings
from eap.platform.context import RequestContext, bind_context, new_id
from eap.platform.errors import PlatformError
from eap.platform.telemetry import get_logger

log = get_logger(__name__)

DESCRIPTION = """
Control plane for running LLM agents inside an enterprise.

Every request passes through the same chain: identity, governance, knowledge, model routing
and runtime. Refusals name the rule that produced them, and every decision is written to a
hash-chained audit log that can be verified at `/v1/audit/verify`.
""".strip()


def create_app(settings: Settings | None = None, *, platform: Platform | None = None) -> FastAPI:
    """Build the application.

    ``platform`` is injectable so tests can assemble one with a manual clock, a scripted
    provider and an in-memory audit sink, then drive the real HTTP surface against it.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.platform = platform or build_platform(resolved)
        log.info("api.started", environment=resolved.environment)
        try:
            yield
        finally:
            await app.state.platform.mcp.shutdown()
            log.info("api.stopped")

    app = FastAPI(
        title="Enterprise Agent Platform",
        description=DESCRIPTION,
        version=resolved.service_version,
        lifespan=lifespan,
        docs_url="/docs" if not resolved.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not resolved.is_production else None,
    )

    @app.middleware("http")
    async def correlate(request: Request, call_next):  # type: ignore[no-untyped-def]
        correlation_id = request.headers.get("x-correlation-id") or new_id("req")
        with bind_context(RequestContext(correlation_id=correlation_id)):
            response = await call_next(request)
        response.headers["x-correlation-id"] = correlation_id
        return response

    @app.exception_handler(PlatformError)
    async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", None) or request.headers.get(
            "x-correlation-id", ""
        )
        log.warning(
            "api.request_refused",
            code=exc.code,
            status=exc.status_code,
            path=request.url.path,
            detail=exc.message,
        )
        headers = {}
        retry_after = exc.details.get("retry_after_seconds")
        if retry_after is not None:
            headers["Retry-After"] = str(int(float(retry_after)) or 1)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_problem(correlation_id=correlation_id or None),
            media_type="application/problem+json",
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """Anything reaching here is a bug. Log it fully; tell the caller nothing."""
        correlation_id = getattr(request.state, "correlation_id", None) or ""
        log.error(
            "api.unhandled_exception",
            path=request.url.path,
            error=type(exc).__name__,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "type": "https://docs.eap.dev/errors/internal_error",
                "title": "internal error",
                "status": 500,
                "detail": "the request could not be completed",
                "correlation_id": correlation_id,
            },
            media_type="application/problem+json",
        )

    app.include_router(operations.router)
    app.include_router(agent.router)

    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")
    return app


app = create_app()
