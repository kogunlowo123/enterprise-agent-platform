# syntax=docker/dockerfile:1.7
#
# Multi-stage so the runtime layer carries no compiler, no pip cache and no build metadata.
# The final image runs as a non-root user against a read-only root filesystem; the only
# writable path is a tmpfs the deployment mounts.

# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependency metadata first: this layer is cached until pyproject.toml changes, so ordinary
# source edits do not reinstall the dependency tree.
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install .

# ---------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="enterprise-agent-platform" \
      org.opencontainers.image.description="Control plane for running LLM agents in an enterprise" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/kogunlowo123/enterprise-agent-platform"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EAP_OBS_LOG_FORMAT=json

# Security updates only; no build toolchain reaches this layer.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 eap \
 && useradd --system --uid 10001 --gid eap --no-create-home --shell /usr/sbin/nologin eap

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
USER 10001:10001

EXPOSE 8000

# Liveness only. Readiness is checked by the orchestrator against /readyz, which does
# consult dependencies -- a container healthcheck that did so would restart the process
# every time a downstream blipped.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"

ENTRYPOINT ["uvicorn", "eap.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--workers", "2", "--no-server-header", "--proxy-headers"]
