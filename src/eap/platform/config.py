"""Configuration.

Read once at startup from the environment, validated eagerly, then frozen. A setting that
is wrong should stop the process from starting, not surface as a 500 an hour later under
load. Anything secret is typed ``SecretStr`` so that a stray ``repr()`` in a log line or a
traceback prints ``**********`` instead of the value.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from eap.platform.errors import ConfigurationError

Environment = Literal["local", "dev", "staging", "prod"]


class IdentitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EAP_IDENTITY_", extra="ignore")

    issuer: str = "https://issuer.local/eap"
    audience: str = "eap-gateway"
    jwks_url: str | None = None
    """OIDC JWKS endpoint. Required outside local: production must verify RS256 against
    the identity provider's published keys rather than a shared secret."""

    dev_signing_key: SecretStr = SecretStr("")
    """HS256 key used only when ``jwks_url`` is unset. Rejected outside local."""

    clock_skew_seconds: int = Field(default=60, ge=0, le=300)
    jwks_cache_seconds: int = Field(default=600, ge=0)


class SecOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EAP_SECOPS_", extra="ignore")

    block_on_injection: bool = True
    """When false, injection findings are recorded and surfaced but do not stop the turn.
    Useful for shadow-mode rollout against real traffic before enforcing."""

    redact_pii_in_prompts: bool = True
    injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    audit_sink: Literal["memory", "file"] = "memory"
    audit_file_path: str = "./var/audit/audit.jsonl"


class NetOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EAP_NETOPS_", extra="ignore")

    rate_limit_per_minute: int = Field(default=120, ge=1)
    rate_limit_burst: int = Field(default=30, ge=1)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    provider_timeout_seconds: float = Field(default=45.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_seconds: float = Field(default=30.0, gt=0)
    egress_allowlist: tuple[str, ...] = ("api.anthropic.com", "api.openai.com", "api.github.com")


class DataOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EAP_DATAOPS_", extra="ignore")

    vector_store: Literal["memory", "pgvector"] = "memory"
    postgres_dsn: SecretStr | None = None
    embedding_provider: Literal["deterministic", "openai"] = "deterministic"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=256, ge=32, le=4096)
    chunk_max_tokens: int = Field(default=512, ge=64)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"

    @model_validator(mode="after")
    def _pgvector_needs_a_dsn(self) -> DataOpsSettings:
        if self.vector_store == "pgvector" and self.postgres_dsn is None:
            raise ValueError("EAP_DATAOPS_POSTGRES_DSN is required when vector_store=pgvector")
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_max_tokens")
        return self


class LLMOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EAP_LLMOPS_", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    default_route: str = "balanced"
    daily_tenant_budget_usd: float = Field(default=25.0, ge=0)
    max_output_tokens: int = Field(default=2048, ge=1)


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EAP_OBS_", extra="ignore")

    service_name: str = "enterprise-agent-platform"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    otlp_endpoint: str | None = None
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    capture_prompt_content: bool = False
    """Off by default. Prompt bodies routinely contain customer data, and a trace backend
    is rarely in the same compliance boundary as the primary datastore."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EAP_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Environment = "local"
    service_version: str = "0.1.0"

    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    secops: SecOpsSettings = Field(default_factory=SecOpsSettings)
    netops: NetOpsSettings = Field(default_factory=NetOpsSettings)
    dataops: DataOpsSettings = Field(default_factory=DataOpsSettings)
    llmops: LLMOpsSettings = Field(default_factory=LLMOpsSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @field_validator("environment", mode="before")
    @classmethod
    def _normalise(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def is_production(self) -> bool:
        return self.environment in ("staging", "prod")

    @model_validator(mode="after")
    def _production_refuses_development_shortcuts(self) -> Settings:
        """The checks that stop a convenient local default from reaching a real deployment."""
        if not self.is_production:
            return self

        problems: list[str] = []
        if not self.identity.jwks_url:
            problems.append("identity.jwks_url must be set (HS256 dev signing is not permitted)")
        if self.identity.dev_signing_key.get_secret_value():
            problems.append("identity.dev_signing_key must be empty")
        if self.dataops.vector_store == "memory":
            problems.append("dataops.vector_store=memory loses all knowledge on restart")
        if self.dataops.embedding_provider == "deterministic":
            problems.append("dataops.embedding_provider=deterministic carries no semantics")
        if not self.secops.block_on_injection:
            problems.append("secops.block_on_injection must stay enabled")
        if self.observability.capture_prompt_content:
            problems.append("observability.capture_prompt_content exports prompt bodies")
        if self.secops.audit_sink == "memory":
            problems.append("secops.audit_sink=memory is not durable")

        if problems:
            raise ConfigurationError(
                f"configuration is not safe for environment={self.environment}",
                problems=problems,
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so that config is read once, not per request."""
    return Settings()
