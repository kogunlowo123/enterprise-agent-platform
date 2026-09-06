"""Configuration guards.

The value of these is entirely in the negative cases: local development defaults are chosen
for convenience, and every one of them is unsafe in production. The point of the validator
is that the process refuses to start rather than running with an in-memory audit log and a
shared HMAC signing key because someone forgot an environment variable.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr
from tests.synthetic_credentials import GITHUB_TOKEN

from eap.platform.config import (
    DataOpsSettings,
    IdentitySettings,
    ObservabilitySettings,
    SecOpsSettings,
    Settings,
)
from eap.platform.errors import ConfigurationError


def production(**overrides: object) -> Settings:
    """A production-shaped configuration that is valid unless an override breaks it."""
    base: dict[str, object] = {
        "environment": "prod",
        "identity": IdentitySettings(
            issuer="https://login.acme.test",
            audience="eap",
            jwks_url="https://login.acme.test/.well-known/jwks.json",
            dev_signing_key=SecretStr(""),
        ),
        "secops": SecOpsSettings(audit_sink="file", block_on_injection=True),
        "dataops": DataOpsSettings(
            vector_store="pgvector",
            postgres_dsn=SecretStr("postgresql://user:pw@db/eap"),
            embedding_provider="openai",
        ),
        "observability": ObservabilitySettings(capture_prompt_content=False),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class TestLocalDefaults:
    def test_local_starts_with_no_configuration_at_all(self) -> None:
        settings = Settings(environment="local")
        assert not settings.is_production
        assert settings.dataops.vector_store == "memory"

    def test_the_environment_name_is_normalised(self) -> None:
        # Normalisation has to happen before the production guard runs, or "  PROD  " would
        # slip past it and start with every local default intact.
        assert production(environment="  PROD  ").environment == "prod"
        assert Settings(environment=" LOCAL ").environment == "local"  # type: ignore[arg-type]


class TestProductionGuards:
    def test_a_correctly_configured_production_setup_is_accepted(self) -> None:
        assert production().is_production

    def test_hs256_development_signing_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            production(
                identity=IdentitySettings(
                    jwks_url=None, dev_signing_key=SecretStr("a-shared-secret")
                )
            )
        problems = " ".join(exc.value.details["problems"])
        assert "jwks_url" in problems

    def test_a_leftover_dev_signing_key_is_refused_even_with_jwks(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            production(
                identity=IdentitySettings(
                    jwks_url="https://login.acme.test/jwks",
                    dev_signing_key=SecretStr("left-over-from-local"),
                )
            )
        assert any("dev_signing_key" in p for p in exc.value.details["problems"])

    def test_an_in_memory_vector_store_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            production(dataops=DataOpsSettings(vector_store="memory", embedding_provider="openai"))
        assert any("vector_store" in p for p in exc.value.details["problems"])

    def test_the_deterministic_embedder_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            production(
                dataops=DataOpsSettings(
                    vector_store="pgvector",
                    postgres_dsn=SecretStr("postgresql://db/eap"),
                    embedding_provider="deterministic",
                )
            )
        assert any("embedding_provider" in p for p in exc.value.details["problems"])

    def test_disabling_injection_enforcement_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            production(secops=SecOpsSettings(audit_sink="file", block_on_injection=False))
        assert any("block_on_injection" in p for p in exc.value.details["problems"])

    def test_a_non_durable_audit_sink_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            production(secops=SecOpsSettings(audit_sink="memory"))
        assert any("audit_sink" in p for p in exc.value.details["problems"])

    def test_exporting_prompt_content_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            production(observability=ObservabilitySettings(capture_prompt_content=True))
        assert any("capture_prompt_content" in p for p in exc.value.details["problems"])

    def test_every_problem_is_reported_at_once(self) -> None:
        """One deploy, one list of everything wrong — not one failure per attempt."""
        with pytest.raises(ConfigurationError) as exc:
            production(
                secops=SecOpsSettings(audit_sink="memory", block_on_injection=False),
                observability=ObservabilitySettings(capture_prompt_content=True),
            )
        assert len(exc.value.details["problems"]) >= 3


class TestFieldValidation:
    def test_pgvector_without_a_dsn_is_refused(self) -> None:
        with pytest.raises(ValueError, match="POSTGRES_DSN"):
            DataOpsSettings(vector_store="pgvector", postgres_dsn=None)

    def test_chunk_overlap_must_be_smaller_than_the_chunk(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            DataOpsSettings(chunk_max_tokens=128, chunk_overlap_tokens=128)

    def test_secrets_do_not_appear_in_the_repr(self) -> None:
        settings = DataOpsSettings(github_token=SecretStr(GITHUB_TOKEN))
        assert GITHUB_TOKEN not in repr(settings)
        assert GITHUB_TOKEN not in str(settings.github_token)

    @pytest.mark.parametrize("ratio", [-0.1, 1.1])
    def test_the_trace_sample_ratio_is_bounded(self, ratio: float) -> None:
        with pytest.raises(ValueError):
            ObservabilitySettings(trace_sample_ratio=ratio)
