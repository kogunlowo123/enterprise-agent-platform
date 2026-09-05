"""Identity plane: role resolution, the caller-agent intersection, token verification."""

from __future__ import annotations

import pytest
from tests.conftest import AUDIENCE, DEV_KEY, ISSUER, mint_token

from eap.identity.models import AgentIdentity, Principal, PrincipalType, Role, Tenant
from eap.identity.rbac import (
    PERM_KNOWLEDGE_READ,
    PERM_KNOWLEDGE_WRITE,
    PERM_TENANT_ADMIN,
    PERM_TOOL_EXECUTE_WRITE,
    Authorizer,
    RoleRegistry,
)
from eap.identity.tokens import TokenVerifier
from eap.platform.config import IdentitySettings
from eap.platform.errors import AuthenticationError, AuthorizationError, TenantIsolationError


class TestRoleResolution:
    def test_inheritance_is_followed_transitively(self) -> None:
        registry = RoleRegistry()
        permissions = registry.resolve(frozenset({"platform.admin"}))
        # platform.admin -> agent.operator -> agent.user -> agent.reader
        assert PERM_KNOWLEDGE_READ in permissions
        assert PERM_TOOL_EXECUTE_WRITE in permissions
        assert PERM_TENANT_ADMIN in permissions

    def test_unknown_roles_grant_nothing_rather_than_raising(self) -> None:
        registry = RoleRegistry()
        assert registry.resolve(frozenset({"role.that.was.deleted"})) == frozenset()

    def test_a_cycle_in_role_inheritance_terminates(self) -> None:
        registry = RoleRegistry()
        registry.register(Role("a", frozenset({"p:a"}), inherits=("b",)))
        registry.register(Role("b", frozenset({"p:b"}), inherits=("a",)))
        assert registry.resolve(frozenset({"a"})) == frozenset({"p:a", "p:b"})


class TestWildcardMatching:
    @pytest.mark.parametrize(
        ("granted", "required", "expected"),
        [
            (frozenset({"knowledge:*"}), "knowledge:read", True),
            (frozenset({"knowledge:*"}), "knowledge:write", True),
            (frozenset({"*"}), "anything:at:all", True),
            (frozenset({"tool:execute"}), "tool:execute:write", False),
            (frozenset({"tool:*"}), "tool:execute:write", True),
            (frozenset({"knowledge:read"}), "knowledge:write", False),
        ],
    )
    def test_wildcards_expand_rightwards_only(
        self, granted: frozenset[str], required: str, expected: bool
    ) -> None:
        from eap.identity.models import _matches

        assert _matches(required, granted) is expected


class TestPermissionIntersection:
    """The agent's grant narrows the caller's authority; it can never widen it."""

    def test_caller_permission_absent_from_agent_grant_is_refused(
        self, authorizer: Authorizer, tenant: Tenant
    ) -> None:
        principal = Principal(
            subject="alice",
            tenant_id="acme",
            principal_type=PrincipalType.USER,
            roles=frozenset({"platform.admin"}),
        )
        narrow_agent = AgentIdentity(
            id="narrow",
            name="Narrow",
            mission="read only",
            owner="team",
            granted_permissions=frozenset({PERM_KNOWLEDGE_READ}),
        )
        ctx = authorizer.build_context(principal, tenant, agent=narrow_agent)

        assert ctx.has(PERM_KNOWLEDGE_READ)
        assert not ctx.has(PERM_KNOWLEDGE_WRITE)

        with pytest.raises(AuthorizationError) as exc:
            authorizer.require(ctx, PERM_KNOWLEDGE_WRITE)
        assert exc.value.details["reason"] == "agent_grant_narrower_than_caller"

    def test_agent_grant_cannot_exceed_caller(self, authorizer: Authorizer, tenant: Tenant) -> None:
        reader = Principal(
            subject="bob",
            tenant_id="acme",
            principal_type=PrincipalType.USER,
            roles=frozenset({"agent.reader"}),
        )
        powerful_agent = AgentIdentity(
            id="powerful",
            name="Powerful",
            mission="everything",
            owner="team",
            granted_permissions=frozenset({"*"}),
        )
        ctx = authorizer.build_context(reader, tenant, agent=powerful_agent)
        assert ctx.has(PERM_KNOWLEDGE_READ)
        assert not ctx.has(PERM_KNOWLEDGE_WRITE)

    def test_token_scopes_narrow_role_permissions(
        self, authorizer: Authorizer, tenant: Tenant
    ) -> None:
        delegated = Principal(
            subject="job-runner",
            tenant_id="acme",
            principal_type=PrincipalType.SERVICE,
            roles=frozenset({"platform.admin"}),
            scopes=frozenset({PERM_KNOWLEDGE_READ}),
        )
        ctx = authorizer.build_context(delegated, tenant)
        assert ctx.has(PERM_KNOWLEDGE_READ)
        assert not ctx.has(PERM_TENANT_ADMIN)


class TestTenantIsolation:
    def test_principal_cannot_bind_to_a_foreign_tenant(
        self, authorizer: Authorizer, other_tenant: Tenant, user_principal: Principal
    ) -> None:
        with pytest.raises(TenantIsolationError):
            authorizer.build_context(user_principal, other_tenant)

    def test_suspended_tenant_is_refused(
        self, authorizer: Authorizer, user_principal: Principal
    ) -> None:
        suspended = Tenant(id="acme", name="Acme", active=False)
        with pytest.raises(AuthorizationError):
            authorizer.build_context(user_principal, suspended)

    def test_cross_tenant_resource_access_is_refused(
        self, authorizer: Authorizer, security
    ) -> None:
        authorizer.require_tenant(security, "acme")
        with pytest.raises(TenantIsolationError):
            authorizer.require_tenant(security, "globex")


class TestTokenVerification:
    @pytest.fixture
    def verifier(self) -> TokenVerifier:
        from pydantic import SecretStr

        return TokenVerifier(
            IdentitySettings(
                issuer=ISSUER,
                audience=AUDIENCE,
                jwks_url=None,
                dev_signing_key=SecretStr(DEV_KEY),
            )
        )

    def test_a_valid_token_produces_a_principal(self, verifier: TokenVerifier) -> None:
        principal = verifier.verify(mint_token(roles=("agent.user",)))
        assert principal.subject == "alice"
        assert principal.tenant_id == "acme"
        assert principal.principal_type is PrincipalType.USER
        assert "agent.user" in principal.roles

    def test_expired_token_is_refused(self, verifier: TokenVerifier) -> None:
        with pytest.raises(AuthenticationError, match="expired"):
            verifier.verify(mint_token(expires_in_minutes=-120))

    def test_wrong_audience_is_refused(self, verifier: TokenVerifier) -> None:
        with pytest.raises(AuthenticationError, match="audience"):
            verifier.verify(mint_token(audience="some-other-service"))

    def test_wrong_issuer_is_refused(self, verifier: TokenVerifier) -> None:
        with pytest.raises(AuthenticationError, match="issuer"):
            verifier.verify(mint_token(issuer="https://attacker.test"))

    def test_token_signed_with_another_key_is_refused(self, verifier: TokenVerifier) -> None:
        with pytest.raises(AuthenticationError):
            verifier.verify(mint_token(key="a-different-key-entirely-and-long-enough-for-sha256"))

    def test_token_without_a_tenant_claim_is_refused(self, verifier: TokenVerifier) -> None:
        from datetime import UTC, datetime, timedelta

        import jwt as pyjwt

        now = datetime.now(UTC)
        token = pyjwt.encode(
            {
                "sub": "alice",
                "iss": ISSUER,
                "aud": AUDIENCE,
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(minutes=10)).timestamp()),
            },
            DEV_KEY,
            algorithm="HS256",
        )
        with pytest.raises(AuthenticationError, match="tenant"):
            verifier.verify(token)

    def test_malformed_token_is_refused_before_any_parsing(self, verifier: TokenVerifier) -> None:
        with pytest.raises(AuthenticationError, match="well-formed"):
            verifier.verify("not-a-jwt")

    def test_space_delimited_scope_claim_is_parsed(self, verifier: TokenVerifier) -> None:
        principal = verifier.verify(mint_token(scopes=("knowledge:read", "agent:invoke")))
        assert principal.scopes == frozenset({"knowledge:read", "agent:invoke"})
