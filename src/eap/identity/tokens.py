"""Token verification.

Turns a bearer token into a :class:`Principal`, or refuses. Two modes:

* **Production** verifies RS256/ES256 signatures against the identity provider's JWKS,
  fetched over HTTPS and cached. Keys are selected by ``kid``.
* **Local** verifies HS256 against a configured development key. ``Settings`` refuses to
  start in staging or prod if this mode is configured, so the shortcut cannot escape a
  laptop.

Both modes verify issuer, audience and expiry, and both require the custom ``tenant``
claim. A token without a tenant is rejected rather than defaulted, because a default
tenant is a cross-tenant data leak waiting for the first misconfigured client.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient

from eap.identity.models import Principal, PrincipalType
from eap.platform.config import IdentitySettings
from eap.platform.errors import AuthenticationError, ConfigurationError

_ASYMMETRIC_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384"]


class JWKSCache:
    """Caches the provider's signing keys.

    Refetching JWKS per request would put the identity provider in the hot path of every
    call; never refetching means a key rotation locks everyone out. Cache with a TTL, and
    force one refresh when an unknown ``kid`` appears — that is exactly the signal that a
    rotation happened.
    """

    def __init__(self, jwks_url: str, *, ttl_seconds: int = 600) -> None:
        self._url = jwks_url
        self._ttl = ttl_seconds
        self._client: PyJWKClient | None = None
        self._fetched_at = 0.0

    def _refresh(self) -> PyJWKClient:
        self._client = PyJWKClient(self._url, cache_keys=True, lifespan=self._ttl)
        self._fetched_at = time.monotonic()
        return self._client

    def signing_key(self, token: str) -> Any:
        if self._client is None or (time.monotonic() - self._fetched_at) > self._ttl:
            self._refresh()
        assert self._client is not None
        try:
            return self._client.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientError:
            # Unknown kid: assume rotation, refresh once, then give up.
            try:
                return self._refresh().get_signing_key_from_jwt(token).key
            except (jwt.PyJWKClientError, httpx.HTTPError) as exc:
                raise AuthenticationError("no signing key matches this token") from exc


class TokenVerifier:
    """Verifies a bearer token and maps its claims onto a principal."""

    def __init__(self, settings: IdentitySettings) -> None:
        self._settings = settings
        self._jwks: JWKSCache | None = None
        if settings.jwks_url:
            self._jwks = JWKSCache(settings.jwks_url, ttl_seconds=settings.jwks_cache_seconds)
        elif not settings.dev_signing_key.get_secret_value():
            raise ConfigurationError(
                "identity requires either jwks_url (production) or dev_signing_key (local)"
            )

    def verify(self, token: str) -> Principal:
        claims = self._decode(token)
        return self._to_principal(claims)

    def _decode(self, token: str) -> dict[str, Any]:
        if not token or token.count(".") != 2:
            raise AuthenticationError("bearer token is not a well-formed JWT")

        if self._jwks is not None:
            key: Any = self._jwks.signing_key(token)
            algorithms = _ASYMMETRIC_ALGORITHMS
        else:
            key = self._settings.dev_signing_key.get_secret_value()
            algorithms = ["HS256"]

        try:
            return jwt.decode(
                token,
                key=key,
                algorithms=algorithms,
                audience=self._settings.audience,
                issuer=self._settings.issuer,
                leeway=self._settings.clock_skew_seconds,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token has expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthenticationError("token audience does not match this gateway") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthenticationError("token issuer is not trusted") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise AuthenticationError(f"token is missing required claim: {exc.claim}") from exc
        except jwt.InvalidTokenError as exc:
            # Deliberately vague to the caller; the detail is logged, not returned.
            raise AuthenticationError("token verification failed") from exc

    def _to_principal(self, claims: dict[str, Any]) -> Principal:
        tenant_id = claims.get("tenant") or claims.get("tid")
        if not tenant_id:
            raise AuthenticationError("token carries no tenant claim")

        raw_type = str(claims.get("typ", "user")).lower()
        try:
            principal_type = PrincipalType(raw_type)
        except ValueError as exc:
            raise AuthenticationError(f"unknown principal type '{raw_type}'") from exc

        return Principal(
            subject=str(claims["sub"]),
            tenant_id=str(tenant_id),
            principal_type=principal_type,
            roles=frozenset(_as_sequence(claims.get("roles"))),
            scopes=frozenset(_as_sequence(claims.get("scope") or claims.get("scp"))),
            email=claims.get("email"),
            display_name=claims.get("name"),
            issued_at=_epoch(claims.get("iat")),
            expires_at=_epoch(claims.get("exp")),
        )


def _as_sequence(value: object) -> list[str]:
    """Accept both list-valued and space-delimited claims. Providers disagree on this."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.replace(",", " ").split() if part]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value]
    return []


def _epoch(value: object) -> Any:
    if value is None:
        return None
    from datetime import UTC, datetime

    return datetime.fromtimestamp(float(value), tz=UTC)  # type: ignore[arg-type]
