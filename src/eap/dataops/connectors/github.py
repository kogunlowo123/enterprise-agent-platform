"""GitHub repository connector.

Pulls an existing enterprise repository — documentation, ADRs, runbooks, source — into the
knowledge plane so that agents answer from what the organisation has actually written down
rather than from what the base model absorbed during pretraining. It works against
github.com and against GitHub Enterprise Server by changing ``api_url``.

Design notes worth stating, because each one is a bug the naive version has:

**Trees, not crawling.** One call to the git trees API with ``recursive=1`` returns the
whole repository listing. Walking ``/contents`` directory by directory costs one request
per directory and exhausts the rate limit on any repository with real structure.

**Pinned to a commit.** The tree is read at a resolved commit SHA, and every citation is a
permalink to that SHA. Citing a branch produces a link whose content has changed by the
time anyone follows it, which quietly turns a verifiable citation into an unverifiable one.

**Rate limit awareness.** GitHub returns the remaining budget on every response. The
connector reads it and waits when the budget is nearly spent, instead of discovering the
limit by getting a 403 partway through a large repository and losing the run.

**Secret hygiene.** Repositories contain committed credentials more often than anyone would
like. Files matching credential patterns are skipped entirely, and the reason is reported,
so that ingesting a repository does not copy its leaked keys into a vector store that a
model will later quote back.
"""

from __future__ import annotations

import asyncio
import base64
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import httpx

from eap.dataops.connectors.base import SourceDocument, SyncStats
from eap.platform.errors import NotFoundError, ProviderError
from eap.platform.telemetry import get_logger
from eap.secops.guardrails.pii import SensitiveDataDetector

log = get_logger(__name__)

TEXT_EXTENSIONS = frozenset(
    """.md .markdown .rst .txt .adoc .py .ts .tsx .js .jsx .go .rs .java .kt .rb .php .cs
    .sql .sh .bash .yaml .yml .toml .ini .cfg .json .proto .tf .tfvars .gradle .swift
    .c .h .cpp .hpp .scala .clj .ex .exs .r .m .lua""".split()
)

CODE_EXTENSIONS = frozenset(
    """.py .ts .tsx .js .jsx .go .rs .java .kt .rb .php .cs .sql .sh .bash .proto .tf
    .c .h .cpp .hpp .scala .clj .ex .exs .swift .lua .r .m""".split()
)

MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown", ".rst", ".adoc"})


@lru_cache(maxsize=2048)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a path glob with git-style ``**`` semantics.

    ``fnmatch`` is the obvious choice and the wrong one: its ``*`` matches ``/``, so
    ``docs/*`` silently matches ``docs/a/b/c.md``, and it has no concept of ``**`` at all —
    it treats ``**/*`` as "two stars then a slash", which requires a slash to be present and
    therefore excludes every file at the repository root. That failure is invisible, because
    the connector simply reports fewer files rather than erroring.

    Here ``*`` matches within one path segment, ``**`` crosses segments, and ``**/`` also
    matches nothing at all, so ``**/*.md`` covers both ``README.md`` and ``docs/guide.md``.
    """
    parts: list[str] = []
    index = 0
    length = len(pattern)

    while index < length:
        character = pattern[index]
        if character == "*":
            if pattern[index : index + 3] == "**/":
                parts.append("(?:.*/)?")
                index += 3
                continue
            if pattern[index : index + 2] == "**":
                parts.append(".*")
                index += 2
                continue
            parts.append("[^/]*")
        elif character == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(character))
        index += 1

    return re.compile("^" + "".join(parts) + "$")


def path_matches(path: str, pattern: str) -> bool:
    return _compile_glob(pattern).match(path) is not None


# Paths that are never worth embedding: generated output, dependency trees, lockfiles and
# binaries. Ingesting them wastes budget and pollutes retrieval with noise that matches
# everything and means nothing.
# fmt: off
DEFAULT_EXCLUDES = (
    "**/node_modules/**", "**/.git/**", "**/dist/**", "**/build/**", "**/vendor/**",
    "**/.venv/**", "**/__pycache__/**", "**/testdata/**", "**/migrations/**",
    "**/*.min.js", "**/*.min.css", "**/*.lock", "**/*.snap",
    "**/package-lock.json", "**/yarn.lock", "**/poetry.lock", "**/Cargo.lock", "**/go.sum",
    "**/*.svg", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.ico", "**/*.pdf",
    "**/*.zip", "**/*.gz", "**/*.woff", "**/*.woff2",
)
# fmt: on


@dataclass(frozen=True, slots=True)
class GitHubRepoConfig:
    owner: str
    repo: str
    ref: str = "HEAD"
    """Branch, tag or SHA. Resolved to a concrete commit before anything is read."""

    include_globs: tuple[str, ...] = ("**/*",)
    exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDES
    max_file_bytes: int = 400_000
    max_files: int = 5_000
    api_url: str = "https://api.github.com"
    skip_files_with_secrets: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def is_enterprise_server(self) -> bool:
        return not self.api_url.startswith("https://api.github.com")

    def web_base(self) -> str:
        if not self.is_enterprise_server:
            return f"https://github.com/{self.slug}"
        host = self.api_url.removesuffix("/api/v3").rstrip("/")
        return f"{host}/{self.slug}"


class GitHubConnector:
    """Reads a repository at a pinned commit and yields its text files as documents."""

    name = "github"

    def __init__(
        self,
        config: GitHubRepoConfig,
        *,
        token: str | None = None,
        client: httpx.AsyncClient | None = None,
        detector: SensitiveDataDetector | None = None,
    ) -> None:
        self._config = config
        self._token = token
        self._client = client
        self._owns_client = client is None
        self._detector = detector or SensitiveDataDetector(redact=False)
        self._stats = SyncStats()
        self._resolved_sha: str | None = None

    @property
    def stats(self) -> SyncStats:
        return self._stats

    @property
    def resolved_sha(self) -> str | None:
        """The commit everything in this run was read from. Record it to make syncs
        incremental: nothing needs re-reading while this value is unchanged."""
        return self._resolved_sha

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "enterprise-agent-platform",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, **params: Any) -> Any:
        client = await self._http()
        url = f"{self._config.api_url.rstrip('/')}{path}"
        response = await client.get(url, headers=self._headers(), params=params or None)

        await self._respect_rate_limit(response)

        if response.status_code == 404:
            raise NotFoundError(
                f"{self._config.slug}: {path} not found, or the token cannot see it",
                repo=self._config.slug,
            )
        if response.status_code in (401, 403):
            raise ProviderError(
                f"GitHub refused the request ({response.status_code}); check token scopes "
                f"for {self._config.slug}",
                provider="github",
                retryable=False,
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"GitHub returned {response.status_code} for {path}",
                provider="github",
                retryable=response.status_code >= 500,
            )
        return response.json()

    @staticmethod
    async def _respect_rate_limit(response: httpx.Response) -> None:
        """Pause before the budget runs out rather than after.

        A run that trips the limit loses everything it has not yet yielded, so it is
        cheaper to sleep for the reset window than to fail and start the repository again.
        """
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining is None or int(remaining) > 5:
            return
        reset_at = int(response.headers.get("X-RateLimit-Reset", "0"))
        wait = max(0.0, reset_at - datetime.now(UTC).timestamp()) + 1.0
        if wait > 0:
            log.warning("github.rate_limit_pause", seconds=round(wait, 1), remaining=remaining)
            await asyncio.sleep(min(wait, 300.0))

    async def resolve_commit(self) -> str:
        """Turn a branch, tag or partial SHA into the full commit SHA."""
        if self._resolved_sha:
            return self._resolved_sha
        ref = self._config.ref
        path = f"/repos/{self._config.slug}/commits/{'HEAD' if ref == 'HEAD' else ref}"
        payload = await self._get(path)
        self._resolved_sha = str(payload["sha"])
        return self._resolved_sha

    async def list_tree(self) -> list[dict[str, Any]]:
        """The full recursive listing at the resolved commit, already filtered."""
        sha = await self.resolve_commit()
        payload = await self._get(f"/repos/{self._config.slug}/git/trees/{sha}", recursive="1")

        if payload.get("truncated"):
            # GitHub caps the tree response. Silently ingesting a partial repository would
            # produce a knowledge base with invisible holes, so this is worth saying loudly.
            log.warning(
                "github.tree_truncated",
                repo=self._config.slug,
                detail="repository exceeds the tree API limit; narrow include_globs",
            )

        return [entry for entry in payload.get("tree", []) if self._wants(entry)]

    def _wants(self, entry: dict[str, Any]) -> bool:
        if entry.get("type") != "blob":
            return False
        path = entry.get("path", "")

        if not any(path_matches(path, pattern) for pattern in self._config.include_globs):
            return self._skip(path, "not_included")
        if any(path_matches(path, pattern) for pattern in self._config.exclude_globs):
            return self._skip(path, "excluded")
        if not any(path.lower().endswith(ext) for ext in TEXT_EXTENSIONS):
            return self._skip(path, "not_text")
        if int(entry.get("size", 0)) > self._config.max_file_bytes:
            return self._skip(path, "too_large")
        return True

    def _skip(self, _path: str, reason: str) -> bool:
        self._stats = SyncStats(
            documents=self._stats.documents,
            bytes_read=self._stats.bytes_read,
            skipped=self._stats.skipped + 1,
            skip_reasons={
                **self._stats.skip_reasons,
                reason: self._stats.skip_reasons.get(reason, 0) + 1,
            },
        )
        return False

    async def fetch(self) -> AsyncIterator[SourceDocument]:
        """Yield each qualifying file as a document pinned to the resolved commit."""
        sha = await self.resolve_commit()
        entries = await self.list_tree()

        log.info(
            "github.sync_started",
            repo=self._config.slug,
            commit=sha[:12],
            candidate_files=len(entries),
            skipped=self._stats.skipped,
        )

        for entry in entries[: self._config.max_files]:
            path = entry["path"]
            try:
                text = await self._read_blob(entry["sha"])
            except (ProviderError, NotFoundError) as exc:
                log.warning("github.blob_failed", repo=self._config.slug, path=path, error=str(exc))
                self._skip(path, "read_failed")
                continue

            if text is None:
                self._skip(path, "binary")
                continue

            if self._config.skip_files_with_secrets and self._detector.contains_credentials(text):
                log.warning("github.secret_detected", repo=self._config.slug, path=path)
                self._skip(path, "contains_credentials")
                continue

            self._stats = SyncStats(
                documents=self._stats.documents + 1,
                bytes_read=self._stats.bytes_read + len(text),
                skipped=self._stats.skipped,
                skip_reasons=self._stats.skip_reasons,
            )

            yield SourceDocument(
                source_id=f"github:{self._config.slug}:{path}",
                title=path,
                text=text,
                uri=f"{self._config.web_base()}/blob/{sha}/{path}",
                content_type=_content_type_for(path),
                revision=sha,
                modified_at=None,
                metadata={
                    "connector": self.name,
                    "repo": self._config.slug,
                    "path": path,
                    "commit": sha,
                    "ref": self._config.ref,
                    **self._config.metadata,
                },
            )

        log.info(
            "github.sync_completed",
            repo=self._config.slug,
            commit=sha[:12],
            documents=self._stats.documents,
            skipped=self._stats.skipped,
            skip_reasons=self._stats.skip_reasons,
        )

    async def _read_blob(self, blob_sha: str) -> str | None:
        """Fetch and decode one blob. Returns ``None`` when the content is not text."""
        payload = await self._get(f"/repos/{self._config.slug}/git/blobs/{blob_sha}")
        if payload.get("encoding") != "base64":
            return str(payload.get("content", "")) or None
        raw = base64.b64decode(payload.get("content", ""))
        if b"\x00" in raw[:8192]:
            return None  # NUL bytes in the head: binary despite the extension.
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")


def _content_type_for(path: str) -> str:
    lowered = path.lower()
    if any(lowered.endswith(ext) for ext in MARKDOWN_EXTENSIONS):
        return "markdown"
    if any(lowered.endswith(ext) for ext in CODE_EXTENSIONS):
        return "code"
    return "text"
