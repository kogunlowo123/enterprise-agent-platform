"""GitHub connector.

Driven against an httpx MockTransport that serves the real GitHub response shapes, so the
request construction, pagination-free tree walk, base64 decoding, filtering, rate-limit
handling and secret quarantine are all exercised without a network or a token.
"""

from __future__ import annotations

import base64

import httpx
import pytest
from tests.synthetic_credentials import AWS_ACCESS_KEY

from eap.dataops.connectors.github import GitHubConnector, GitHubRepoConfig
from eap.platform.errors import NotFoundError, ProviderError

COMMIT_SHA = "4f2a1c9e8b7d6a5c4b3a2918f7e6d5c4b3a29187"

FILES: dict[str, tuple[str, str]] = {
    # blob sha -> (path, content)
    "blob-readme": ("README.md", "# Acme Platform\n\nDeployments run weekdays 09:00-16:00 UTC.\n"),
    "blob-runbook": (
        "docs/runbook.md",
        "# Runbook\n\nRollback with `make rollback ENV=prod`.\n",
    ),
    "blob-loader": ("src/loader.py", "def load(path):\n    return open(path).read()\n"),
    "blob-secret": (
        "config/credentials.md",
        f"# Credentials\n\nProduction key: {AWS_ACCESS_KEY}\n",
    ),
    "blob-binary": ("assets/logo.png", "\x00\x01binary"),
}

TREE_ENTRIES = [
    {"path": "README.md", "type": "blob", "sha": "blob-readme", "size": 70},
    {"path": "docs/runbook.md", "type": "blob", "sha": "blob-runbook", "size": 50},
    {"path": "src/loader.py", "type": "blob", "sha": "blob-loader", "size": 45},
    {"path": "config/credentials.md", "type": "blob", "sha": "blob-secret", "size": 60},
    {"path": "assets/logo.png", "type": "blob", "sha": "blob-binary", "size": 9},
    {"path": "docs/architecture.drawio", "type": "blob", "sha": "blob-drawio", "size": 40},
    {"path": "docs", "type": "tree", "sha": "tree-docs"},
    {"path": "node_modules/left-pad/index.js", "type": "blob", "sha": "blob-dep", "size": 100},
    {"path": "package-lock.json", "type": "blob", "sha": "blob-lock", "size": 900_000},
    {"path": "huge.md", "type": "blob", "sha": "blob-huge", "size": 5_000_000},
]


def _handler(
    *,
    truncated: bool = False,
    rate_remaining: str = "4999",
    fail_status: int | None = None,
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        headers = {
            "X-RateLimit-Remaining": rate_remaining,
            "X-RateLimit-Reset": "0",
        }
        path = request.url.path

        if fail_status is not None:
            return httpx.Response(fail_status, json={"message": "nope"}, headers=headers)

        if "/commits/" in path:
            return httpx.Response(200, json={"sha": COMMIT_SHA}, headers=headers)

        if "/git/trees/" in path:
            return httpx.Response(
                200,
                json={"sha": COMMIT_SHA, "tree": TREE_ENTRIES, "truncated": truncated},
                headers=headers,
            )

        if "/git/blobs/" in path:
            blob_sha = path.rsplit("/", 1)[-1]
            if blob_sha not in FILES:
                return httpx.Response(404, json={"message": "Not Found"}, headers=headers)
            _, content = FILES[blob_sha]
            return httpx.Response(
                200,
                json={
                    "sha": blob_sha,
                    "encoding": "base64",
                    "content": base64.b64encode(
                        content.encode("utf-8", "surrogateescape")
                    ).decode(),
                },
                headers=headers,
            )

        return httpx.Response(404, json={"message": "Not Found"}, headers=headers)

    return httpx.MockTransport(handle)


@pytest.fixture
def config() -> GitHubRepoConfig:
    return GitHubRepoConfig(owner="acme", repo="platform", ref="main", max_file_bytes=400_000)


async def _collect(connector: GitHubConnector) -> list:
    return [document async for document in connector.fetch()]


class TestTreeWalk:
    async def test_a_ref_is_resolved_to_a_full_commit_sha(self, config: GitHubRepoConfig) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(config, client=client)
        assert await connector.resolve_commit() == COMMIT_SHA
        await client.aclose()

    async def test_the_resolved_sha_is_cached_for_the_run(self, config: GitHubRepoConfig) -> None:
        calls = 0

        def handle(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"sha": COMMIT_SHA})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        connector = GitHubConnector(config, client=client)
        await connector.resolve_commit()
        await connector.resolve_commit()
        assert calls == 1
        await client.aclose()

    async def test_only_qualifying_text_files_are_listed(self, config: GitHubRepoConfig) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(config, client=client)
        paths = {entry["path"] for entry in await connector.list_tree()}

        assert paths == {
            "README.md",
            "docs/runbook.md",
            "src/loader.py",
            "config/credentials.md",
        }
        await client.aclose()

    async def test_skip_reasons_are_reported(self, config: GitHubRepoConfig) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(config, client=client)
        await connector.list_tree()

        reasons = connector.stats.skip_reasons
        # node_modules, package-lock.json and logo.png all match the default exclude set.
        assert reasons["excluded"] >= 3
        # .drawio is neither excluded nor a known text extension.
        assert reasons["not_text"] == 1
        assert reasons["too_large"] == 1  # huge.md
        await client.aclose()

    async def test_include_globs_narrow_the_listing(self) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(
            GitHubRepoConfig(owner="acme", repo="platform", include_globs=("docs/*",)),
            client=client,
        )
        paths = {entry["path"] for entry in await connector.list_tree()}
        assert paths == {"docs/runbook.md"}
        await client.aclose()


class TestDocuments:
    async def test_documents_carry_a_commit_pinned_permalink(
        self, config: GitHubRepoConfig
    ) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(config, client=client)
        documents = await _collect(connector)

        readme = next(d for d in documents if d.title == "README.md")
        assert readme.uri == f"https://github.com/acme/platform/blob/{COMMIT_SHA}/README.md"
        assert readme.revision == COMMIT_SHA
        assert readme.source_id == "github:acme/platform:README.md"
        await client.aclose()

    async def test_content_type_is_derived_from_the_extension(
        self, config: GitHubRepoConfig
    ) -> None:
        client = httpx.AsyncClient(transport=_handler())
        documents = await _collect(GitHubConnector(config, client=client))
        by_title = {document.title: document for document in documents}

        assert by_title["README.md"].content_type == "markdown"
        assert by_title["src/loader.py"].content_type == "code"
        await client.aclose()

    async def test_a_file_containing_a_credential_is_skipped(
        self, config: GitHubRepoConfig
    ) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(config, client=client)
        documents = await _collect(connector)

        assert "config/credentials.md" not in {d.title for d in documents}
        assert connector.stats.skip_reasons["contains_credentials"] == 1
        await client.aclose()

    async def test_secret_skipping_can_be_disabled_deliberately(self) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(
            GitHubRepoConfig(owner="acme", repo="platform", skip_files_with_secrets=False),
            client=client,
        )
        documents = await _collect(connector)
        assert "config/credentials.md" in {d.title for d in documents}
        await client.aclose()

    async def test_max_files_caps_the_run(self, config: GitHubRepoConfig) -> None:
        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(
            GitHubRepoConfig(owner="acme", repo="platform", max_files=1), client=client
        )
        assert len(await _collect(connector)) == 1
        await client.aclose()

    async def test_a_missing_blob_is_skipped_rather_than_failing_the_run(
        self, config: GitHubRepoConfig
    ) -> None:
        entries = [
            *TREE_ENTRIES,
            {"path": "gone.md", "type": "blob", "sha": "blob-missing", "size": 10},
        ]

        def handle(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if "/commits/" in path:
                return httpx.Response(200, json={"sha": COMMIT_SHA})
            if "/git/trees/" in path:
                return httpx.Response(200, json={"tree": entries, "truncated": False})
            blob_sha = path.rsplit("/", 1)[-1]
            if blob_sha in FILES:
                _, content = FILES[blob_sha]
                return httpx.Response(
                    200,
                    json={
                        "encoding": "base64",
                        "content": base64.b64encode(
                            content.encode("utf-8", "surrogateescape")
                        ).decode(),
                    },
                )
            return httpx.Response(404, json={"message": "Not Found"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        connector = GitHubConnector(config, client=client)
        documents = await _collect(connector)

        assert "gone.md" not in {d.title for d in documents}
        assert connector.stats.skip_reasons.get("read_failed") == 1
        await client.aclose()


class TestErrorHandling:
    async def test_a_404_on_the_repository_raises_not_found(self, config: GitHubRepoConfig) -> None:
        client = httpx.AsyncClient(transport=_handler(fail_status=404))
        with pytest.raises(NotFoundError):
            await GitHubConnector(config, client=client).resolve_commit()
        await client.aclose()

    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_failures_are_not_retryable(
        self, config: GitHubRepoConfig, status: int
    ) -> None:
        client = httpx.AsyncClient(transport=_handler(fail_status=status))
        with pytest.raises(ProviderError) as exc:
            await GitHubConnector(config, client=client).resolve_commit()
        assert exc.value.retryable is False
        await client.aclose()

    async def test_a_server_error_is_retryable(self, config: GitHubRepoConfig) -> None:
        client = httpx.AsyncClient(transport=_handler(fail_status=503))
        with pytest.raises(ProviderError) as exc:
            await GitHubConnector(config, client=client).resolve_commit()
        assert exc.value.retryable is True
        await client.aclose()

    async def test_a_truncated_tree_still_yields_what_it_can(
        self, config: GitHubRepoConfig
    ) -> None:
        client = httpx.AsyncClient(transport=_handler(truncated=True))
        connector = GitHubConnector(config, client=client)
        assert len(await connector.list_tree()) > 0
        await client.aclose()


class TestEnterpriseServer:
    def test_enterprise_server_web_urls_are_derived_from_the_api_url(self) -> None:
        config = GitHubRepoConfig(
            owner="acme", repo="platform", api_url="https://github.acme.internal/api/v3"
        )
        assert config.is_enterprise_server
        assert config.web_base() == "https://github.acme.internal/acme/platform"

    def test_github_dot_com_web_urls(self) -> None:
        config = GitHubRepoConfig(owner="acme", repo="platform")
        assert not config.is_enterprise_server
        assert config.web_base() == "https://github.com/acme/platform"

    async def test_the_token_is_sent_as_a_bearer_header(self, config: GitHubRepoConfig) -> None:
        seen: dict[str, str] = {}

        def handle(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json={"sha": COMMIT_SHA})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        await GitHubConnector(config, token="ghs_example", client=client).resolve_commit()

        assert seen["authorization"] == "Bearer ghs_example"
        assert seen["x-github-api-version"] == "2022-11-28"
        await client.aclose()

    async def test_no_authorization_header_when_unauthenticated(
        self, config: GitHubRepoConfig
    ) -> None:
        seen: dict[str, str] = {}

        def handle(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json={"sha": COMMIT_SHA})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        await GitHubConnector(config, client=client).resolve_commit()
        assert "authorization" not in seen
        await client.aclose()


class TestIngestionIntegration:
    async def test_a_repository_becomes_searchable_knowledge(
        self, config: GitHubRepoConfig, ingestion, store, lexical, embedder
    ) -> None:
        from eap.dataops.retrieval import HybridRetriever

        client = httpx.AsyncClient(transport=_handler())
        connector = GitHubConnector(config, client=client)

        report = await ingestion.ingest(connector, tenant_id="acme", corpus="engineering")
        await client.aclose()

        assert report.chunks_written > 0
        lexical.build("acme", await store.all_documents("acme"))

        retriever = HybridRetriever(store=store, embedder=embedder, lexical_index=lexical)
        result = await retriever.retrieve("rollback", tenant_id="acme", top_k=3)

        assert result.documents
        assert any("runbook" in scored.citation for scored in result.documents)
        assert all(COMMIT_SHA in scored.citation for scored in result.documents)

    async def test_the_credential_file_never_reaches_the_store(
        self, config: GitHubRepoConfig, ingestion, store
    ) -> None:
        client = httpx.AsyncClient(transport=_handler())
        await ingestion.ingest(GitHubConnector(config, client=client), tenant_id="acme")
        await client.aclose()

        documents = await store.all_documents("acme")
        assert not any(AWS_ACCESS_KEY in document.text for document in documents)
