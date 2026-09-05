"""Command line interface.

Operational commands that need to run without the HTTP surface: ingest a repository, verify
the audit chain, inspect routing, mint a development token, run an evaluation suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta

from eap.bootstrap import build_platform
from eap.dataops.connectors.github import GitHubConnector, GitHubRepoConfig
from eap.platform.config import get_settings


def _add_ingest(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("ingest-github", help="ingest a GitHub repository")
    parser.add_argument("repository", help="owner/repo")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--corpus", default="default")
    parser.add_argument("--include", action="append", default=None, help="glob, repeatable")
    parser.add_argument("--max-files", type=int, default=2000)
    parser.add_argument("--api-url", default=None)


def _add_search(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("search", help="query the knowledge base")
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--top-k", type=int, default=8)


def _add_token(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("dev-token", help="mint a local HS256 token")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--subject", default="dev-user")
    parser.add_argument("--roles", default="agent.operator")
    parser.add_argument("--minutes", type=int, default=60)


async def _ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    platform = build_platform(settings)
    owner, _, repo = args.repository.partition("/")
    if not repo:
        print("repository must be given as owner/repo", file=sys.stderr)
        return 2

    token = settings.dataops.github_token
    connector = GitHubConnector(
        GitHubRepoConfig(
            owner=owner,
            repo=repo,
            ref=args.ref,
            include_globs=tuple(args.include) if args.include else ("**/*",),
            max_files=args.max_files,
            api_url=args.api_url or settings.dataops.github_api_url,
        ),
        token=token.get_secret_value() if token else None,
    )

    try:
        report = await platform.ingestion.ingest(
            connector, tenant_id=args.tenant, corpus=args.corpus, actor="cli"
        )
    finally:
        await connector.aclose()

    indexed = await platform.reindex_lexical(args.tenant)
    print(
        json.dumps(
            {
                **report.summary(),
                "commit": connector.resolved_sha,
                "lexical_documents": indexed,
                "skipped": connector.stats.skip_reasons,
            },
            indent=2,
        )
    )
    return 0 if report.succeeded else 1


async def _search(args: argparse.Namespace) -> int:
    platform = build_platform()
    await platform.reindex_lexical(args.tenant)
    result = await platform.retriever.retrieve(args.query, tenant_id=args.tenant, top_k=args.top_k)
    if not result.documents:
        print("no results", file=sys.stderr)
        return 1
    for position, document in enumerate(result.documents, start=1):
        print(f"[{position}] {document.score:.4f} ({document.retriever}) {document.citation}")
        print(f"    {document.text[:200].replace(chr(10), ' ')}")
    return 0


def _verify_audit() -> int:
    platform = build_platform()
    result = platform.audit.verify()
    print(
        json.dumps(
            {
                "valid": result.valid,
                "records_checked": result.records_checked,
                "head_hash": result.head_hash,
                "broken_at": result.broken_at,
                "reason": result.reason,
            },
            indent=2,
        )
    )
    return 0 if result.valid else 1


def _routes() -> int:
    platform = build_platform()
    for name, route in sorted(platform.router.routes.items()):
        print(f"{name}: {route.description}")
        for position, candidate in enumerate(route.candidates):
            marker = "primary " if position == 0 else "fallback"
            print(f"  {marker} {candidate.provider}/{candidate.model} ({candidate.reason})")
    return 0


def _dev_token(args: argparse.Namespace) -> int:
    """Mint an HS256 token for local development.

    Refuses outside a local environment. The signing key it uses is the same one
    ``Settings`` forbids in staging and production, so this command cannot mint anything a
    real deployment would accept.
    """
    import jwt

    settings = get_settings()
    if settings.is_production:
        print("dev-token is not available outside local environments", file=sys.stderr)
        return 2

    key = settings.identity.dev_signing_key.get_secret_value()
    if not key:
        print("set EAP_IDENTITY_DEV_SIGNING_KEY first", file=sys.stderr)
        return 2

    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": args.subject,
            "tenant": args.tenant,
            "typ": "user",
            "roles": args.roles.split(","),
            "iss": settings.identity.issuer,
            "aud": settings.identity.audience,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=args.minutes)).timestamp()),
        },
        key,
        algorithm="HS256",
    )
    print(token)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eap", description="Enterprise Agent Platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_ingest(subparsers)
    _add_search(subparsers)
    _add_token(subparsers)
    subparsers.add_parser("verify-audit", help="verify the audit hash chain")
    subparsers.add_parser("routes", help="show the resolved model routing table")

    args = parser.parse_args(argv)

    if args.command == "ingest-github":
        return asyncio.run(_ingest(args))
    if args.command == "search":
        return asyncio.run(_search(args))
    if args.command == "verify-audit":
        return _verify_audit()
    if args.command == "routes":
        return _routes()
    if args.command == "dev-token":
        return _dev_token(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
