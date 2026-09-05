"""Vector storage.

Every read and write takes a ``tenant_id``, and the in-memory implementation partitions by
it rather than filtering after the fact. That distinction matters: a store that searches
globally and then drops foreign results will leak the moment someone adds a code path that
forgets the filter, whereas a partitioned store has no expressible way to reach another
tenant's vectors.

Two implementations ship. :class:`InMemoryVectorStore` does exact brute-force cosine search
— correct, dependency-free, and fast enough for the tens of thousands of chunks a test
suite or a single-team pilot deals with. :class:`PgVectorStore` targets Postgres with the
pgvector extension for anything real.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from eap.dataops.embeddings import Vector, cosine_similarity
from eap.platform.errors import TenantIsolationError


@dataclass(frozen=True, slots=True)
class Document:
    """A stored chunk with its vector and provenance."""

    id: str
    tenant_id: str
    text: str
    vector: Vector
    source_id: str
    citation: str
    corpus: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    document: Document
    score: float
    retriever: str = "vector"

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def citation(self) -> str:
        return self.document.citation


class VectorStore(Protocol):
    async def upsert(self, tenant_id: str, documents: Sequence[Document]) -> int: ...

    async def search(
        self,
        tenant_id: str,
        query_vector: Vector,
        *,
        top_k: int = 8,
        corpus: str | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredDocument]: ...

    async def delete_source(self, tenant_id: str, source_id: str) -> int: ...

    async def count(self, tenant_id: str) -> int: ...


class InMemoryVectorStore:
    """Exact cosine search over a per-tenant partition."""

    def __init__(self) -> None:
        self._partitions: dict[str, dict[str, Document]] = defaultdict(dict)

    async def upsert(self, tenant_id: str, documents: Sequence[Document]) -> int:
        partition = self._partitions[tenant_id]
        for document in documents:
            if document.tenant_id != tenant_id:
                raise TenantIsolationError(
                    "document tenant does not match the partition being written",
                    document_tenant=document.tenant_id,
                    partition_tenant=tenant_id,
                )
            partition[document.id] = document
        return len(documents)

    async def search(
        self,
        tenant_id: str,
        query_vector: Vector,
        *,
        top_k: int = 8,
        corpus: str | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredDocument]:
        partition = self._partitions.get(tenant_id)
        if not partition:
            return []

        scored: list[ScoredDocument] = []
        for document in partition.values():
            if corpus is not None and document.corpus != corpus:
                continue
            score = cosine_similarity(query_vector, document.vector)
            if score >= min_score:
                scored.append(ScoredDocument(document=document, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    async def delete_source(self, tenant_id: str, source_id: str) -> int:
        partition = self._partitions.get(tenant_id)
        if not partition:
            return 0
        doomed = [k for k, d in partition.items() if d.source_id == source_id]
        for key in doomed:
            del partition[key]
        return len(doomed)

    async def count(self, tenant_id: str) -> int:
        return len(self._partitions.get(tenant_id, {}))

    async def all_documents(self, tenant_id: str) -> list[Document]:
        """Used by the lexical index at build time. Not part of the protocol."""
        return list(self._partitions.get(tenant_id, {}).values())


class PgVectorStore:
    """Postgres + pgvector.

    Isolation is enforced twice: every statement carries a ``tenant_id`` predicate, and the
    schema below enables row-level security so that a connection using the application role
    physically cannot read another tenant's rows even if a query is written wrongly.
    Defence in depth, because the application-layer predicate is one typo from absent.
    """

    SCHEMA = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS knowledge_chunk (
        id           TEXT PRIMARY KEY,
        tenant_id    TEXT NOT NULL,
        corpus       TEXT NOT NULL DEFAULT 'default',
        source_id    TEXT NOT NULL,
        citation     TEXT NOT NULL,
        text         TEXT NOT NULL,
        embedding    VECTOR(%(dimensions)s) NOT NULL,
        metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS knowledge_chunk_tenant_corpus_idx
        ON knowledge_chunk (tenant_id, corpus);
    CREATE INDEX IF NOT EXISTS knowledge_chunk_source_idx
        ON knowledge_chunk (tenant_id, source_id);

    -- Cosine distance index. Build it after bulk load: HNSW on an empty table then filled
    -- row by row is markedly slower to construct than one built over existing data.
    CREATE INDEX IF NOT EXISTS knowledge_chunk_embedding_idx
        ON knowledge_chunk USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

    ALTER TABLE knowledge_chunk ENABLE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS knowledge_chunk_tenant_isolation ON knowledge_chunk;
    CREATE POLICY knowledge_chunk_tenant_isolation ON knowledge_chunk
        USING (tenant_id = current_setting('eap.tenant_id', TRUE));
    """

    def __init__(self, pool: Any, *, dimensions: int) -> None:
        self._pool = pool
        self._dimensions = dimensions

    async def upsert(self, tenant_id: str, documents: Sequence[Document]) -> int:
        if not documents:
            return 0
        rows = [
            (
                d.id,
                tenant_id,
                d.corpus,
                d.source_id,
                d.citation,
                d.text,
                _to_pgvector(d.vector),
                d.metadata,
            )
            for d in documents
        ]
        async with self._pool.acquire() as connection:
            await _bind_tenant(connection, tenant_id)
            await connection.executemany(
                """
                INSERT INTO knowledge_chunk
                    (id, tenant_id, corpus, source_id, citation, text, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    citation = EXCLUDED.citation,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )
        return len(rows)

    async def search(
        self,
        tenant_id: str,
        query_vector: Vector,
        *,
        top_k: int = 8,
        corpus: str | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredDocument]:
        async with self._pool.acquire() as connection:
            await _bind_tenant(connection, tenant_id)
            records = await connection.fetch(
                """
                SELECT id, corpus, source_id, citation, text, metadata,
                       1 - (embedding <=> $2::vector) AS score
                FROM knowledge_chunk
                WHERE tenant_id = $1
                  AND ($3::text IS NULL OR corpus = $3)
                  AND 1 - (embedding <=> $2::vector) >= $4
                ORDER BY embedding <=> $2::vector
                LIMIT $5
                """,
                tenant_id,
                _to_pgvector(query_vector),
                corpus,
                min_score,
                top_k,
            )
        return [
            ScoredDocument(
                document=Document(
                    id=record["id"],
                    tenant_id=tenant_id,
                    text=record["text"],
                    # Not returned: large payload, and nothing downstream reads it.
                    vector=[],
                    source_id=record["source_id"],
                    citation=record["citation"],
                    corpus=record["corpus"],
                    metadata=record["metadata"] or {},
                ),
                score=float(record["score"]),
            )
            for record in records
        ]

    async def delete_source(self, tenant_id: str, source_id: str) -> int:
        async with self._pool.acquire() as connection:
            await _bind_tenant(connection, tenant_id)
            status = await connection.execute(
                "DELETE FROM knowledge_chunk WHERE tenant_id = $1 AND source_id = $2",
                tenant_id,
                source_id,
            )
        return int(status.rsplit(" ", 1)[-1] or 0)

    async def count(self, tenant_id: str) -> int:
        async with self._pool.acquire() as connection:
            await _bind_tenant(connection, tenant_id)
            return int(
                await connection.fetchval(
                    "SELECT count(*) FROM knowledge_chunk WHERE tenant_id = $1", tenant_id
                )
            )


async def _bind_tenant(connection: Any, tenant_id: str) -> None:
    """Set the session variable that the row-level-security policy reads."""
    await connection.execute("SELECT set_config('eap.tenant_id', $1, TRUE)", tenant_id)


def _to_pgvector(vector: Vector) -> str:
    return "[" + ",".join(f"{component:.6f}" for component in vector) + "]"
