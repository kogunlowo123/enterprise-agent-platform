"""Embedding providers.

The platform never imports an embedding vendor directly. Everything downstream depends on
:class:`Embedder`, so swapping OpenAI for Azure, Cohere or a self-hosted model is a
configuration change rather than a refactor.

:class:`DeterministicEmbedder` is a hashing embedder used for local development, tests and
CI. It is a genuine embedding function — hashed character n-grams projected onto a fixed
dimension and L2-normalised — so lexically similar strings land near each other and the
retrieval, ranking and evaluation code exercises real vector arithmetic without a network
call or a model download. What it does not have is semantics: it cannot tell that
"physician" and "doctor" are related. ``Settings`` refuses to start with it in staging or
production for exactly that reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from eap.platform.errors import ConfigurationError, ProviderError

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    model: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Dot product for unit vectors; falls back to full cosine when they are not."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def l2_normalise(vector: Vector) -> Vector:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


class DeterministicEmbedder:
    """Hashed character-n-gram embedder. Local and CI only.

    Character 3-grams rather than words, so that morphological variants ("authorise",
    "authorisation") share most of their features. Each n-gram is hashed to a dimension and
    signed by a second hash bit, which keeps collisions from systematically inflating
    similarity — a collision is as likely to subtract as to add.
    """

    def __init__(self, *, dimensions: int = 256, model: str = "deterministic-hash-v1") -> None:
        if dimensions < 32:
            raise ConfigurationError("embedding dimensions must be at least 32")
        self.dimensions = dimensions
        self.model = model

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        normalised = re.sub(r"\s+", " ", text.lower().strip())
        if not normalised:
            return vector

        tokens = normalised.split(" ")
        for token in tokens:
            self._accumulate(vector, token, weight=1.0)
            padded = f" {token} "
            for i in range(len(padded) - 2):
                self._accumulate(vector, padded[i : i + 3], weight=0.5)

        return l2_normalise(vector)

    def _accumulate(self, vector: Vector, feature: str, *, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight


class OpenAIEmbedder:
    """OpenAI and Azure OpenAI embeddings over their shared REST shape."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
        max_batch: int = 128,
    ) -> None:
        if not api_key:
            raise ConfigurationError("OpenAI embeddings require an API key")
        self.model = model
        self.dimensions = dimensions
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._max_batch = max_batch

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        batches = [texts[i : i + self._max_batch] for i in range(0, len(texts), self._max_batch)]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        return [vector for batch in results for vector in batch]

    async def _embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        payload = {"input": list(texts), "model": self.model, "dimensions": self.dimensions}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = self._client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await client.post(
                f"{self._base_url}/embeddings", json=payload, headers=headers
            )
            if response.status_code >= 400:
                raise ProviderError(
                    f"embedding request failed with {response.status_code}",
                    provider="openai",
                    retryable=response.status_code in (429, 500, 502, 503, 504),
                )
            data = response.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        except httpx.HTTPError as exc:
            raise ProviderError(f"embedding transport failure: {exc}", provider="openai") from exc
        finally:
            if self._client is None:
                await client.aclose()


def build_embedder(
    provider: str, *, dimensions: int, model: str, api_key: str | None = None
) -> Embedder:
    if provider == "openai":
        if not api_key:
            raise ConfigurationError("embedding_provider=openai requires EAP_LLMOPS_OPENAI_API_KEY")
        return OpenAIEmbedder(api_key=api_key, model=model, dimensions=dimensions)
    return DeterministicEmbedder(dimensions=dimensions)
