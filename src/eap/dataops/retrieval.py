"""Hybrid retrieval.

Pure vector search has a well-known failure mode: it is good at paraphrase and bad at rare
exact tokens. Ask for error code ``ERR_4021`` or a part number and the embedding places it
near a cloud of similar-looking identifiers, none of which is the one you asked for. Pure
lexical search has the mirror failure: it cannot match "how do I reset my password" against
a document titled "Credential recovery procedure".

So both run, and the rankings are combined with Reciprocal Rank Fusion. RRF combines by
*rank* rather than by score, which is what makes it work here: a BM25 score of 14.2 and a
cosine similarity of 0.81 are not comparable quantities, and normalising them requires
per-corpus calibration that drifts. Ranks are always comparable.

The fused list is then diversified with Maximal Marginal Relevance, because the top five
results of a good retriever are frequently five near-copies of the same paragraph. Filling
a context window with the same fact five times is how a system with excellent retrieval
metrics still produces an answer missing the thing the user actually needed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from eap.dataops.embeddings import Embedder, cosine_similarity
from eap.dataops.vectorstore import Document, ScoredDocument, VectorStore
from eap.platform.errors import TenantIsolationError
from eap.platform.telemetry import get_logger

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9_]+")

# Removing these lifts BM25 precision noticeably: they appear in nearly every document, so
# they carry almost no discriminative signal while still consuming term-frequency mass.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i in is it its of on or that the
    to was were what when where which who will with you your do does did can could should
    would this these those there their them then than""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass(slots=True)
class BM25Index:
    """Okapi BM25 over an in-memory corpus.

    ``k1`` controls term-frequency saturation and ``b`` controls length normalisation; 1.5
    and 0.75 are the standard defaults and hold up well without per-corpus tuning.
    """

    k1: float = 1.5
    b: float = 0.75
    _documents: list[Document] = field(default_factory=list)
    _term_frequencies: list[Counter[str]] = field(default_factory=list)
    _lengths: list[int] = field(default_factory=list)
    _document_frequency: Counter[str] = field(default_factory=Counter)
    _average_length: float = 0.0

    def build(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        self._term_frequencies = []
        self._lengths = []
        self._document_frequency = Counter()

        for document in self._documents:
            tokens = tokenize(document.text)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._lengths.append(len(tokens))
            self._document_frequency.update(frequencies.keys())

        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def search(self, query: str, *, top_k: int = 8) -> list[ScoredDocument]:
        if not self._documents:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        total = len(self._documents)
        scored: list[ScoredDocument] = []

        for position, document in enumerate(self._documents):
            frequencies = self._term_frequencies[position]
            length = self._lengths[position] or 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                containing = self._document_frequency[term]
                # Robertson-Sparck-Jones IDF with the +0.5 smoothing that keeps the value
                # positive when a term appears in more than half the corpus.
                idf = math.log(1 + (total - containing + 0.5) / (containing + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._average_length or 1)
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append(ScoredDocument(document=document, score=score, retriever="lexical"))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class LexicalIndex:
    """Tenant-partitioned BM25.

    The vector store partitions by tenant, and the lexical index must do the same or it
    becomes the hole in the middle of an otherwise isolated retrieval path: a single shared
    BM25 index will happily return one tenant's document to another tenant's query, and
    fusion then places it in the model's context. The vector side being correct makes this
    worse rather than better, because the leak appears only in hybrid mode.

    One index per tenant, and the index for a tenant is only ever searched with that
    tenant's id.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._partitions: dict[str, BM25Index] = {}

    def build(self, tenant_id: str, documents: Sequence[Document]) -> int:
        foreign = [d.id for d in documents if d.tenant_id != tenant_id]
        if foreign:
            raise TenantIsolationError(
                "refusing to index documents belonging to another tenant",
                partition_tenant=tenant_id,
                offending_documents=foreign[:5],
            )
        index = BM25Index(k1=self._k1, b=self._b)
        index.build(documents)
        self._partitions[tenant_id] = index
        return len(documents)

    def search(self, tenant_id: str, query: str, *, top_k: int = 8) -> list[ScoredDocument]:
        index = self._partitions.get(tenant_id)
        if index is None:
            return []
        return index.search(query, top_k=top_k)

    def tenants(self) -> tuple[str, ...]:
        return tuple(sorted(self._partitions))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredDocument]], *, k: int = 60, top_k: int = 8
) -> list[ScoredDocument]:
    """Fuse ranked lists by ``sum(1 / (k + rank))``.

    ``k=60`` is the value from the original Cormack et al. formulation. It flattens the
    contribution curve so that being first rather than third in one list is worth less than
    appearing in both lists, which is the behaviour you want: agreement between independent
    retrievers is stronger evidence than confidence from one.
    """
    fused: dict[str, float] = {}
    documents: dict[str, Document] = {}
    sources: dict[str, set[str]] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            key = scored.document.id
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            documents[key] = scored.document
            sources.setdefault(key, set()).add(scored.retriever)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [
        ScoredDocument(
            document=documents[key],
            score=round(score, 6),
            retriever="+".join(sorted(sources[key])),
        )
        for key, score in ordered[:top_k]
    ]


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: Sequence[ScoredDocument],
    *,
    top_k: int = 8,
    diversity: float = 0.3,
) -> list[ScoredDocument]:
    """Greedily select results that are relevant *and* unlike what is already selected.

    ``diversity`` is lambda inverted for readability: 0 keeps pure relevance ordering, 1
    maximises dissimilarity. 0.3 keeps ranking mostly intact while dropping near-duplicates.
    """
    if not candidates or diversity <= 0:
        return list(candidates[:top_k])

    remaining = list(candidates)
    selected: list[ScoredDocument] = []

    while remaining and len(selected) < top_k:
        best_index = 0
        best_value = -math.inf
        for index, candidate in enumerate(remaining):
            relevance = candidate.score
            if selected and candidate.document.vector:
                redundancy = max(
                    cosine_similarity(candidate.document.vector, chosen.document.vector)
                    for chosen in selected
                    if chosen.document.vector
                )
            else:
                redundancy = 0.0
            value = (1 - diversity) * relevance - diversity * redundancy
            if value > best_value:
                best_value, best_index = value, index
        selected.append(remaining.pop(best_index))

    return selected


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    documents: tuple[ScoredDocument, ...]
    strategy: str

    @property
    def citations(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(d.citation for d in self.documents))

    def to_context(self, *, max_chars: int = 12_000) -> str:
        """Render for the prompt, numbered so the model can cite by index.

        Numbering is what makes grounding checkable: an answer that says "[2]" can be
        verified against source 2 mechanically, whereas an answer that paraphrases three
        blended sources cannot be.
        """
        blocks: list[str] = []
        budget = max_chars
        for position, scored in enumerate(self.documents, start=1):
            block = f"[{position}] {scored.citation}\n{scored.text}"
            if len(block) > budget:
                break
            blocks.append(block)
            budget -= len(block)
        return "\n\n".join(blocks)


class HybridRetriever:
    """Vector + BM25, fused with RRF and diversified with MMR."""

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: Embedder,
        lexical_index: LexicalIndex | None = None,
        candidate_multiplier: int = 3,
        diversity: float = 0.3,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._lexical = lexical_index
        self._multiplier = candidate_multiplier
        self._diversity = diversity

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 8,
        corpus: str | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        # Over-fetch before fusion: RRF and MMR both need more candidates than the caller
        # asked for, or there is nothing to fuse and nothing to diversify away.
        candidate_k = top_k * self._multiplier

        query_vector = (await self._embedder.embed([query]))[0]
        vector_hits = await self._store.search(
            tenant_id, query_vector, top_k=candidate_k, corpus=corpus, min_score=min_score
        )

        rankings: list[Sequence[ScoredDocument]] = [vector_hits]
        strategy = "vector"

        if self._lexical is not None:
            lexical_hits = self._lexical.search(tenant_id, query, top_k=candidate_k)
            # The index is already partitioned by tenant. Filtering again is redundant by
            # design: isolation is the one property worth paying for twice, and a stray
            # foreign document here would be a silent leak rather than a loud failure.
            safe_hits = [hit for hit in lexical_hits if hit.document.tenant_id == tenant_id]
            if len(safe_hits) != len(lexical_hits):
                log.error(
                    "retrieval.foreign_documents_dropped",
                    tenant_id=tenant_id,
                    dropped=len(lexical_hits) - len(safe_hits),
                )
            if safe_hits:
                rankings.append(safe_hits)
                strategy = "hybrid_rrf"

        fused = (
            reciprocal_rank_fusion(rankings, top_k=candidate_k)
            if len(rankings) > 1
            else list(vector_hits[:candidate_k])
        )
        final = maximal_marginal_relevance(
            query_vector, fused, top_k=top_k, diversity=self._diversity
        )

        return RetrievalResult(
            query=query,
            documents=tuple(final),
            strategy=f"{strategy}+mmr" if self._diversity > 0 else strategy,
        )
