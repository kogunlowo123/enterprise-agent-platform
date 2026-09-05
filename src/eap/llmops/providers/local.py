"""Deterministic provider for tests, CI and offline development.

Not a mock. It implements the full provider contract and produces a reproducible answer
derived from the prompt it was given, so the routing, budgeting, guardrail, orchestration
and evaluation paths all execute exactly as they do against a real vendor — no network, no
key, no per-run cost, no flake from sampling variance.

It answers by extracting sentences from the retrieved context that overlap the question,
and cites the numbered source they came from. That makes it genuinely useful for testing
the parts of the system that care about grounding: a citation-coverage metric run against
it measures the retriever and the prompt assembly rather than the model's mood.

The failure-injection controls exist to test resilience paths that are otherwise
unreachable without breaking a real provider on purpose.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import deque

from eap.llmops.providers.base import (
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    Role,
    Usage,
)
from eap.platform.errors import ProviderError

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9']+")
_CITATION_MARKER = re.compile(r"^\[(\d+)\]", re.MULTILINE)


class DeterministicProvider(LLMProvider):
    """A provider whose output is a pure function of its input."""

    name = "local"
    supported_models = frozenset({"local-deterministic"})

    def __init__(
        self,
        *,
        model: str = "local-deterministic",
        latency_ms: float = 0.0,
        fail_times: int = 0,
        fail_with: ProviderError | None = None,
        healthy: bool = True,
    ) -> None:
        self._model = model
        self.supported_models = frozenset({model})
        self._latency_ms = latency_ms
        self._failures: deque[ProviderError] = deque(
            [fail_with or ProviderError("injected failure", provider=self.name)] * fail_times
        )
        self._healthy = healthy
        self.call_count = 0
        self.last_request: CompletionRequest | None = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        self.last_request = request

        if self._failures:
            raise self._failures.popleft()

        started = time.perf_counter()
        question = _last_user_message(request)
        context = _system_context(request)
        answer = _answer_from_context(question, context)

        return CompletionResponse(
            text=answer,
            model=self._model,
            provider=self.name,
            usage=Usage(
                input_tokens=_estimate_tokens(request),
                output_tokens=max(1, len(answer) // 4),
            ),
            finish_reason="stop",
            latency_ms=self._latency_ms or round((time.perf_counter() - started) * 1000, 3),
            metadata={"deterministic": "true", "digest": _digest(question, context)},
        )

    async def health(self) -> bool:
        return self._healthy


def _last_user_message(request: CompletionRequest) -> str:
    for message in reversed(list(request.messages)):
        if message.role is Role.USER:
            return message.content
    return ""


def _system_context(request: CompletionRequest) -> str:
    return "\n\n".join(m.content for m in request.messages if m.role is Role.SYSTEM)


_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the to
    was were what when where which who will with you your do does did can could should
    would this these those there""".split()
)

_MATCH_THRESHOLD = 0.25
_STEM_PREFIX = 5


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def _related(a: str, b: str) -> bool:
    """Loose stemming by shared prefix.

    Exact token equality is too strict for a fixture: a question about "deployment" would
    fail to match a source that says "deploys", and the resulting refusal would look like a
    retrieval failure rather than what it is. A shared five-character prefix covers the
    ordinary inflections without pulling in a stemmer dependency.
    """
    if a == b:
        return True
    if len(a) < _STEM_PREFIX or len(b) < _STEM_PREFIX:
        return False
    return a[:_STEM_PREFIX] == b[:_STEM_PREFIX]


def _coverage(question_words: set[str], sentence_words: set[str]) -> float:
    """Fraction of the question's content words that the sentence addresses."""
    if not question_words:
        return 0.0
    matched = sum(
        1 for word in question_words if any(_related(word, other) for other in sentence_words)
    )
    return matched / len(question_words)


def _answer_from_context(question: str, context: str) -> str:
    """Select the context sentences that best cover the question, and cite their source.

    Content-word coverage rather than Jaccard similarity, because Jaccard punishes a long
    source sentence for containing detail the question did not mention — which is exactly
    what a good source does. This makes output a real function of retrieval quality: better
    context produces a better answer, which is the property an evaluation fixture needs.
    """
    if not context.strip():
        return "No supporting context was supplied, so there is nothing to ground an answer on."

    question_words = _content_words(question)
    if not question_words:
        return "The request contained no answerable content."

    # Answer only from numbered source blocks. Without this the provider quotes the
    # instruction text surrounding the sources, which produces answers citing source
    # numbers that do not exist and scoring as ungrounded against the corpus -- measuring
    # the prompt template rather than the retrieval.
    #
    # No numbered sources means nothing citable was supplied, which is a refusal rather
    # than an invitation to answer from whatever else happens to be in the prompt. That
    # distinction is the whole point of a grounded-answer fixture.
    blocks = [block for block in context.split("\n\n") if _CITATION_MARKER.search(block)]
    if not blocks:
        return (
            "No numbered sources were supplied, so the available sources do not contain "
            "material that answers this question."
        )

    scored: list[tuple[float, int, str]] = []
    for source_index, block in enumerate(blocks, start=1):
        marker = _CITATION_MARKER.search(block)
        citation = int(marker.group(1)) if marker else source_index

        body = block
        _header, newline, remainder = block.partition("\n")
        if marker is not None and newline:
            # Drop the "[n] citation" header line; it is provenance, not content.
            body = remainder

        for sentence in _SENTENCE.split(body):
            sentence = " ".join(sentence.split())
            if len(sentence) < 15:
                continue
            coverage = _coverage(question_words, _content_words(sentence))
            if coverage >= _MATCH_THRESHOLD:
                scored.append((coverage, citation, sentence))

    if not scored:
        return (
            "The supplied context does not contain material that answers this question. "
            "Answering would require going beyond the retrieved sources."
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return " ".join(f"{sentence} [{citation}]" for _, citation, sentence in scored[:3])


def _estimate_tokens(request: CompletionRequest) -> int:
    return max(1, sum(len(m.content) for m in request.messages) // 4)


def _digest(question: str, context: str) -> str:
    return hashlib.blake2b(f"{question}|{context}".encode(), digest_size=8).hexdigest()
