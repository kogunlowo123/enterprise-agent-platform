"""Structure-aware chunking.

Fixed-size chunking is the single largest source of bad retrieval quality in RAG systems
that otherwise look correct. Splitting every 512 tokens regardless of content cuts tables
in half, separates a heading from the paragraph it introduces, and severs a function
signature from its body — so the retrieved chunk is syntactically intact but semantically
useless, and the model grounds an answer on a fragment.

This chunker splits on structure first and falls back to size only when a structural unit
is genuinely too large:

* **Markdown** splits at heading boundaries and carries the heading path onto every chunk,
  so a chunk from deep in a policy document still says which section it came from. That
  path is what makes a citation legible to the person checking the answer.
* **Code** splits at top-level definitions, so a function stays whole.
* **Plain text** splits at paragraph breaks, then sentences.

Token counts are estimated from character length rather than a real tokenizer. The
estimate is calibrated on English prose and code, runs without a model dependency, and is
used only to decide where to cut — a 10% error moves a boundary slightly and costs
nothing, whereas a hard tokenizer dependency costs a 200MB download at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

CHARS_PER_TOKEN = 4.0

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CODE_DEFINITION = re.compile(
    r"^(?:(?:async\s+)?def\s+\w+|class\s+\w+|"
    r"(?:export\s+)?(?:async\s+)?function\s+\w+|"
    r"(?:public|private|protected)\s+[\w<>\[\]]+\s+\w+\s*\()",
    re.MULTILINE,
)
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit, carrying enough provenance to cite it."""

    text: str
    index: int
    source_id: str
    token_estimate: int
    heading_path: tuple[str, ...] = ()
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """A human-checkable pointer back to the original material."""
        location = self.source_id
        if self.heading_path:
            location += " § " + " > ".join(self.heading_path)
        if self.start_line is not None:
            location += f":{self.start_line}"
            if self.end_line is not None and self.end_line != self.start_line:
                location += f"-{self.end_line}"
        return location


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 32
    """Chunks below this are merged forward. A 6-token chunk retrieves noisily: it matches
    on a single term and contributes nothing the model can ground on."""

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")


class Chunker:
    """Dispatches to a structure-aware strategy based on content type."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._config = config or ChunkingConfig()

    def chunk(
        self,
        text: str,
        *,
        source_id: str,
        content_type: str = "text",
        metadata: dict[str, str] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        base_metadata = metadata or {}

        if content_type == "markdown":
            segments = self._split_markdown(text)
        elif content_type == "code":
            segments = self._split_code(text)
        else:
            segments = [((), text, 1)]

        chunks: list[Chunk] = []
        for heading_path, body, start_line in segments:
            for piece, offset in self._enforce_size(body):
                stripped = piece.strip()
                if not stripped:
                    continue
                line = start_line + body[:offset].count("\n")
                chunks.append(
                    Chunk(
                        text=stripped,
                        index=len(chunks),
                        source_id=source_id,
                        token_estimate=estimate_tokens(stripped),
                        heading_path=heading_path,
                        start_line=line,
                        end_line=line + stripped.count("\n"),
                        metadata=dict(base_metadata),
                    )
                )
        return self._merge_undersized(chunks)

    def _split_markdown(self, text: str) -> list[tuple[tuple[str, ...], str, int]]:
        """Split at headings, tracking the full heading path down the tree."""
        matches = list(_MARKDOWN_HEADING.finditer(text))
        if not matches:
            return [((), text, 1)]

        segments: list[tuple[tuple[str, ...], str, int]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            segments.append(((), preamble, 1))

        path: list[str] = []
        for position, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            del path[level - 1 :]
            path.append(title)

            end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            body = text[match.start() : end]
            segments.append((tuple(path), body, text[: match.start()].count("\n") + 1))
        return segments

    def _split_code(self, text: str) -> list[tuple[tuple[str, ...], str, int]]:
        """Split before each top-level definition so a function is never cut in half."""
        boundaries = [m.start() for m in _CODE_DEFINITION.finditer(text)]
        if not boundaries:
            return [((), text, 1)]
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        boundaries.append(len(text))

        segments: list[tuple[tuple[str, ...], str, int]] = []
        for start, end in pairwise(boundaries):
            body = text[start:end]
            if body.strip():
                segments.append(((), body, text[:start].count("\n") + 1))
        return segments

    def _enforce_size(self, body: str) -> list[tuple[str, int]]:
        """Cut an oversized segment on paragraph, then sentence, then hard boundaries."""
        max_chars = int(self._config.max_tokens * CHARS_PER_TOKEN)
        overlap_chars = int(self._config.overlap_tokens * CHARS_PER_TOKEN)
        if len(body) <= max_chars:
            return [(body, 0)]

        units = self._candidate_units(body, max_chars)
        pieces: list[tuple[str, int]] = []
        buffer = ""
        buffer_offset = 0
        cursor = 0

        for unit in units:
            if buffer and len(buffer) + len(unit) > max_chars:
                pieces.append((buffer, buffer_offset))
                # Overlap carries the tail of the previous chunk forward, so a fact that
                # straddles a boundary appears whole in at least one chunk.
                carry = buffer[-overlap_chars:] if overlap_chars else ""
                buffer_offset = cursor - len(carry)
                buffer = carry + unit
            else:
                if not buffer:
                    buffer_offset = cursor
                buffer += unit
            cursor += len(unit)

        if buffer.strip():
            pieces.append((buffer, buffer_offset))
        return pieces

    @staticmethod
    def _candidate_units(body: str, max_chars: int) -> list[str]:
        units = [p for p in _PARAGRAPH.split(body) if p]
        if all(len(u) <= max_chars for u in units):
            return [u + "\n\n" for u in units]

        finer: list[str] = []
        for unit in units:
            if len(unit) <= max_chars:
                finer.append(unit + "\n\n")
                continue
            sentences = _SENTENCE.split(unit)
            for sentence in sentences:
                if len(sentence) <= max_chars:
                    finer.append(sentence + " ")
                else:
                    # A single unbroken run longer than the limit (minified data, a base64
                    # blob). Nothing structural is left; cut on size.
                    finer.extend(
                        sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)
                    )
        return finer

    def _merge_undersized(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fold a too-small chunk into its neighbour when they share a heading path."""
        if len(chunks) < 2:
            return chunks

        merged: list[Chunk] = []
        for chunk in chunks:
            if (
                merged
                and chunk.token_estimate < self._config.min_tokens
                and merged[-1].heading_path == chunk.heading_path
                and merged[-1].token_estimate + chunk.token_estimate <= self._config.max_tokens
            ):
                previous = merged.pop()
                text = f"{previous.text}\n\n{chunk.text}"
                merged.append(
                    Chunk(
                        text=text,
                        index=previous.index,
                        source_id=previous.source_id,
                        token_estimate=estimate_tokens(text),
                        heading_path=previous.heading_path,
                        start_line=previous.start_line,
                        end_line=chunk.end_line,
                        metadata=previous.metadata,
                    )
                )
            else:
                merged.append(chunk)

        return [
            Chunk(
                text=c.text,
                index=position,
                source_id=c.source_id,
                token_estimate=c.token_estimate,
                heading_path=c.heading_path,
                start_line=c.start_line,
                end_line=c.end_line,
                metadata=c.metadata,
            )
            for position, c in enumerate(merged)
        ]
