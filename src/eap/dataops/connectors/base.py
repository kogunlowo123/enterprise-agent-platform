"""Source connector contract.

A connector's whole job is to yield :class:`SourceDocument` objects with honest provenance.
It does not chunk, embed or store — those belong to the ingestion pipeline, and keeping
them out of the connector is what allows a new source to be added without touching
retrieval.

Provenance is the part connectors get wrong. A citation that says "the engineering handbook"
is not checkable; one that says "handbook/oncall.md at commit 4f2a1c9, lines 40-58" is. Every
connector must produce a stable ``uri`` that a human can follow back to the exact revision
the answer was grounded on, because the source will have changed by the time anyone
disputes the answer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One retrievable artifact as it exists at the source."""

    source_id: str
    """Stable identity across syncs. Re-ingesting the same source_id replaces its chunks."""

    title: str
    text: str
    uri: str
    """Permalink pinned to a revision, not to a branch."""

    content_type: str = "text"
    revision: str | None = None
    modified_at: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncStats:
    documents: int = 0
    bytes_read: int = 0
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)


class Connector(Protocol):
    name: str

    def fetch(self) -> AsyncIterator[SourceDocument]: ...
