"""Model provider contract.

One request shape and one response shape across every vendor. The platform's routing,
budgeting, evaluation and observability all sit above this line, so adding a provider means
writing one adapter rather than touching any of them.

The contract deliberately does not expose vendor-specific knobs. Anything that only one
provider supports goes in ``extra`` and is the adapter's problem, because a lowest common
denominator that leaks vendor concepts stops being a common denominator within two
providers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool as the model sees it. JSON Schema, because every provider accepts it."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: Sequence[Message]
    model: str | None = None
    """None means "let the router decide". A caller that pins a model bypasses routing and
    needs the model:route:override permission."""

    max_output_tokens: int = 1024
    temperature: float = 0.0
    tools: Sequence[ToolSchema] = ()
    stop_sequences: Sequence[str] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def system_prompt(self) -> str | None:
        parts = [m.content for m in self.messages if m.role is Role.SYSTEM]
        return "\n\n".join(parts) if parts else None

    def without_system(self) -> list[Message]:
        return [m for m in self.messages if m.role is not Role.SYSTEM]


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    text: str
    model: str
    provider: str
    usage: Usage
    finish_reason: str = "stop"
    tool_calls: tuple[ToolCall, ...] = ()
    latency_ms: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def wants_tool_call(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    name: str
    supported_models: frozenset[str]

    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    async def health(self) -> bool: ...
