"""Tool contract.

Every tool declares three things the runtime enforces before it runs:

* ``required_permission`` — checked against the caller ∩ agent intersection.
* ``mutates`` — whether it changes state outside the platform. Mutating tools attract the
  human-approval obligation from policy, and are refused outright to agent principals.
* ``parameters`` — JSON Schema, validated before dispatch.

Argument validation happening *before* dispatch is the part that matters. Tool arguments are
model-generated, which makes them untrusted input in the ordinary sense: they can be
malformed, out of range, or shaped by an injected instruction. A tool that validates
internally has already been entered with hostile input.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from eap.llmops.providers.base import ToolSchema
from eap.platform.errors import ValidationError


@dataclass(frozen=True, slots=True)
class ToolResult:
    output: str
    success: bool = True
    metadata: dict[str, str] = field(default_factory=dict)
    """``sensitive`` marks output that must not be echoed into a log or a trace."""

    @property
    def is_sensitive(self) -> bool:
        return self.metadata.get("sensitive") == "true"


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]
    required_permission: str
    mutates: bool

    async def run(self, arguments: dict[str, Any]) -> ToolResult: ...


@dataclass(slots=True)
class FunctionTool:
    """Wraps an async callable as a tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[ToolResult]]
    required_permission: str = "tool:execute"
    mutates: bool = False

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await self.handler(arguments)

    def schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate against the subset of JSON Schema that tool parameters actually use.

    A full JSON Schema implementation is a dependency and a large attack surface for what
    tool definitions need in practice: object types, required keys, primitive types, enums
    and numeric bounds. Anything richer than this belongs inside the tool.

    Unknown properties are rejected rather than ignored. A model that invents an argument is
    a model that has misunderstood the tool, and silently dropping the argument turns that
    into a wrong result instead of an error.
    """
    if schema.get("type") != "object":
        return arguments

    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValidationError("tool arguments are missing required fields", missing=missing)

    if not schema.get("additionalProperties", False):
        unexpected = [name for name in arguments if name not in properties]
        if unexpected:
            raise ValidationError(
                "tool arguments contain fields the tool does not accept",
                unexpected=unexpected,
                accepted=sorted(properties),
            )

    validated: dict[str, Any] = {}
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            continue
        validated[name] = _coerce(name, value, spec)
    return validated


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _coerce(name: str, value: Any, spec: dict[str, Any]) -> Any:
    expected = spec.get("type")
    if expected in _JSON_TYPES:
        # bool is a subclass of int in Python, so an integer field would accept True.
        if expected in ("integer", "number") and isinstance(value, bool):
            raise ValidationError(f"argument '{name}' must be {expected}", got="boolean")
        if not isinstance(value, _JSON_TYPES[expected]):
            raise ValidationError(f"argument '{name}' must be {expected}", got=type(value).__name__)

    if "enum" in spec and value not in spec["enum"]:
        raise ValidationError(
            f"argument '{name}' is not one of the permitted values",
            permitted=spec["enum"],
            got=value,
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            raise ValidationError(
                f"argument '{name}' is below its minimum", minimum=spec["minimum"]
            )
        if "maximum" in spec and value > spec["maximum"]:
            raise ValidationError(
                f"argument '{name}' is above its maximum", maximum=spec["maximum"]
            )

    if isinstance(value, str):
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            raise ValidationError(
                f"argument '{name}' exceeds its maximum length", maximum=spec["maxLength"]
            )
        if "minLength" in spec and len(value) < spec["minLength"]:
            raise ValidationError(
                f"argument '{name}' is shorter than its minimum length", minimum=spec["minLength"]
            )

    return value
