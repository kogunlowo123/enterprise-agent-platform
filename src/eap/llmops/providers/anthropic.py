"""Anthropic Messages API adapter."""

from __future__ import annotations

import time
from typing import Any

import httpx

from eap.llmops.providers.base import (
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    Role,
    ToolCall,
    Usage,
)
from eap.platform.errors import ProviderError

ANTHROPIC_MODELS = frozenset(
    {
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5-1",
        "claude-haiku-4-5-20251001",
    }
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supported_models = ANTHROPIC_MODELS

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        api_version: str = "2023-06-01",
        client: httpx.AsyncClient | None = None,
        timeout: float = 45.0,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "Anthropic provider requires an API key", provider=self.name, retryable=False
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or "claude-sonnet-5"
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": _role(message.role), "content": message.content}
                for message in request.without_system()
            ],
        }
        system = request.system_prompt()
        if system:
            payload["system"] = system
        if request.stop_sequences:
            payload["stop_sequences"] = list(request.stop_sequences)
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]

        started = time.perf_counter()
        client = await self._http()
        try:
            response = await client.post(
                f"{self._base_url}/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self._api_version,
                    "content-type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport failure: {exc}", provider=self.name) from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"Anthropic returned {response.status_code}: {_error_detail(response)}",
                provider=self.name,
                # 4xx other than 429 will fail identically on retry.
                retryable=response.status_code == 429 or response.status_code >= 500,
                status_code=response.status_code,
            )

        body = response.json()
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in body.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}) or {},
                    )
                )

        usage_body = body.get("usage", {})
        return CompletionResponse(
            text="".join(text_parts),
            model=body.get("model", model),
            provider=self.name,
            usage=Usage(
                input_tokens=int(usage_body.get("input_tokens", 0)),
                output_tokens=int(usage_body.get("output_tokens", 0)),
                cached_input_tokens=int(usage_body.get("cache_read_input_tokens", 0)),
            ),
            finish_reason=body.get("stop_reason", "stop"),
            tool_calls=tuple(tool_calls),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def health(self) -> bool:
        try:
            client = await self._http()
            response = await client.post(
                f"{self._base_url}/v1/messages",
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "ok"}],
                },
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self._api_version,
                    "content-type": "application/json",
                },
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False


def _role(role: Role) -> str:
    # Anthropic has no tool role on the messages array; tool results are user turns.
    return "assistant" if role is Role.ASSISTANT else "user"


def _error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", response.text))[:300]
    except ValueError:
        return response.text[:300]
