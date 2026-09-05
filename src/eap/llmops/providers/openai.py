"""OpenAI Chat Completions adapter.

Also covers Azure OpenAI and the many gateways that speak the same wire format; point
``base_url`` at them and set ``supported_models`` accordingly.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from eap.llmops.providers.base import (
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
    Usage,
)
from eap.platform.errors import ProviderError

OPENAI_MODELS = frozenset({"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini"})


class OpenAIProvider(LLMProvider):
    name = "openai"
    supported_models = OPENAI_MODELS

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
        timeout: float = 45.0,
        models: frozenset[str] | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "OpenAI provider requires an API key", provider=self.name, retryable=False
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        if models is not None:
            self.supported_models = models

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or "gpt-4o-mini"
        payload: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": str(message.role), "content": message.content}
                | ({"name": message.name} if message.name else {})
                | ({"tool_call_id": message.tool_call_id} if message.tool_call_id else {})
                for message in request.messages
            ],
        }
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]

        started = time.perf_counter()
        client = await self._http()
        try:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport failure: {exc}", provider=self.name) from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI returned {response.status_code}: {_error_detail(response)}",
                provider=self.name,
                retryable=response.status_code == 429 or response.status_code >= 500,
                status_code=response.status_code,
            )

        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message", {})

        tool_calls = tuple(
            ToolCall(
                id=call.get("id", ""),
                name=call.get("function", {}).get("name", ""),
                arguments=_safe_json(call.get("function", {}).get("arguments", "{}")),
            )
            for call in message.get("tool_calls") or []
        )

        usage_body = body.get("usage") or {}
        details = usage_body.get("prompt_tokens_details") or {}
        return CompletionResponse(
            text=message.get("content") or "",
            model=body.get("model", model),
            provider=self.name,
            usage=Usage(
                input_tokens=int(usage_body.get("prompt_tokens", 0)),
                output_tokens=int(usage_body.get("completion_tokens", 0)),
                cached_input_tokens=int(details.get("cached_tokens", 0)),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=tool_calls,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def health(self) -> bool:
        try:
            client = await self._http()
            response = await client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False


def _safe_json(raw: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string the model generated, so it may not parse."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"_unparsed": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", response.text))[:300]
    except ValueError:
        return response.text[:300]
