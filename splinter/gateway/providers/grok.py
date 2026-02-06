"""Grok (xAI) provider implementation for Splinter Gateway.

xAI's Grok API is OpenAI-compatible, using the same SDK with a different base URL.
"""

import time
from typing import Any, AsyncIterator

from ...exceptions import ProviderError
from ...types import LLMMessage, LLMProvider, LLMRequest, LLMResponse
from .base import BaseProvider


class GrokProvider(BaseProvider):
    """Grok (xAI) API provider.

    Uses OpenAI-compatible API at api.x.ai.

    Example:
        provider = GrokProvider(api_key="xai-...")
        response = await provider.complete(request)
    """

    provider_type = LLMProvider.GROK

    supported_models = [
        "grok-3",
        "grok-3-fast",
        "grok-3-mini",
        "grok-3-mini-fast",
        "grok-2",
        "grok-2-mini",
        "grok-2-vision",
        "grok-beta",
    ]

    default_model = "grok-3-mini-fast"

    # Pricing per 1M tokens (input, output) - estimated
    pricing = {
        "grok-3": (3.0, 15.0),
        "grok-3-fast": (5.0, 25.0),
        "grok-3-mini": (0.3, 0.5),
        "grok-3-mini-fast": (0.6, 4.0),
        "grok-2": (2.0, 10.0),
        "grok-2-mini": (0.2, 0.4),
        "grok-2-vision": (2.0, 10.0),
        "grok-beta": (5.0, 15.0),
    }

    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        """Initialize Grok provider.

        Args:
            api_key: xAI API key. If None, uses XAI_API_KEY env var.
            base_url: Optional custom base URL.
            timeout: Request timeout in seconds.
        """
        super().__init__()
        self._api_key = api_key
        self._base_url = base_url or self.XAI_BASE_URL
        self._timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize and return the OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Grok uses OpenAI-compatible API. "
                    "Install with: pip install openai"
                )

            import os

            api_key = self._api_key or os.environ.get("XAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "xAI API key not provided. Set XAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )

        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to Grok.

        Args:
            request: The LLM request.

        Returns:
            LLMResponse with content and metrics.

        Raises:
            ProviderError: If the request fails.
        """
        client = self._get_client()
        model = request.model or self.default_model

        start_time = time.time()

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": self.format_messages(request.messages),
            }

            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            if request.tools:
                kwargs["tools"] = self.format_tools(request.tools)

            response = await client.chat.completions.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            message = response.choices[0].message
            content = message.content

            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self.calculate_cost(model, input_tokens, output_tokens)

            llm_response = LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model=model,
                provider=self.provider_type,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

            self.record_metrics(llm_response)
            return llm_response

        except Exception as e:
            raise ProviderError("grok", str(e), e)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion response from Grok.

        Args:
            request: The LLM request.

        Yields:
            Response content chunks.

        Raises:
            ProviderError: If the request fails.
        """
        client = self._get_client()
        model = request.model or self.default_model

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": self.format_messages(request.messages),
                "stream": True,
            }

            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens

            async for chunk in await client.chat.completions.create(**kwargs):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ProviderError("grok", str(e), e)

    def format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for Grok API (OpenAI-compatible)."""
        if not tools:
            return None

        formatted = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                formatted.append(tool)
            else:
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", tool.get("input_schema", {})),
                    },
                })

        return formatted
