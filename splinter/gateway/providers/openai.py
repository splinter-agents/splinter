"""OpenAI provider implementation for Splinter Gateway."""

import time
from typing import Any, AsyncIterator

from ...exceptions import ProviderError
from ...types import LLMMessage, LLMProvider, LLMRequest, LLMResponse
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider.

    Supports GPT-4, GPT-3.5, and other OpenAI models.

    Example:
        provider = OpenAIProvider(api_key="sk-...")
        response = await provider.complete(request)
    """

    provider_type = LLMProvider.OPENAI

    supported_models = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o1-preview",
    ]

    default_model = "gpt-4o"

    # Pricing per 1M tokens (input, output) as of 2024
    pricing = {
        "gpt-4": (30.0, 60.0),
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-4-turbo-preview": (10.0, 30.0),
        "gpt-4o": (5.0, 15.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-3.5-turbo": (0.5, 1.5),
        "o1": (15.0, 60.0),
        "o1-mini": (3.0, 12.0),
        "o1-preview": (15.0, 60.0),
    }

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            organization: Optional organization ID.
            base_url: Optional custom base URL (for Azure, etc.).
            timeout: Request timeout in seconds.
        """
        super().__init__()
        self._api_key = api_key
        self._organization = organization
        self._base_url = base_url
        self._timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize and return the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install splinter[openai]"
                )

            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._organization:
                kwargs["organization"] = self._organization
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncOpenAI(**kwargs)

        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to OpenAI.

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
            # Build request kwargs
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

            # Make request
            response = await client.chat.completions.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Extract response data
            message = response.choices[0].message
            content = message.content

            # Extract tool calls if present
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

            # Get token usage
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
            raise ProviderError("openai", str(e), e)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion response from OpenAI.

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
            raise ProviderError("openai", str(e), e)

    def format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for OpenAI API.

        Args:
            tools: List of tool definitions.

        Returns:
            OpenAI-formatted tools.
        """
        if not tools:
            return None

        formatted = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                # Already in OpenAI format
                formatted.append(tool)
            else:
                # Convert from generic format
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", tool.get("input_schema", {})),
                    },
                })

        return formatted
