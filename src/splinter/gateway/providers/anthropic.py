"""Anthropic provider implementation for Splinter Gateway."""

import time
from typing import Any, AsyncIterator

from ...exceptions import ProviderError
from ...schemas import LLMMessage, LLMProvider, LLMRequest, LLMResponse
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic API provider.

    Supports Claude 3, Claude 3.5, and other Anthropic models.

    Example:
        provider = AnthropicProvider(api_key="sk-ant-...")
        response = await provider.complete(request)
    """

    provider_type = LLMProvider.ANTHROPIC

    supported_models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-5-20251101",
    ]

    default_model = "claude-sonnet-4-20250514"

    # Pricing per 1M tokens (input, output) as of 2024
    pricing = {
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-3-sonnet-20240229": (3.0, 15.0),
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-3-5-sonnet-20240620": (3.0, 15.0),
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-3-5-haiku-20241022": (0.80, 4.0),
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-opus-4-5-20251101": (15.0, 75.0),
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4096,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            base_url: Optional custom base URL.
            timeout: Request timeout in seconds.
            max_tokens: Default max tokens for responses.
        """
        super().__init__()
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize and return the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install splinter[anthropic]"
                )

            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncAnthropic(**kwargs)

        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to Anthropic.

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
            # Separate system message from other messages
            system_message = None
            messages = []

            for msg in request.messages:
                if msg.role == "system":
                    system_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                else:
                    messages.append(self._format_message(msg))

            # Build request kwargs
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens or self._max_tokens,
            }

            if system_message:
                kwargs["system"] = system_message
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.tools:
                kwargs["tools"] = self.format_tools(request.tools)

            # Make request
            response = await client.messages.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Extract content and tool calls
            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    })

            # Get token usage
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            cost = self.calculate_cost(model, input_tokens, output_tokens)

            llm_response = LLMResponse(
                content=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
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
            raise ProviderError("anthropic", str(e), e)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion response from Anthropic.

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
            # Separate system message
            system_message = None
            messages = []

            for msg in request.messages:
                if msg.role == "system":
                    system_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                else:
                    messages.append(self._format_message(msg))

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens or self._max_tokens,
            }

            if system_message:
                kwargs["system"] = system_message
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature

            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise ProviderError("anthropic", str(e), e)

    def _format_message(self, msg: LLMMessage) -> dict[str, Any]:
        """Format a single message for Anthropic API."""
        formatted: dict[str, Any] = {"role": msg.role}

        if isinstance(msg.content, str):
            formatted["content"] = msg.content
        elif isinstance(msg.content, list):
            # Handle multi-part content
            formatted["content"] = msg.content
        else:
            formatted["content"] = str(msg.content)

        return formatted

    def format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for Anthropic API.

        Args:
            tools: List of tool definitions.

        Returns:
            Anthropic-formatted tools.
        """
        if not tools:
            return None

        formatted = []
        for tool in tools:
            if "input_schema" in tool:
                # Already in Anthropic format
                formatted.append(tool)
            elif "type" in tool and tool["type"] == "function":
                # Convert from OpenAI format
                func = tool.get("function", {})
                formatted.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            else:
                # Convert from generic format
                formatted.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {}),
                })

        return formatted
