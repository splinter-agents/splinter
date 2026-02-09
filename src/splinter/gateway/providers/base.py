"""Base provider interface for Splinter Gateway.

All LLM provider implementations must inherit from BaseProvider.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ...schemas import LLMMessage, LLMProvider, LLMRequest, LLMResponse


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations (OpenAI, Anthropic, Gemini, etc.)
    must inherit from this class and implement the required methods.

    Example:
        class OpenAIProvider(BaseProvider):
            def __init__(self, api_key: str):
                super().__init__()
                self.api_key = api_key

            async def complete(self, request: LLMRequest) -> LLMResponse:
                # Implementation
                pass
    """

    # Provider identifier
    provider_type: LLMProvider

    # Supported models for this provider
    supported_models: list[str] = []

    # Default model
    default_model: str = ""

    # Pricing per 1M tokens (input, output)
    pricing: dict[str, tuple[float, float]] = {}

    def __init__(self):
        """Initialize the base provider."""
        self._total_calls = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency_ms = 0.0

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to the LLM.

        Args:
            request: The LLM request with messages, tools, etc.

        Returns:
            LLMResponse with content, tool calls, and metrics.

        Raises:
            ProviderError: If the request fails.
        """
        pass

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion response.

        Default implementation falls back to non-streaming complete().
        Providers can override for true streaming support.

        Args:
            request: The LLM request.

        Yields:
            Response content chunks.
        """
        response = await self.complete(request)
        if response.content:
            yield response.content

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a request.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in dollars.
        """
        if model not in self.pricing:
            # Try to find partial match
            for model_key in self.pricing:
                if model_key in model or model in model_key:
                    model = model_key
                    break
            else:
                return 0.0

        input_price, output_price = self.pricing[model]
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    def record_metrics(self, response: LLMResponse) -> None:
        """Record metrics from a response.

        Args:
            response: The LLM response.
        """
        self._total_calls += 1
        self._total_tokens += response.input_tokens + response.output_tokens
        self._total_cost += response.cost
        self._total_latency_ms += response.latency_ms

    def get_metrics(self) -> dict[str, Any]:
        """Get provider metrics.

        Returns:
            Dict of metrics.
        """
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._total_calls if self._total_calls > 0 else 0
            ),
        }

    def reset_metrics(self) -> None:
        """Reset provider metrics."""
        self._total_calls = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency_ms = 0.0

    def validate_model(self, model: str) -> bool:
        """Check if a model is supported.

        Args:
            model: Model name to validate.

        Returns:
            True if supported.
        """
        if not self.supported_models:
            return True
        return model in self.supported_models or any(
            m in model for m in self.supported_models
        )

    def format_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Format messages for this provider's API.

        Default implementation returns basic format.
        Providers should override for their specific format.

        Args:
            messages: List of LLMMessage objects.

        Returns:
            List of message dicts for the API.
        """
        formatted = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            formatted.append(m)
        return formatted

    def format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for this provider's API.

        Default implementation returns tools as-is.
        Providers should override for their specific format.

        Args:
            tools: List of tool definitions.

        Returns:
            Formatted tools for the API.
        """
        return tools


class ProviderFactory:
    """Factory for creating provider instances.

    This class manages provider registration and instantiation.

    Example:
        factory = ProviderFactory()
        factory.register("openai", OpenAIProvider)

        provider = factory.create("openai", api_key="sk-...")
    """

    _providers: dict[str, type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseProvider]) -> None:
        """Register a provider class.

        Args:
            name: Provider name (e.g., "openai").
            provider_class: Provider class to register.
        """
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseProvider:
        """Create a provider instance.

        Args:
            name: Provider name.
            **kwargs: Arguments to pass to provider constructor.

        Returns:
            Provider instance.

        Raises:
            KeyError: If provider not registered.
        """
        if name not in cls._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return cls._providers[name](**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List registered providers."""
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered."""
        return name in cls._providers
