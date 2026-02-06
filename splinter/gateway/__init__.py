"""Splinter Gateway Layer.

The Gateway is the core routing layer that intercepts all LLM calls.
It provides:
- Routing to LLM providers (OpenAI, Anthropic, Gemini)
- Integration with Control Layer for limit enforcement
- Tracking of cost, tokens, and latency
- Unified interface for all LLM operations

Agents don't call LLM providers directly - all calls go through the Gateway.
"""

from .gateway import CallRecord, Gateway
from .providers import (
    AnthropicProvider,
    BaseProvider,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
    ProviderFactory,
    SmartMockProvider,
)

__all__ = [
    "Gateway",
    "CallRecord",
    "BaseProvider",
    "ProviderFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MockProvider",
    "SmartMockProvider",
]
