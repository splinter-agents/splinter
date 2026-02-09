"""LLM Provider implementations for Splinter Gateway.

Supported providers:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3, Claude 3.5, etc.)
- Google Gemini (Gemini Pro, Gemini 1.5, etc.)
- Grok (xAI - Grok 2, Grok 3, etc.)
- Mock (for testing)
"""

from .anthropic import AnthropicProvider
from .base import BaseProvider, ProviderFactory
from .gemini import GeminiProvider
from .grok import GrokProvider
from .mock import MockProvider, SmartMockProvider
from .openai import OpenAIProvider

# Register providers
ProviderFactory.register("openai", OpenAIProvider)
ProviderFactory.register("anthropic", AnthropicProvider)
ProviderFactory.register("gemini", GeminiProvider)
ProviderFactory.register("grok", GrokProvider)
ProviderFactory.register("mock", MockProvider)

__all__ = [
    "BaseProvider",
    "ProviderFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "GrokProvider",
    "MockProvider",
    "SmartMockProvider",
]
