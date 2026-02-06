"""Mock provider for testing Splinter without external API calls."""

import asyncio
import json
import random
import time
from typing import Any, AsyncIterator, Callable

from ...types import LLMMessage, LLMProvider, LLMRequest, LLMResponse
from .base import BaseProvider


class MockProvider(BaseProvider):
    """Mock LLM provider for testing.

    This provider returns configurable mock responses without making
    any external API calls. Useful for:
    - Unit testing
    - Integration testing
    - Development without API keys
    - Demos

    Example:
        provider = MockProvider(
            responses={
                "default": "This is a mock response",
                "json": '{"result": "mock data"}',
            },
            latency_ms=100,
        )
    """

    provider_type = LLMProvider.OPENAI  # Pretend to be OpenAI for compatibility

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "This is a mock response from Splinter MockProvider.",
        latency_ms: float = 50,
        cost_per_call: float = 0.001,
        tokens_per_call: tuple[int, int] = (100, 50),
        fail_rate: float = 0.0,
        response_generator: Callable[[LLMRequest], str] | None = None,
    ):
        """Initialize mock provider.

        Args:
            responses: Dict mapping keywords to responses.
            default_response: Default response when no keyword matches.
            latency_ms: Simulated latency in milliseconds.
            cost_per_call: Simulated cost per call.
            tokens_per_call: Tuple of (input_tokens, output_tokens) per call.
            fail_rate: Probability of simulated failure (0.0-1.0).
            response_generator: Custom function to generate responses.
        """
        super().__init__()
        self._responses = responses or {}
        self._default_response = default_response
        self._latency_ms = latency_ms
        self._cost_per_call = cost_per_call
        self._tokens_per_call = tokens_per_call
        self._fail_rate = fail_rate
        self._response_generator = response_generator
        self._call_log: list[LLMRequest] = []

    @property
    def call_log(self) -> list[LLMRequest]:
        """Get log of all requests made to this provider."""
        return self._call_log

    def set_response(self, keyword: str, response: str) -> None:
        """Set a response for a keyword."""
        self._responses[keyword] = response

    def set_generator(self, generator: Callable[[LLMRequest], str]) -> None:
        """Set a custom response generator."""
        self._response_generator = generator

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a mock completion response.

        Args:
            request: The LLM request.

        Returns:
            Mock LLMResponse.
        """
        self._call_log.append(request)

        # Simulate latency
        await asyncio.sleep(self._latency_ms / 1000)

        start_time = time.time()

        # Simulate random failures
        if random.random() < self._fail_rate:
            from ...exceptions import ProviderError
            raise ProviderError("mock", "Simulated failure")

        # Generate response content
        content = self._generate_response(request)

        latency_ms = (time.time() - start_time) * 1000 + self._latency_ms
        input_tokens, output_tokens = self._tokens_per_call

        response = LLMResponse(
            content=content,
            tool_calls=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self._cost_per_call,
            model=request.model,
            provider=self.provider_type,
            latency_ms=latency_ms,
        )

        self.record_metrics(response)
        return response

    def _generate_response(self, request: LLMRequest) -> str:
        """Generate response based on request."""
        # Use custom generator if provided
        if self._response_generator:
            return self._response_generator(request)

        # Extract user message content
        user_content = ""
        for msg in request.messages:
            if msg.role == "user":
                user_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Check for keyword matches
        for keyword, response in self._responses.items():
            if keyword.lower() in user_content.lower():
                return response

        # Default response
        return self._default_response

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a mock response."""
        response = await self.complete(request)
        if response.content:
            # Simulate streaming by yielding word by word
            words = response.content.split()
            for word in words:
                await asyncio.sleep(10 / 1000)  # 10ms between words
                yield word + " "

    def reset(self) -> None:
        """Reset call log and metrics."""
        self._call_log.clear()
        self.reset_metrics()


class SmartMockProvider(MockProvider):
    """Enhanced mock provider with intelligent response generation.

    This provider generates responses that simulate realistic LLM behavior:
    - Understands JSON output requirements
    - Responds contextually to different agent types
    - Simulates tool calls when tools are provided
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._response_templates = {
            "research": {
                "findings": ["Finding 1: Mock research result", "Finding 2: Another discovery"],
                "themes": ["Theme A", "Theme B"],
                "sources": ["Source 1", "Source 2"],
            },
            "summary": {
                "executive_summary": "This is a mock executive summary of the research.",
                "key_points": ["Key point 1", "Key point 2", "Key point 3"],
                "recommendations": ["Recommendation 1", "Recommendation 2"],
            },
            "analysis": {
                "analysis": "Mock analysis of the provided topic.",
                "points": ["Analysis point 1", "Analysis point 2"],
            },
            "plan": {
                "objectives": ["Objective 1", "Objective 2"],
                "action_items": ["Action 1", "Action 2", "Action 3"],
                "timeline": "Week 1-2: Planning, Week 3-4: Execution",
            },
        }

    def _generate_response(self, request: LLMRequest) -> str:
        """Generate smart contextual response."""
        # Check system prompt for agent type hints
        system_content = ""
        user_content = ""

        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif msg.role == "user":
                user_content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Detect if JSON output is expected
        wants_json = "json" in system_content.lower() or "json" in user_content.lower()

        # Detect agent type from system prompt
        agent_type = None
        for keyword in self._response_templates:
            if keyword in system_content.lower():
                agent_type = keyword
                break

        if agent_type and wants_json:
            return json.dumps(self._response_templates[agent_type], indent=2)
        elif wants_json:
            return json.dumps({
                "result": "Mock JSON response",
                "data": ["item1", "item2"],
                "success": True,
            })
        else:
            # Plain text response
            topic = user_content[:50] if user_content else "the topic"
            return f"Mock response about {topic}. This is generated by Splinter's SmartMockProvider for testing purposes."
