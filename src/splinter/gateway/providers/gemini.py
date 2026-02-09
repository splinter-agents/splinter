"""Google Gemini provider implementation for Splinter Gateway."""

import time
from typing import Any, AsyncIterator

from ...exceptions import ProviderError
from ...schemas import LLMMessage, LLMProvider, LLMRequest, LLMResponse
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini API provider.

    Supports Gemini Pro, Gemini Ultra, and other Google models.

    Example:
        provider = GeminiProvider(api_key="...")
        response = await provider.complete(request)
    """

    provider_type = LLMProvider.GEMINI

    supported_models = [
        "gemini-pro",
        "gemini-1.0-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
    ]

    default_model = "gemini-1.5-pro"

    # Pricing per 1M tokens (input, output) - varies by context length
    pricing = {
        "gemini-pro": (0.5, 1.5),
        "gemini-1.0-pro": (0.5, 1.5),
        "gemini-1.5-pro": (3.5, 10.5),  # Up to 128k context
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-2.0-flash-exp": (0.075, 0.30),
    }

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        super().__init__()
        self._api_key = api_key
        self._timeout = timeout
        self._client: Any = None
        self._model_instances: dict[str, Any] = {}

    def _get_client(self, model: str) -> Any:
        """Lazily initialize and return a Gemini model client."""
        if model not in self._model_instances:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "Google Generative AI package not installed. "
                    "Install with: pip install splinter[google]"
                )

            if self._api_key:
                genai.configure(api_key=self._api_key)

            self._model_instances[model] = genai.GenerativeModel(model)

        return self._model_instances[model]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to Gemini.

        Args:
            request: The LLM request.

        Returns:
            LLMResponse with content and metrics.

        Raises:
            ProviderError: If the request fails.
        """
        model_name = request.model or self.default_model
        model = self._get_client(model_name)

        start_time = time.time()

        try:
            # Convert messages to Gemini format
            contents = self._convert_messages(request.messages)

            # Build generation config
            generation_config: dict[str, Any] = {}
            if request.temperature is not None:
                generation_config["temperature"] = request.temperature
            if request.max_tokens is not None:
                generation_config["max_output_tokens"] = request.max_tokens

            # Handle tools if present
            tools = None
            if request.tools:
                tools = self._convert_tools(request.tools)

            # Make request (run sync API in thread for async compatibility)
            import asyncio

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    contents,
                    generation_config=generation_config if generation_config else None,
                    tools=tools,
                ),
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            content = ""
            tool_calls = []

            if response.candidates:
                candidate = response.candidates[0]
                for part in candidate.content.parts:
                    if hasattr(part, "text"):
                        content += part.text
                    elif hasattr(part, "function_call"):
                        fc = part.function_call
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": dict(fc.args) if fc.args else {},
                            },
                        })

            # Estimate token usage (Gemini doesn't always provide this)
            input_tokens = 0
            output_tokens = 0

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
            else:
                # Rough estimate: ~4 chars per token
                input_tokens = sum(len(str(m.content)) for m in request.messages) // 4
                output_tokens = len(content) // 4

            cost = self.calculate_cost(model_name, input_tokens, output_tokens)

            llm_response = LLMResponse(
                content=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model=model_name,
                provider=self.provider_type,
                latency_ms=latency_ms,
                raw_response=None,  # Gemini response isn't easily serializable
            )

            self.record_metrics(llm_response)
            return llm_response

        except Exception as e:
            raise ProviderError("gemini", str(e), e)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion response from Gemini.

        Args:
            request: The LLM request.

        Yields:
            Response content chunks.

        Raises:
            ProviderError: If the request fails.
        """
        model_name = request.model or self.default_model
        model = self._get_client(model_name)

        try:
            contents = self._convert_messages(request.messages)

            generation_config: dict[str, Any] = {}
            if request.temperature is not None:
                generation_config["temperature"] = request.temperature
            if request.max_tokens is not None:
                generation_config["max_output_tokens"] = request.max_tokens

            import asyncio

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    contents,
                    generation_config=generation_config if generation_config else None,
                    stream=True,
                ),
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            raise ProviderError("gemini", str(e), e)

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert messages to Gemini format.

        Gemini uses a different format with 'parts' instead of 'content'.
        """
        contents = []
        system_parts = []

        for msg in messages:
            if msg.role == "system":
                # Gemini handles system prompts differently
                system_parts.append({"text": msg.content if isinstance(msg.content, str) else str(msg.content)})
            elif msg.role == "user":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })
            elif msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })
            elif msg.role == "tool":
                # Handle tool responses
                contents.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": msg.name or "tool",
                            "response": {"result": msg.content},
                        }
                    }],
                })

        # Prepend system message to first user message if present
        if system_parts and contents:
            first_user_idx = next(
                (i for i, c in enumerate(contents) if c["role"] == "user"), 0
            )
            contents[first_user_idx]["parts"] = system_parts + contents[first_user_idx]["parts"]

        return contents

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[Any]:
        """Convert tools to Gemini format."""
        try:
            from google.generativeai.types import FunctionDeclaration, Tool
        except ImportError:
            return []

        function_declarations = []

        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", {})
            else:
                name = tool.get("name", "")
                description = tool.get("description", "")
                parameters = tool.get("parameters", tool.get("input_schema", {}))

            function_declarations.append(
                FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )

        return [Tool(function_declarations=function_declarations)]
