"""Gateway - THE central orchestrator. All LLM calls flow through here.

FLOW:
1. Agent calls gateway.call()
2. Gateway checks CONTROL (limits, loops, tools)
3. Gateway routes to LLM PROVIDER
4. Gateway updates COORDINATION (state, checkpoint)
5. Gateway returns result

Gateway orchestrates: Control -> LLM -> Coordination
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable

from ..control.limits import LimitsEnforcer
from ..control.loop_detection import LoopDetector
from ..control.tool_access import ToolAccessController
from ..coordination.checkpoint import CheckpointManager, InMemoryCheckpointStorage
from ..coordination.state import SharedState
from ..exceptions import (
    ProviderError,
    ProviderNotFoundError,
)
from ..types import (
    AgentStatus,
    ExecutionLimits,
    ExecutionMetrics,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LoopDetectionConfig,
)
from .providers import (
    AnthropicProvider,
    BaseProvider,
    GeminiProvider,
    GrokProvider,
    MockProvider,
    OpenAIProvider,
    SmartMockProvider,
)

logger = logging.getLogger(__name__)


class CallRecord:
    """Record of a single LLM call."""

    __slots__ = ('agent_id', 'provider', 'model', 'request', 'response', 'error', 'timestamp')

    def __init__(
        self,
        agent_id: str,
        provider: LLMProvider,
        model: str,
        request: LLMRequest,
        response: LLMResponse | None = None,
        error: str | None = None,
    ):
        self.agent_id = agent_id
        self.provider = provider
        self.model = model
        self.request = request
        self.response = response
        self.error = error
        self.timestamp = datetime.now()

    @property
    def success(self) -> bool:
        return self.response is not None and self.error is None

    @property
    def cost(self) -> float:
        return self.response.cost if self.response else 0.0

    @property
    def latency_ms(self) -> float:
        return self.response.latency_ms if self.response else 0.0


class Gateway:
    """THE central orchestrator for all LLM calls.

    Flow:
        request -> CONTROL check -> LLM call -> COORDINATION update -> response

    Usage:
        gateway = Gateway(limits=ExecutionLimits(max_budget=10.0))
        gateway.configure_provider("openai", api_key="sk-...")
        response = await gateway.call(agent_id="researcher", ...)
    """

    def __init__(
        self,
        limits: ExecutionLimits | None = None,
        loop_detection: LoopDetectionConfig | None = None,
        tool_access: ToolAccessController | None = None,
        checkpoint_enabled: bool = False,
        on_before_call: Callable[[str, LLMRequest], None] | None = None,
        on_after_call: Callable[[str, LLMResponse], None] | None = None,
    ):
        """Create gateway."""
        # CONTROL LAYER
        self._control = _ControlLayer(
            limits=limits,
            loop_detection=loop_detection,
            tool_access=tool_access or ToolAccessController(),
        )

        # LLM PROVIDERS
        self._providers: dict[str, BaseProvider] = {}
        self._provider_cache: dict[str, LLMProvider] = {}  # Cache enum lookups

        # COORDINATION LAYER
        self._coordination = _CoordinationLayer(
            checkpoint_enabled=checkpoint_enabled,
        )

        # CALL TRACKING (deque for O(1) append)
        self._call_history: deque[CallRecord] = deque(maxlen=10000)
        self._on_before_call = on_before_call
        self._on_after_call = on_after_call

        self._lock = threading.Lock()
        self._started = False

    # =========================================================================
    # PROVIDER CONFIGURATION
    # =========================================================================

    def configure_provider(
        self,
        provider: str | LLMProvider,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "Gateway":
        """Configure an LLM provider."""
        name = provider.value if isinstance(provider, LLMProvider) else provider

        if name == "openai":
            self._providers["openai"] = OpenAIProvider(api_key=api_key, **kwargs)
        elif name == "anthropic":
            self._providers["anthropic"] = AnthropicProvider(api_key=api_key, **kwargs)
        elif name == "gemini":
            self._providers["gemini"] = GeminiProvider(api_key=api_key, **kwargs)
        elif name == "grok":
            self._providers["grok"] = GrokProvider(api_key=api_key, **kwargs)
        elif name == "mock":
            self._providers["mock"] = MockProvider(**kwargs)
        elif name == "smart_mock":
            self._providers["smart_mock"] = SmartMockProvider(**kwargs)
        else:
            raise ProviderNotFoundError(name)

        return self

    def get_provider(self, provider: str | LLMProvider) -> BaseProvider:
        """Get a configured provider."""
        name = provider.value if isinstance(provider, LLMProvider) else provider
        if name not in self._providers:
            raise ProviderNotFoundError(name)
        return self._providers[name]

    # =========================================================================
    # MAIN ENTRY POINT - THE FLOW
    # =========================================================================

    async def call(
        self,
        agent_id: str,
        provider: str | LLMProvider,
        model: str,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        state: SharedState | dict[str, Any] | None = None,
        workflow_id: str | None = None,
        step: int | None = None,
    ) -> LLMResponse:
        """Make an LLM call through the gateway.

        THE FLOW:
        1. CONTROL: Check limits, tools, loops
        2. LLM: Route to provider, get response
        3. COORDINATION: Update state, checkpoint

        Args:
            agent_id: Agent making the call
            provider: LLM provider
            model: Model name
            messages: Conversation messages
            tools: Available tools (optional)
            temperature: Temperature (optional)
            max_tokens: Max tokens (optional)
            state: Shared state for coordination
            workflow_id: Workflow ID for checkpointing
            step: Current step for checkpointing

        Returns:
            LLM response
        """
        # Start tracking if not started
        if not self._started:
            self._control.start()
            self._started = True

        provider_name = provider.value if isinstance(provider, LLMProvider) else provider
        provider_enum = LLMProvider(provider_name)

        # Build request
        request = LLMRequest(
            provider=provider_enum,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # === STEP 1: CONTROL CHECKS ===
        self._control.check_before_call(agent_id, tools)

        if self._on_before_call:
            self._on_before_call(agent_id, request)

        record = CallRecord(
            agent_id=agent_id,
            provider=provider_enum,
            model=model,
            request=request,
        )

        try:
            # ROUTE TO LLM
            provider_instance = self._providers.get(provider_name)
            if not provider_instance:
                raise ProviderNotFoundError(provider_name)
            response = await provider_instance.complete(request)
            record.response = response

            # UPDATE METRICS
            self._control.record_call(response, agent_id, state)

            # CHECKPOINT
            if workflow_id and step is not None and state:
                self._coordination.checkpoint(
                    workflow_id=workflow_id,
                    step=step,
                    agent_id=agent_id,
                    state=state,
                    metrics=self._control.metrics,
                )

            if self._on_after_call:
                self._on_after_call(agent_id, response)

            return response

        except Exception as e:
            record.error = str(e)
            raise

        finally:
            self._call_history.append(record)

    # Alias for backward compatibility
    async def complete(
        self,
        agent_id: str,
        provider: str | LLMProvider,
        model: str,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Alias for call() - backward compatibility."""
        return await self.call(
            agent_id=agent_id,
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            state=state,
        )

    async def call_with_tools(
        self,
        agent_id: str,
        provider: str | LLMProvider,
        model: str,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
        max_iterations: int = 10,
        state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Make LLM call with automatic tool execution loop.

        Handles the full cycle:
        1. Call LLM
        2. If tool calls, execute them
        3. Send results back to LLM
        4. Repeat until done
        """
        current_messages = list(messages)

        for _ in range(max_iterations):
            response = await self.call(
                agent_id=agent_id,
                provider=provider,
                model=model,
                messages=current_messages,
                tools=tools,
                state=state,
            )

            if not response.tool_calls:
                return response

            # Add assistant message
            current_messages.append(
                LLMMessage(role="assistant", content=response.content or "")
            )

            # Execute each tool
            for tool_call in response.tool_calls:
                func = tool_call.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})

                if isinstance(tool_args, str):
                    import json
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                # Check tool access
                self._control.check_tool_access(agent_id, tool_name)

                # Execute
                try:
                    result = tool_executor(tool_name, tool_args)
                except Exception as e:
                    result = f"Error: {e}"

                # Add result
                current_messages.append(
                    LLMMessage(
                        role="tool",
                        content=str(result),
                        tool_call_id=tool_call.get("id"),
                        name=tool_name,
                    )
                )

        return response

    # Alias for backward compatibility
    async def complete_with_tools(self, *args, **kwargs) -> LLMResponse:
        """Alias for call_with_tools() - backward compatibility."""
        return await self.call_with_tools(*args, **kwargs)

    # =========================================================================
    # CONTROL LAYER ACCESS
    # =========================================================================

    def update_limits(self, limits: ExecutionLimits) -> None:
        """Update execution limits."""
        self._control.update_limits(limits)

    def start(self) -> None:
        """Start the gateway (begins time tracking)."""
        if not self._started:
            self._control.start()
            self._started = True

    @property
    def metrics(self) -> ExecutionMetrics:
        """Get current execution metrics."""
        return self._control.metrics

    # =========================================================================
    # COORDINATION LAYER ACCESS
    # =========================================================================

    def get_checkpoint(self, workflow_id: str, step: int | None = None):
        """Get a checkpoint."""
        return self._coordination.get_checkpoint(workflow_id, step)

    def get_resume_point(self, workflow_id: str):
        """Get resume point for a workflow."""
        return self._coordination.get_resume_point(workflow_id)

    # =========================================================================
    # OBSERVABILITY
    # =========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """Get all gateway metrics."""
        metrics = self._control.metrics
        return {
            "total_cost": metrics.total_cost,
            "total_steps": metrics.total_steps,
            "total_tokens": metrics.total_tokens,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "elapsed_seconds": metrics.elapsed_time(),
            "remaining": self._control.get_remaining(),
            "call_count": len(self._call_history),
        }

    def get_call_history(
        self,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[CallRecord]:
        """Get call history, optionally filtered."""
        history = list(self._call_history)
        if agent_id:
            history = [r for r in history if r.agent_id == agent_id]
        if limit:
            history = history[-limit:]
        return history

    def reset(self) -> None:
        """Reset gateway state."""
        self._control.reset()
        self._coordination.reset()
        self._call_history.clear()
        self._started = False


# =============================================================================
# INTERNAL: CONTROL LAYER
# =============================================================================

class _ControlLayer:
    """Internal: Handles limits, loops, tool access."""

    def __init__(
        self,
        limits: ExecutionLimits | None,
        loop_detection: LoopDetectionConfig | None,
        tool_access: ToolAccessController,
    ):
        self._limits = LimitsEnforcer(limits=limits)
        self._loop_detection_enabled = loop_detection is not None
        self._loops = LoopDetector(config=loop_detection) if loop_detection else None
        self._tools = tool_access

    def start(self) -> None:
        self._limits.start()

    def check_before_call(
        self,
        agent_id: str,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Check all control rules before making a call."""
        # Check limits
        self._limits.check_limits()

        # Check tool access
        if tools:
            for tool in tools:
                tool_name = tool.get("name") or tool.get("function", {}).get("name", "")
                if tool_name:
                    self._tools.check_access(agent_id, tool_name)

    def check_tool_access(self, agent_id: str, tool_name: str) -> None:
        """Check if agent can use a tool."""
        self._tools.check_access(agent_id, tool_name)

    def record_call(
        self,
        response: LLMResponse,
        agent_id: str,
        state: SharedState | dict[str, Any] | None,
    ) -> None:
        """Record call metrics and check for loops."""
        # Record cost/tokens
        self._limits.record_cost(
            response.cost,
            response.input_tokens,
            response.output_tokens,
        )
        self._limits.increment_steps()

        # Check for loops (only if enabled)
        if self._loop_detection_enabled and self._loops and state is not None:
            state_dict = state.to_dict() if hasattr(state, "to_dict") else state
            self._loops.record_step(
                agent_id=agent_id,
                input_data={},
                output_data=response.content,
                state=state_dict,
            )
            self._loops.check_for_loops()

    def update_limits(self, limits: ExecutionLimits) -> None:
        self._limits.update_limits(limits)

    def get_remaining(self) -> dict[str, Any]:
        return self._limits.get_remaining()

    @property
    def metrics(self) -> ExecutionMetrics:
        return self._limits.metrics

    def reset(self) -> None:
        self._limits.reset()
        if self._loops:
            self._loops.reset()


# =============================================================================
# INTERNAL: COORDINATION LAYER
# =============================================================================

class _CoordinationLayer:
    """Internal: Handles checkpointing."""

    def __init__(self, checkpoint_enabled: bool = False):
        self._enabled = checkpoint_enabled
        self._manager = CheckpointManager(
            storage=InMemoryCheckpointStorage()
        ) if checkpoint_enabled else None

    def checkpoint(
        self,
        workflow_id: str,
        step: int,
        agent_id: str,
        state: SharedState | dict[str, Any],
        metrics: ExecutionMetrics,
    ) -> None:
        """Create a checkpoint."""
        if not self._enabled or not self._manager:
            return

        # Convert state if needed
        if hasattr(state, "snapshot"):
            state_obj = state
        else:
            state_obj = SharedState(initial_data=state)

        self._manager.create_checkpoint(
            workflow_id=workflow_id,
            step=step,
            agent_id=agent_id,
            status=AgentStatus.COMPLETED,
            state=state_obj,
            metrics=metrics,
        )

    def get_checkpoint(self, workflow_id: str, step: int | None = None):
        if not self._manager:
            return None
        if step is not None:
            return self._manager.get_checkpoint(workflow_id, step)
        return self._manager.get_latest_checkpoint(workflow_id)

    def get_resume_point(self, workflow_id: str):
        if not self._manager:
            return None
        return self._manager.get_resume_point(workflow_id)

    def reset(self) -> None:
        if self._manager:
            self._manager._checkpoints.clear()
