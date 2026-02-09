"""Loop detection for Splinter.

This module detects no-progress execution based on:
- Repeated identical outputs
- No shared-state changes across steps
- Repeating step signatures
- A↔B ping-pong patterns

This saves cost, time, and human intervention by stopping wasted loops
before the step limit is hit.
"""

import hashlib
import json
import threading
from collections import Counter, deque
from datetime import datetime
from typing import Any, Callable

from ..exceptions import LoopDetectedError
from ..schemas import LoopDetectionConfig, StepSignature


def _hash_value(value: Any) -> str:
    """Create a stable hash of any value."""
    try:
        serialized = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(value)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class LoopDetector:
    """Detects loop patterns in agent execution.

    This class analyzes execution patterns to detect:
    1. Repeated identical outputs from the same agent
    2. No state changes across multiple steps
    3. Ping-pong patterns between agents (A → B → A → B...)
    4. Repeated tool calls with same arguments

    Example:
        detector = LoopDetector(config=LoopDetectionConfig(
            max_repeated_outputs=3,
            max_no_state_change=5,
        ))

        # After each step
        detector.record_step(
            agent_id="researcher",
            input_data={"query": "search term"},
            output_data={"results": [...]},
            state=current_state,
            tool_name="web_search",
        )

        # Check for loops
        detector.check_for_loops()  # Raises LoopDetectedError if detected
    """

    def __init__(
        self,
        config: LoopDetectionConfig | None = None,
        on_loop_warning: Callable[[str, int], None] | None = None,
    ):
        """Initialize the loop detector.

        Args:
            config: Loop detection configuration.
            on_loop_warning: Callback when loop pattern detected (pattern, count).
        """
        self._config = config or LoopDetectionConfig()
        self._on_loop_warning = on_loop_warning
        self._lock = threading.RLock()

        # Sliding window of recent step signatures
        self._step_history: deque[StepSignature] = deque(maxlen=self._config.window_size)

        # Track output hashes per agent
        self._output_history: dict[str, deque[str]] = {}

        # Track state hashes for no-progress detection
        self._state_hashes: deque[str] = deque(maxlen=self._config.window_size)

        # Track agent execution order for ping-pong detection
        self._agent_sequence: deque[str] = deque(maxlen=self._config.window_size)

        # Track tool calls for repeated tool detection
        self._tool_call_history: deque[tuple[str, str, str]] = deque(
            maxlen=self._config.window_size
        )

    def update_config(self, config: LoopDetectionConfig) -> None:
        """Update loop detection configuration."""
        with self._lock:
            self._config = config

    def record_step(
        self,
        agent_id: str,
        input_data: Any,
        output_data: Any,
        state: dict[str, Any],
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> StepSignature:
        """Record a step for loop detection analysis.

        Args:
            agent_id: The agent that executed this step.
            input_data: Input to the agent/tool.
            output_data: Output from the agent/tool.
            state: Current shared state after this step.
            tool_name: Name of tool called (if any).
            tool_args: Arguments passed to tool (if any).

        Returns:
            The step signature created.
        """
        with self._lock:
            input_hash = _hash_value(input_data)
            output_hash = _hash_value(output_data)
            state_hash = _hash_value(state)

            signature = StepSignature(
                agent_id=agent_id,
                tool_name=tool_name,
                input_hash=input_hash,
                output_hash=output_hash,
                state_hash=state_hash,
                timestamp=datetime.now(),
            )

            self._step_history.append(signature)
            self._state_hashes.append(state_hash)
            self._agent_sequence.append(agent_id)

            # Track outputs per agent
            if agent_id not in self._output_history:
                self._output_history[agent_id] = deque(maxlen=self._config.window_size)
            self._output_history[agent_id].append(output_hash)

            # Track tool calls
            if tool_name:
                args_hash = _hash_value(tool_args) if tool_args else ""
                self._tool_call_history.append((agent_id, tool_name, args_hash))

            return signature

    def check_for_loops(self) -> None:
        """Check for loop patterns and raise if detected.

        Raises:
            LoopDetectedError: If a loop pattern is detected.
        """
        with self._lock:
            self._check_repeated_outputs()
            self._check_no_state_change()
            self._check_ping_pong()
            self._check_repeated_tool_calls()

    def _check_repeated_outputs(self) -> None:
        """Check for repeated identical outputs from any agent."""
        for agent_id, outputs in self._output_history.items():
            if len(outputs) < self._config.max_repeated_outputs:
                continue

            # Check last N outputs for identical values
            recent = list(outputs)[-self._config.max_repeated_outputs :]
            if len(set(recent)) == 1:
                pattern = f"Agent '{agent_id}' produced identical output"
                if self._on_loop_warning:
                    self._on_loop_warning(pattern, self._config.max_repeated_outputs)
                raise LoopDetectedError(pattern, self._config.max_repeated_outputs)

    def _check_no_state_change(self) -> None:
        """Check for no state changes across multiple steps."""
        if len(self._state_hashes) < self._config.max_no_state_change:
            return

        recent = list(self._state_hashes)[-self._config.max_no_state_change :]
        if len(set(recent)) == 1:
            pattern = "No state change detected"
            if self._on_loop_warning:
                self._on_loop_warning(pattern, self._config.max_no_state_change)
            raise LoopDetectedError(pattern, self._config.max_no_state_change)

    def _check_ping_pong(self) -> None:
        """Check for A↔B ping-pong patterns."""
        if len(self._agent_sequence) < self._config.max_ping_pong * 2:
            return

        recent = list(self._agent_sequence)[-(self._config.max_ping_pong * 2) :]

        # Look for alternating pattern
        unique_agents = set(recent)
        if len(unique_agents) == 2:
            agents = list(unique_agents)
            expected = [agents[i % 2] for i in range(len(recent))]
            expected_reverse = [agents[(i + 1) % 2] for i in range(len(recent))]

            if recent == expected or recent == expected_reverse:
                # Verify state isn't progressing
                recent_states = list(self._state_hashes)[-(self._config.max_ping_pong * 2) :]
                if len(set(recent_states)) <= 2:  # State oscillating between at most 2 values
                    pattern = f"Ping-pong between agents {agents[0]} ↔ {agents[1]}"
                    if self._on_loop_warning:
                        self._on_loop_warning(pattern, self._config.max_ping_pong)
                    raise LoopDetectedError(pattern, self._config.max_ping_pong)

    def _check_repeated_tool_calls(self) -> None:
        """Check for repeated identical tool calls."""
        if len(self._tool_call_history) < self._config.max_repeated_outputs:
            return

        recent = list(self._tool_call_history)[-self._config.max_repeated_outputs :]

        # Check if all recent tool calls are identical
        if len(set(recent)) == 1 and recent[0][1]:  # Has tool name
            agent_id, tool_name, _ = recent[0]
            pattern = f"Agent '{agent_id}' repeatedly calling tool '{tool_name}'"
            if self._on_loop_warning:
                self._on_loop_warning(pattern, self._config.max_repeated_outputs)
            raise LoopDetectedError(pattern, self._config.max_repeated_outputs)

    def get_pattern_analysis(self) -> dict[str, Any]:
        """Get analysis of current execution patterns.

        Returns:
            Dict containing pattern analysis for debugging.
        """
        with self._lock:
            # Output repetition analysis
            output_analysis = {}
            for agent_id, outputs in self._output_history.items():
                output_list = list(outputs)
                counter = Counter(output_list)
                most_common = counter.most_common(3)
                output_analysis[agent_id] = {
                    "total_outputs": len(output_list),
                    "unique_outputs": len(counter),
                    "most_repeated": most_common,
                }

            # State change analysis
            state_list = list(self._state_hashes)
            state_counter = Counter(state_list)

            # Agent sequence analysis
            agent_list = list(self._agent_sequence)
            agent_counter = Counter(agent_list)

            return {
                "total_steps": len(self._step_history),
                "output_analysis": output_analysis,
                "state_analysis": {
                    "total_states": len(state_list),
                    "unique_states": len(state_counter),
                    "most_common_state": state_counter.most_common(1),
                },
                "agent_sequence": {
                    "recent_agents": agent_list[-10:] if agent_list else [],
                    "agent_distribution": dict(agent_counter),
                },
                "tool_calls": {
                    "recent": list(self._tool_call_history)[-5:],
                },
            }

    def reset(self) -> None:
        """Reset all loop detection state."""
        with self._lock:
            self._step_history.clear()
            self._output_history.clear()
            self._state_hashes.clear()
            self._agent_sequence.clear()
            self._tool_call_history.clear()


class LoopBreaker:
    """Strategies for breaking out of detected loops.

    This class provides mechanisms to recover from loops without
    completely terminating the workflow.
    """

    def __init__(
        self,
        max_retries: int = 3,
        on_loop_break: Callable[[str, str], None] | None = None,
    ):
        """Initialize the loop breaker.

        Args:
            max_retries: Maximum retry attempts before giving up.
            on_loop_break: Callback when loop is broken (strategy, reason).
        """
        self._max_retries = max_retries
        self._on_loop_break = on_loop_break
        self._retry_counts: dict[str, int] = {}
        self._lock = threading.RLock()

    def should_retry(self, agent_id: str, error: LoopDetectedError) -> bool:
        """Check if we should retry after a loop detection.

        Args:
            agent_id: The agent that triggered the loop.
            error: The loop detection error.

        Returns:
            True if should retry, False if should fail.
        """
        with self._lock:
            key = f"{agent_id}:{error.pattern}"
            self._retry_counts[key] = self._retry_counts.get(key, 0) + 1
            return self._retry_counts[key] <= self._max_retries

    def get_break_strategy(self, error: LoopDetectedError) -> dict[str, Any]:
        """Get suggested strategy for breaking out of a loop.

        Args:
            error: The loop detection error.

        Returns:
            Strategy dict with suggested actions.
        """
        strategies = []

        if "identical output" in error.pattern:
            strategies.append({
                "action": "modify_prompt",
                "reason": "Agent producing same output - try adding variation to prompt",
            })
            strategies.append({
                "action": "skip_agent",
                "reason": "Agent may have completed its task",
            })

        if "No state change" in error.pattern:
            strategies.append({
                "action": "inject_state",
                "reason": "No progress being made - consider manual state update",
            })

        if "Ping-pong" in error.pattern:
            strategies.append({
                "action": "break_cycle",
                "reason": "Agents in deadlock - skip one agent's turn",
            })
            strategies.append({
                "action": "add_arbiter",
                "reason": "Add third agent to break the tie",
            })

        if "repeatedly calling tool" in error.pattern:
            strategies.append({
                "action": "disable_tool",
                "reason": "Tool call not making progress - temporarily disable",
            })

        return {
            "pattern": error.pattern,
            "occurrences": error.occurrences,
            "strategies": strategies,
        }

    def reset(self) -> None:
        """Reset retry counts."""
        with self._lock:
            self._retry_counts.clear()
