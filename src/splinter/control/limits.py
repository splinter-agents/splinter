"""Execution limits enforcement for Splinter.

This module provides the enforcement mechanism that terminates execution when:
- Total spend exceeds budget
- Total steps exceed limit
- Wall-clock time exceeds timeout

These limits are enforced by the runtime, not by prompts.
"""

import threading
import time
from datetime import datetime
from typing import Callable

from ..exceptions import BudgetExceededError, StepLimitExceededError, TimeLimitExceededError
from ..schemas import ExecutionLimits, ExecutionMetrics


class LimitsEnforcer:
    """Enforces execution limits on agent workflows.

    This class provides hard enforcement of cost, step, and time limits.
    It is designed to be checked before each step/LLM call to prevent runaway execution.

    Thread-safe: All limit checks and updates are protected by a lock.

    Example:
        enforcer = LimitsEnforcer(
            limits=ExecutionLimits(max_budget=10.0, max_steps=100, max_time_seconds=300)
        )

        # Before each step
        enforcer.check_limits()  # Raises if any limit exceeded

        # After each LLM call
        enforcer.record_cost(0.05)
        enforcer.increment_steps()
    """

    def __init__(
        self,
        limits: ExecutionLimits | None = None,
        on_limit_warning: Callable[[str, float, float], None] | None = None,
        warning_threshold: float = 0.8,
    ):
        """Initialize the limits enforcer.

        Args:
            limits: Execution limits to enforce. Can be updated mid-run.
            on_limit_warning: Callback when approaching limits (type, current, limit).
            warning_threshold: Fraction of limit at which to warn (0.8 = 80%).
        """
        self._limits = limits or ExecutionLimits()
        self._metrics = ExecutionMetrics()
        self._lock = threading.RLock()
        self._on_limit_warning = on_limit_warning
        self._warning_threshold = warning_threshold
        self._warnings_issued: set[str] = set()
        self._started = False

    @property
    def limits(self) -> ExecutionLimits:
        """Current execution limits."""
        with self._lock:
            return self._limits

    @property
    def metrics(self) -> ExecutionMetrics:
        """Current execution metrics."""
        with self._lock:
            return self._metrics.model_copy()

    def start(self) -> None:
        """Start tracking execution time."""
        with self._lock:
            if not self._started:
                self._metrics.start_time = datetime.now()
                self._started = True

    def update_limits(self, limits: ExecutionLimits) -> None:
        """Update limits mid-run. Takes effect on next check.

        Args:
            limits: New limits to enforce.
        """
        with self._lock:
            self._limits = limits
            # Clear warnings so they can be re-issued with new limits
            self._warnings_issued.clear()

    def check_limits(self) -> None:
        """Check all limits and raise if any exceeded.

        This should be called before each step/LLM call.

        Raises:
            BudgetExceededError: If budget limit exceeded.
            StepLimitExceededError: If step limit exceeded.
            TimeLimitExceededError: If time limit exceeded.
        """
        with self._lock:
            self._check_budget()
            self._check_steps()
            self._check_time()

    def _check_budget(self) -> None:
        """Check budget limit."""
        if self._limits.max_budget is not None:
            current = self._metrics.total_cost
            limit = self._limits.max_budget

            if current >= limit:
                raise BudgetExceededError(current, limit)

            self._check_warning("budget", current, limit)

    def _check_steps(self) -> None:
        """Check step limit."""
        if self._limits.max_steps is not None:
            current = self._metrics.total_steps
            limit = self._limits.max_steps

            if current >= limit:
                raise StepLimitExceededError(current, limit)

            self._check_warning("steps", float(current), float(limit))

    def _check_time(self) -> None:
        """Check time limit."""
        if self._limits.max_time_seconds is not None and self._metrics.start_time is not None:
            elapsed = (datetime.now() - self._metrics.start_time).total_seconds()
            self._metrics.elapsed_seconds = elapsed
            limit = self._limits.max_time_seconds

            if elapsed >= limit:
                raise TimeLimitExceededError(elapsed, limit)

            self._check_warning("time", elapsed, limit)

    def _check_warning(self, limit_type: str, current: float, limit: float) -> None:
        """Check if we should issue a warning for approaching limit."""
        if limit_type in self._warnings_issued:
            return

        if self._on_limit_warning and current >= limit * self._warning_threshold:
            self._warnings_issued.add(limit_type)
            self._on_limit_warning(limit_type, current, limit)

    def record_cost(self, cost: float, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record cost from an LLM call.

        Args:
            cost: Cost in dollars.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        with self._lock:
            self._metrics.total_cost += cost
            self._metrics.input_tokens += input_tokens
            self._metrics.output_tokens += output_tokens
            self._metrics.total_tokens += input_tokens + output_tokens

    def increment_steps(self, count: int = 1) -> None:
        """Increment the step counter.

        Args:
            count: Number of steps to add.
        """
        with self._lock:
            self._metrics.total_steps += count

    def get_remaining(self) -> dict[str, float | None]:
        """Get remaining budget/steps/time.

        Returns:
            Dict with remaining values (None if no limit set).
        """
        with self._lock:
            remaining_budget = None
            remaining_steps = None
            remaining_time = None

            if self._limits.max_budget is not None:
                remaining_budget = max(0, self._limits.max_budget - self._metrics.total_cost)

            if self._limits.max_steps is not None:
                remaining_steps = max(0, self._limits.max_steps - self._metrics.total_steps)

            if self._limits.max_time_seconds is not None and self._metrics.start_time:
                elapsed = (datetime.now() - self._metrics.start_time).total_seconds()
                remaining_time = max(0, self._limits.max_time_seconds - elapsed)

            return {
                "budget": remaining_budget,
                "steps": remaining_steps,
                "time": remaining_time,
            }

    def reset(self) -> None:
        """Reset metrics to initial state."""
        with self._lock:
            self._metrics = ExecutionMetrics()
            self._warnings_issued.clear()
            self._started = False


class PerAgentLimits:
    """Tracks and enforces per-agent limits within a workflow.

    Each agent can have its own step and budget limits that are enforced
    independently of the workflow-level limits.
    """

    def __init__(
        self,
        default_max_steps: int | None = None,
        default_max_budget: float | None = None,
    ):
        """Initialize per-agent limits tracker.

        Args:
            default_max_steps: Default max steps per agent.
            default_max_budget: Default max budget per agent.
        """
        self._default_max_steps = default_max_steps
        self._default_max_budget = default_max_budget
        self._agent_limits: dict[str, ExecutionLimits] = {}
        self._agent_metrics: dict[str, ExecutionMetrics] = {}
        self._lock = threading.RLock()

    def set_agent_limits(self, agent_id: str, limits: ExecutionLimits) -> None:
        """Set limits for a specific agent."""
        with self._lock:
            self._agent_limits[agent_id] = limits

    def get_agent_metrics(self, agent_id: str) -> ExecutionMetrics:
        """Get metrics for a specific agent."""
        with self._lock:
            if agent_id not in self._agent_metrics:
                self._agent_metrics[agent_id] = ExecutionMetrics()
            return self._agent_metrics[agent_id].model_copy()

    def check_agent_limits(self, agent_id: str) -> None:
        """Check limits for a specific agent.

        Raises:
            BudgetExceededError: If agent budget exceeded.
            StepLimitExceededError: If agent step limit exceeded.
        """
        with self._lock:
            limits = self._agent_limits.get(agent_id)
            metrics = self._agent_metrics.get(agent_id, ExecutionMetrics())

            # Check step limit
            max_steps = (
                limits.max_steps if limits and limits.max_steps is not None
                else self._default_max_steps
            )
            if max_steps is not None and metrics.total_steps >= max_steps:
                raise StepLimitExceededError(metrics.total_steps, max_steps)

            # Check budget limit
            max_budget = (
                limits.max_budget if limits and limits.max_budget is not None
                else self._default_max_budget
            )
            if max_budget is not None and metrics.total_cost >= max_budget:
                raise BudgetExceededError(metrics.total_cost, max_budget)

    def record_agent_step(
        self, agent_id: str, cost: float = 0, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Record a step for an agent.

        Args:
            agent_id: The agent ID.
            cost: Cost of this step.
            input_tokens: Input tokens used.
            output_tokens: Output tokens used.
        """
        with self._lock:
            if agent_id not in self._agent_metrics:
                self._agent_metrics[agent_id] = ExecutionMetrics()

            metrics = self._agent_metrics[agent_id]
            metrics.total_steps += 1
            metrics.total_cost += cost
            metrics.input_tokens += input_tokens
            metrics.output_tokens += output_tokens
            metrics.total_tokens += input_tokens + output_tokens

    def reset_agent(self, agent_id: str) -> None:
        """Reset metrics for a specific agent."""
        with self._lock:
            if agent_id in self._agent_metrics:
                self._agent_metrics[agent_id] = ExecutionMetrics()
