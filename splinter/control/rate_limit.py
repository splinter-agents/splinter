"""Rate limiting for Splinter.

Enforces rate limits per agent and per tool to prevent abuse and ensure fair resource usage.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from ..exceptions import SplinterError


class RateLimitExceededError(SplinterError):
    """Raised when rate limit is exceeded."""

    def __init__(self, entity_type: str, entity_id: str, limit: int, window: float):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded for {entity_type} '{entity_id}': "
            f"{limit} calls per {window}s"
        )


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""

    calls: int  # Max calls allowed
    window_seconds: float  # Time window
    burst: int | None = None  # Allow burst above limit (optional)


@dataclass
class RateLimitState:
    """State for tracking rate limits."""

    timestamps: list[float] = field(default_factory=list)

    def clean(self, window: float) -> None:
        """Remove timestamps outside the window."""
        cutoff = time.time() - window
        self.timestamps = [t for t in self.timestamps if t > cutoff]

    def count(self, window: float) -> int:
        """Count calls within window."""
        self.clean(window)
        return len(self.timestamps)

    def add(self) -> None:
        """Add current timestamp."""
        self.timestamps.append(time.time())


class RateLimiter:
    """Rate limiter for agents and tools.

    Example:
        limiter = RateLimiter()

        # Set agent rate limit
        limiter.set_agent_limit("researcher", calls=10, window_seconds=60)

        # Set tool rate limit
        limiter.set_tool_limit("web_search", calls=5, window_seconds=60)

        # Check before making call
        limiter.check_agent("researcher")  # Raises RateLimitExceededError if exceeded
        limiter.record_agent_call("researcher")
    """

    def __init__(
        self,
        default_agent_limit: RateLimitConfig | None = None,
        default_tool_limit: RateLimitConfig | None = None,
        on_limit_exceeded: Callable[[str, str, int], None] | None = None,
    ):
        """Initialize rate limiter.

        Args:
            default_agent_limit: Default limit for agents without specific config.
            default_tool_limit: Default limit for tools without specific config.
            on_limit_exceeded: Callback when limit exceeded (entity_type, entity_id, count).
        """
        self._default_agent_limit = default_agent_limit
        self._default_tool_limit = default_tool_limit
        self._on_limit_exceeded = on_limit_exceeded

        self._agent_limits: dict[str, RateLimitConfig] = {}
        self._tool_limits: dict[str, RateLimitConfig] = {}
        self._agent_state: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._tool_state: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = threading.RLock()

    def set_agent_limit(
        self,
        agent_id: str,
        calls: int,
        window_seconds: float,
        burst: int | None = None,
    ) -> None:
        """Set rate limit for an agent."""
        with self._lock:
            self._agent_limits[agent_id] = RateLimitConfig(
                calls=calls,
                window_seconds=window_seconds,
                burst=burst,
            )

    def set_tool_limit(
        self,
        tool_name: str,
        calls: int,
        window_seconds: float,
        burst: int | None = None,
    ) -> None:
        """Set rate limit for a tool."""
        with self._lock:
            self._tool_limits[tool_name] = RateLimitConfig(
                calls=calls,
                window_seconds=window_seconds,
                burst=burst,
            )

    def check_agent(self, agent_id: str) -> bool:
        """Check if agent call is within rate limit.

        Returns:
            True if allowed.

        Raises:
            RateLimitExceededError: If rate limit exceeded.
        """
        with self._lock:
            config = self._agent_limits.get(agent_id, self._default_agent_limit)
            if config is None:
                return True

            state = self._agent_state[agent_id]
            current = state.count(config.window_seconds)
            limit = config.burst or config.calls

            if current >= limit:
                if self._on_limit_exceeded:
                    self._on_limit_exceeded("agent", agent_id, current)
                raise RateLimitExceededError(
                    "agent", agent_id, config.calls, config.window_seconds
                )

            return True

    def check_tool(self, tool_name: str) -> bool:
        """Check if tool call is within rate limit."""
        with self._lock:
            config = self._tool_limits.get(tool_name, self._default_tool_limit)
            if config is None:
                return True

            state = self._tool_state[tool_name]
            current = state.count(config.window_seconds)
            limit = config.burst or config.calls

            if current >= limit:
                if self._on_limit_exceeded:
                    self._on_limit_exceeded("tool", tool_name, current)
                raise RateLimitExceededError(
                    "tool", tool_name, config.calls, config.window_seconds
                )

            return True

    def record_agent_call(self, agent_id: str) -> None:
        """Record an agent call."""
        with self._lock:
            self._agent_state[agent_id].add()

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        with self._lock:
            self._tool_state[tool_name].add()

    def get_agent_usage(self, agent_id: str) -> dict[str, Any]:
        """Get rate limit usage for an agent."""
        with self._lock:
            config = self._agent_limits.get(agent_id, self._default_agent_limit)
            if config is None:
                return {"limited": False}

            state = self._agent_state[agent_id]
            current = state.count(config.window_seconds)

            return {
                "limited": True,
                "current": current,
                "limit": config.calls,
                "window_seconds": config.window_seconds,
                "remaining": max(0, config.calls - current),
            }

    def get_tool_usage(self, tool_name: str) -> dict[str, Any]:
        """Get rate limit usage for a tool."""
        with self._lock:
            config = self._tool_limits.get(tool_name, self._default_tool_limit)
            if config is None:
                return {"limited": False}

            state = self._tool_state[tool_name]
            current = state.count(config.window_seconds)

            return {
                "limited": True,
                "current": current,
                "limit": config.calls,
                "window_seconds": config.window_seconds,
                "remaining": max(0, config.calls - current),
            }

    def reset(self, agent_id: str | None = None, tool_name: str | None = None) -> None:
        """Reset rate limit state."""
        with self._lock:
            if agent_id:
                self._agent_state[agent_id] = RateLimitState()
            elif tool_name:
                self._tool_state[tool_name] = RateLimitState()
            else:
                self._agent_state.clear()
                self._tool_state.clear()
