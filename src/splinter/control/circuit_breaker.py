"""Circuit breakers for Splinter.

Global safety triggers that halt execution when critical thresholds are exceeded.
Prevents cascading failures and protects resources.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..exceptions import SplinterError


class CircuitOpenError(SplinterError):
    """Raised when circuit breaker is open."""

    def __init__(self, breaker_id: str, reason: str):
        self.breaker_id = breaker_id
        self.reason = reason
        super().__init__(f"Circuit breaker '{breaker_id}' is OPEN: {reason}")


class CircuitState(str, Enum):
    """State of a circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing (in half-open)
    timeout_seconds: float = 60.0  # Time before trying half-open

    # Optional: Trip on specific conditions
    error_rate_threshold: float | None = None  # e.g., 0.5 = 50% error rate
    min_calls_for_rate: int = 10  # Min calls before error rate applies


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float | None = None
    last_state_change: float = field(default_factory=time.time)
    state: CircuitState = CircuitState.CLOSED

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitBreaker:
    """Circuit breaker for protecting resources.

    Example:
        breaker = CircuitBreaker(
            breaker_id="llm_api",
            config=CircuitBreakerConfig(failure_threshold=5)
        )

        # Check before making call
        breaker.check()  # Raises CircuitOpenError if open

        try:
            result = await make_api_call()
            breaker.record_success()
        except Exception as e:
            breaker.record_failure()
            raise
    """

    def __init__(
        self,
        breaker_id: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
        on_trip: Callable[[str, str], None] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            breaker_id: Unique identifier for this breaker.
            config: Circuit breaker configuration.
            on_state_change: Callback on state change (id, old_state, new_state).
            on_trip: Callback when circuit trips open (id, reason).
        """
        self._id = breaker_id
        self._config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change
        self._on_trip = on_trip
        self._stats = CircuitStats()
        self._lock = threading.RLock()

    @property
    def breaker_id(self) -> str:
        return self._id

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._stats.state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def check(self) -> bool:
        """Check if circuit allows calls.

        Returns:
            True if allowed.

        Raises:
            CircuitOpenError: If circuit is open.
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._stats.state == CircuitState.OPEN:
                time_since_open = time.time() - self._stats.last_state_change
                if time_since_open >= self._config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    raise CircuitOpenError(
                        self._id,
                        f"Waiting {self._config.timeout_seconds - time_since_open:.1f}s before retry"
                    )

            return True

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes += 1

            # In HALF_OPEN, check if we can close the circuit
            if self._stats.state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_successes = 0
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = time.time()

            # Check if we should trip the circuit
            should_trip = False
            reason = ""

            # Check consecutive failures
            if self._stats.consecutive_failures >= self._config.failure_threshold:
                should_trip = True
                reason = f"Consecutive failures: {self._stats.consecutive_failures}"

            # Check error rate
            if (
                self._config.error_rate_threshold is not None
                and self._stats.total_calls >= self._config.min_calls_for_rate
            ):
                if self._stats.error_rate >= self._config.error_rate_threshold:
                    should_trip = True
                    reason = f"Error rate: {self._stats.error_rate:.1%}"

            # In HALF_OPEN, single failure trips back to OPEN
            if self._stats.state == CircuitState.HALF_OPEN:
                should_trip = True
                reason = "Failure in half-open state"

            if should_trip and self._stats.state != CircuitState.OPEN:
                self._trip(reason)

    def _trip(self, reason: str) -> None:
        """Trip the circuit to OPEN state."""
        self._transition_to(CircuitState.OPEN)
        if self._on_trip:
            self._on_trip(self._id, reason)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._stats.state
        if old_state == new_state:
            return

        self._stats.state = new_state
        self._stats.last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._stats.consecutive_successes = 0
            self._stats.consecutive_failures = 0

        if self._on_state_change:
            self._on_state_change(self._id, old_state, new_state)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._stats = CircuitStats()

    def force_open(self, reason: str = "Manual override") -> None:
        """Force circuit to open state."""
        with self._lock:
            self._trip(reason)

    def force_close(self) -> None:
        """Force circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "breaker_id": self._id,
                "state": self._stats.state.value,
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "error_rate": self._stats.error_rate,
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "last_failure_time": self._stats.last_failure_time,
                "last_state_change": self._stats.last_state_change,
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.

    Example:
        registry = CircuitBreakerRegistry()

        # Register breakers
        registry.register("llm_api", config=CircuitBreakerConfig(failure_threshold=5))
        registry.register("database", config=CircuitBreakerConfig(failure_threshold=3))

        # Check all breakers
        registry.check_all()  # Raises if any are open

        # Global trip (emergency stop)
        registry.trip_all("Emergency shutdown")
    """

    def __init__(
        self,
        on_any_trip: Callable[[str, str], None] | None = None,
    ):
        """Initialize registry.

        Args:
            on_any_trip: Callback when any breaker trips.
        """
        self._breakers: dict[str, CircuitBreaker] = {}
        self._on_any_trip = on_any_trip
        self._lock = threading.RLock()

    def register(
        self,
        breaker_id: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        with self._lock:
            def on_trip(bid: str, reason: str) -> None:
                if self._on_any_trip:
                    self._on_any_trip(bid, reason)

            breaker = CircuitBreaker(
                breaker_id=breaker_id,
                config=config,
                on_trip=on_trip,
            )
            self._breakers[breaker_id] = breaker
            return breaker

    def get(self, breaker_id: str) -> CircuitBreaker | None:
        """Get a circuit breaker by ID."""
        with self._lock:
            return self._breakers.get(breaker_id)

    def check(self, breaker_id: str) -> bool:
        """Check a specific breaker."""
        with self._lock:
            breaker = self._breakers.get(breaker_id)
            if breaker is None:
                return True
            return breaker.check()

    def check_all(self) -> bool:
        """Check all breakers.

        Raises:
            CircuitOpenError: If any breaker is open.
        """
        with self._lock:
            for breaker in self._breakers.values():
                breaker.check()
            return True

    def trip_all(self, reason: str = "Global trip") -> None:
        """Trip all circuit breakers (emergency stop)."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_open(reason)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all breakers."""
        with self._lock:
            return {
                bid: breaker.get_stats()
                for bid, breaker in self._breakers.items()
            }

    def get_open_breakers(self) -> list[str]:
        """Get list of open breaker IDs."""
        with self._lock:
            return [
                bid for bid, breaker in self._breakers.items()
                if breaker.is_open
            ]
