"""Retry strategy for Splinter.

Provides retry mechanisms with configurable strategies:
- Basic: Simple retry with backoff
- Fail-Closed: Fail immediately on certain errors, retry others
"""

import asyncio
import logging
import random
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Type

from ..exceptions import SplinterError

logger = logging.getLogger(__name__)


class RetryExhaustedError(SplinterError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_error}")


class RetryMode(str, Enum):
    """Retry mode."""

    BASIC = "basic"  # Retry all errors
    FAIL_CLOSED = "fail_closed"  # Fail immediately on critical errors


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to delay
    mode: RetryMode = RetryMode.BASIC

    # Errors that should NOT be retried (fail immediately)
    fail_fast_errors: list[Type[Exception]] = field(default_factory=list)

    # Errors that SHOULD be retried
    retryable_errors: list[Type[Exception]] = field(default_factory=list)


class RetryStrategy:
    """Handles retry logic for operations.

    Example:
        strategy = RetryStrategy(config=RetryConfig(max_attempts=3))

        # Retry a function
        result = await strategy.execute(async_function, arg1, arg2)

        # Or use as decorator
        @strategy.wrap
        async def my_function():
            ...
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        on_failure: Callable[[int, Exception], None] | None = None,
    ):
        """Initialize retry strategy.

        Args:
            config: Retry configuration.
            on_retry: Callback on retry (attempt, error, delay).
            on_failure: Callback on final failure (attempts, error).
        """
        self._config = config or RetryConfig()
        self._on_retry = on_retry
        self._on_failure = on_failure
        self._lock = threading.RLock()
        self._stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
        }

    def _should_retry(self, error: Exception) -> bool:
        """Determine if error should be retried."""
        # Fail-fast errors never retry
        for error_type in self._config.fail_fast_errors:
            if isinstance(error, error_type):
                return False

        # In fail-closed mode, only retry explicitly retryable errors
        if self._config.mode == RetryMode.FAIL_CLOSED:
            if not self._config.retryable_errors:
                return False
            return any(
                isinstance(error, e) for e in self._config.retryable_errors
            )

        # In basic mode, retry unless explicitly fail-fast
        if self._config.retryable_errors:
            return any(
                isinstance(error, e) for e in self._config.retryable_errors
            )

        return True

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry."""
        delay = self._config.initial_delay * (
            self._config.exponential_base ** (attempt - 1)
        )
        delay = min(delay, self._config.max_delay)

        if self._config.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            RetryExhaustedError: If all attempts fail.
        """
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_attempts + 1):
            with self._lock:
                self._stats["total_attempts"] += 1

            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    with self._lock:
                        self._stats["successful_retries"] += 1
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")

                # Check if we should retry
                if not self._should_retry(e):
                    logger.error(f"Non-retryable error: {e}")
                    with self._lock:
                        self._stats["failed_retries"] += 1
                    if self._on_failure:
                        self._on_failure(attempt, e)
                    raise

                # Check if we have more attempts
                if attempt >= self._config.max_attempts:
                    with self._lock:
                        self._stats["failed_retries"] += 1
                    if self._on_failure:
                        self._on_failure(attempt, e)
                    raise RetryExhaustedError(attempt, e)

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.info(f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{self._config.max_attempts})")

                if self._on_retry:
                    self._on_retry(attempt, e, delay)

                await asyncio.sleep(delay)

        # Should never reach here
        raise RetryExhaustedError(self._config.max_attempts, last_error)

    def execute_sync(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute synchronous function with retry logic."""
        import time

        last_error: Exception | None = None

        for attempt in range(1, self._config.max_attempts + 1):
            with self._lock:
                self._stats["total_attempts"] += 1

            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    with self._lock:
                        self._stats["successful_retries"] += 1
                return result

            except Exception as e:
                last_error = e

                if not self._should_retry(e):
                    with self._lock:
                        self._stats["failed_retries"] += 1
                    raise

                if attempt >= self._config.max_attempts:
                    with self._lock:
                        self._stats["failed_retries"] += 1
                    raise RetryExhaustedError(attempt, e)

                delay = self._calculate_delay(attempt)
                if self._on_retry:
                    self._on_retry(attempt, e, delay)

                time.sleep(delay)

        raise RetryExhaustedError(self._config.max_attempts, last_error)

    def wrap(self, func: Callable) -> Callable:
        """Decorator to wrap function with retry logic."""
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute(func, *args, **kwargs)

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.execute_sync(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def get_stats(self) -> dict[str, int]:
        """Get retry statistics."""
        with self._lock:
            return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = {
                "total_attempts": 0,
                "successful_retries": 0,
                "failed_retries": 0,
            }


# Convenience function
def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    mode: RetryMode = RetryMode.BASIC,
) -> Callable:
    """Decorator factory for retry logic.

    Example:
        @with_retry(max_attempts=3)
        async def my_function():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        mode=mode,
    )
    strategy = RetryStrategy(config=config)
    return strategy.wrap
