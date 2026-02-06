"""Logging utilities for Splinter."""

import logging
import sys
from typing import Any


def configure_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """Configure logging for Splinter.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string: Custom format string.
        handler: Custom handler. Defaults to StreamHandler.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(logging.Formatter(format_string))

    # Configure Splinter logger
    logger = logging.getLogger("splinter")
    logger.setLevel(level)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a Splinter module.

    Args:
        name: Module name (e.g., "gateway", "workflow").

    Returns:
        Configured logger.
    """
    return logging.getLogger(f"splinter.{name}")


class StructuredLogger:
    """Logger that outputs structured data for monitoring systems."""

    def __init__(self, name: str):
        """Initialize structured logger.

        Args:
            name: Logger name.
        """
        self._logger = get_logger(name)
        self._context: dict[str, Any] = {}

    def with_context(self, **kwargs: Any) -> "StructuredLogger":
        """Add context to all log messages.

        Args:
            **kwargs: Context key-value pairs.

        Returns:
            Self for chaining.
        """
        self._context.update(kwargs)
        return self

    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format message with context."""
        data = {**self._context, **kwargs}
        if data:
            pairs = [f"{k}={v}" for k, v in data.items()]
            return f"{message} | {' '.join(pairs)}"
        return message

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.error(self._format_message(message, **kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        self._logger.critical(self._format_message(message, **kwargs))
