"""Memory limits for Splinter.

This module provides hard caps on:
- Total memory size
- Memory lifetime (TTL)
- Eviction strategy (FIFO, LRU, TTL)

Agent memory grows unbounded and silently kills cost, context relevance,
and predictability. This module prevents that.
"""

import json
import sys
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator

from ..exceptions import MemoryLimitError
from ..types import EvictionPolicy, MemoryConfig, MemoryEntry


def _estimate_size(value: Any) -> int:
    """Estimate the size of a value in bytes."""
    try:
        # Try JSON serialization for accurate size
        return len(json.dumps(value, default=str).encode())
    except (TypeError, ValueError):
        # Fall back to sys.getsizeof (less accurate for complex objects)
        return sys.getsizeof(value)


class MemoryStore:
    """Memory store with size limits, TTL, and eviction.

    This class provides bounded memory storage for agent context,
    automatically evicting entries when limits are exceeded.

    Example:
        store = MemoryStore(config=MemoryConfig(
            max_size_bytes=1024 * 1024,  # 1MB
            ttl_seconds=3600,  # 1 hour
            eviction_policy=EvictionPolicy.FIFO,
        ))

        # Store values
        store.set("key1", {"data": "value"})
        store.set("key2", [1, 2, 3], agent_id="researcher")

        # Retrieve values
        value = store.get("key1")

        # Values automatically expire and evict
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        on_eviction: Callable[[str, Any], None] | None = None,
        on_overflow: Callable[[int, int], None] | None = None,
    ):
        """Initialize the memory store.

        Args:
            config: Memory configuration.
            on_eviction: Callback when entry is evicted (key, value).
            on_overflow: Callback when memory limit approached (current, limit).
        """
        self._config = config or MemoryConfig()
        self._on_eviction = on_eviction
        self._on_overflow = on_overflow

        # Use OrderedDict to maintain insertion order for FIFO
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._current_size = 0
        self._lock = threading.RLock()

    @property
    def config(self) -> MemoryConfig:
        """Current memory configuration."""
        return self._config

    @property
    def current_size(self) -> int:
        """Current memory usage in bytes."""
        with self._lock:
            return self._current_size

    @property
    def entry_count(self) -> int:
        """Current number of entries."""
        with self._lock:
            return len(self._entries)

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        agent_id: str | None = None,
    ) -> bool:
        """Store a value in memory.

        Args:
            key: The key to store under.
            value: The value to store.
            ttl_seconds: Optional TTL override for this entry.
            agent_id: Optional agent that owns this entry.

        Returns:
            True if stored successfully, False if rejected due to size.

        Raises:
            MemoryLimitError: If single value exceeds max size.
        """
        with self._lock:
            size = _estimate_size(value)

            # Check if single value exceeds limit
            if size > self._config.max_size_bytes:
                raise MemoryLimitError(size, self._config.max_size_bytes)

            # Remove existing entry if present
            if key in self._entries:
                self._remove_entry(key)

            # Evict until we have space
            self._make_room(size)

            # Check entry count limit
            if self._config.max_entries and len(self._entries) >= self._config.max_entries:
                self._evict_one()

            # Create entry
            entry = MemoryEntry(
                key=key,
                value=value,
                size_bytes=size,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl_seconds=ttl_seconds or self._config.ttl_seconds,
                agent_id=agent_id,
            )

            self._entries[key] = entry
            self._current_size += size

            return True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from memory.

        Args:
            key: The key to retrieve.
            default: Default value if key not found or expired.

        Returns:
            The stored value or default.
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return default

            # Check TTL expiration
            if self._is_expired(entry):
                self._remove_entry(key)
                return default

            # Update access time for LRU
            entry.accessed_at = datetime.now()

            # Move to end for LRU (most recently used)
            if self._config.eviction_policy == EvictionPolicy.LRU:
                self._entries.move_to_end(key)

            return entry.value

    def delete(self, key: str) -> bool:
        """Delete a key from memory.

        Args:
            key: The key to delete.

        Returns:
            True if key existed and was deleted.
        """
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: The key to check.

        Returns:
            True if key exists and is not expired.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                self._remove_entry(key)
                return False
            return True

    def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys, optionally filtered by pattern.

        Args:
            pattern: Optional glob pattern to filter keys.

        Returns:
            List of matching keys.
        """
        import fnmatch

        with self._lock:
            # Clean expired entries first
            self._clean_expired()

            keys = list(self._entries.keys())
            if pattern:
                keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
            return keys

    def get_by_agent(self, agent_id: str) -> dict[str, Any]:
        """Get all entries owned by an agent.

        Args:
            agent_id: The agent ID.

        Returns:
            Dict of key -> value for that agent.
        """
        with self._lock:
            self._clean_expired()
            return {
                key: entry.value
                for key, entry in self._entries.items()
                if entry.agent_id == agent_id
            }

    def clear(self, agent_id: str | None = None) -> int:
        """Clear all entries or entries for a specific agent.

        Args:
            agent_id: Optional agent to clear entries for.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            if agent_id is None:
                count = len(self._entries)
                self._entries.clear()
                self._current_size = 0
                return count

            keys_to_remove = [
                key for key, entry in self._entries.items() if entry.agent_id == agent_id
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
            return len(keys_to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dict with memory stats.
        """
        with self._lock:
            self._clean_expired()

            agent_usage: dict[str, int] = {}
            for entry in self._entries.values():
                agent = entry.agent_id or "_unassigned"
                agent_usage[agent] = agent_usage.get(agent, 0) + entry.size_bytes

            return {
                "total_entries": len(self._entries),
                "total_size_bytes": self._current_size,
                "max_size_bytes": self._config.max_size_bytes,
                "usage_percent": (self._current_size / self._config.max_size_bytes) * 100,
                "max_entries": self._config.max_entries,
                "eviction_policy": self._config.eviction_policy.value,
                "ttl_seconds": self._config.ttl_seconds,
                "agent_usage": agent_usage,
            }

    def _is_expired(self, entry: MemoryEntry) -> bool:
        """Check if an entry has expired."""
        if entry.ttl_seconds is None:
            return False
        expiry = entry.created_at + timedelta(seconds=entry.ttl_seconds)
        return datetime.now() > expiry

    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update size tracking."""
        entry = self._entries.pop(key, None)
        if entry:
            self._current_size -= entry.size_bytes
            if self._on_eviction:
                self._on_eviction(key, entry.value)

    def _make_room(self, needed_size: int) -> None:
        """Evict entries until we have enough room."""
        while self._current_size + needed_size > self._config.max_size_bytes:
            if not self._entries:
                break
            self._evict_one()

        # Warn if approaching limit
        if self._on_overflow:
            usage = (self._current_size + needed_size) / self._config.max_size_bytes
            if usage > 0.9:
                self._on_overflow(self._current_size + needed_size, self._config.max_size_bytes)

    def _evict_one(self) -> None:
        """Evict one entry based on policy."""
        if not self._entries:
            return

        if self._config.eviction_policy == EvictionPolicy.FIFO:
            # Remove first (oldest) entry
            key = next(iter(self._entries))
            self._remove_entry(key)

        elif self._config.eviction_policy == EvictionPolicy.LRU:
            # Remove first entry (least recently accessed due to move_to_end on access)
            key = next(iter(self._entries))
            self._remove_entry(key)

        elif self._config.eviction_policy == EvictionPolicy.TTL:
            # Remove oldest by creation time (which is also FIFO, but check expired first)
            expired_key = None
            oldest_key = None
            oldest_time = None

            for key, entry in self._entries.items():
                if self._is_expired(entry):
                    expired_key = key
                    break
                if oldest_time is None or entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    oldest_key = key

            key_to_remove = expired_key or oldest_key
            if key_to_remove:
                self._remove_entry(key_to_remove)

    def _clean_expired(self) -> None:
        """Remove all expired entries."""
        expired_keys = [key for key, entry in self._entries.items() if self._is_expired(entry)]
        for key in expired_keys:
            self._remove_entry(key)


class AgentMemory:
    """Per-agent memory interface with namespace isolation.

    This provides each agent with its own memory namespace while
    sharing the underlying MemoryStore.

    Example:
        store = MemoryStore(config=config)
        researcher_memory = AgentMemory(store, "researcher")
        executor_memory = AgentMemory(store, "executor")

        # Each agent has isolated namespace
        researcher_memory.set("results", [...])  # Stored as "researcher:results"
        executor_memory.get("results")  # Returns None (different namespace)
    """

    def __init__(
        self,
        store: MemoryStore,
        agent_id: str,
        namespace_separator: str = ":",
    ):
        """Initialize agent memory.

        Args:
            store: The underlying memory store.
            agent_id: The agent's ID.
            namespace_separator: Separator for namespace prefix.
        """
        self._store = store
        self._agent_id = agent_id
        self._sep = namespace_separator

    def _key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self._agent_id}{self._sep}{key}"

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> bool:
        """Store a value in agent's namespace."""
        return self._store.set(
            self._key(key),
            value,
            ttl_seconds=ttl_seconds,
            agent_id=self._agent_id,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from agent's namespace."""
        return self._store.get(self._key(key), default)

    def delete(self, key: str) -> bool:
        """Delete a key from agent's namespace."""
        return self._store.delete(self._key(key))

    def exists(self, key: str) -> bool:
        """Check if key exists in agent's namespace."""
        return self._store.exists(self._key(key))

    def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys in agent's namespace."""
        full_pattern = f"{self._agent_id}{self._sep}{pattern or '*'}"
        all_keys = self._store.keys(full_pattern)
        # Strip namespace prefix
        prefix = f"{self._agent_id}{self._sep}"
        return [k[len(prefix) :] for k in all_keys]

    def clear(self) -> int:
        """Clear all entries for this agent."""
        return self._store.clear(agent_id=self._agent_id)

    def get_all(self) -> dict[str, Any]:
        """Get all entries for this agent."""
        data = self._store.get_by_agent(self._agent_id)
        prefix = f"{self._agent_id}{self._sep}"
        return {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)}
