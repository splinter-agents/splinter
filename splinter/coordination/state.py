"""Shared state store for Splinter.

This module provides a single structured state object that is:
- Readable by all agents
- Written only through controlled updates
- Versioned per step

This is "what has happened so far". Agents must not infer history from chat.
"""

import copy
import hashlib
import json
import threading
from datetime import datetime
from typing import Any, Callable, Iterator

from ..exceptions import StateError
from ..types import StateSnapshot, StateVersion


def _deep_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a nested value using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _deep_set(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a nested value using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _deep_delete(data: dict[str, Any], path: str) -> bool:
    """Delete a nested value using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False
    if keys[-1] in current:
        del current[keys[-1]]
        return True
    return False


def _compute_hash(data: dict[str, Any]) -> str:
    """Compute a deterministic hash of state data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class SharedState:
    """Canonical shared state store for multi-agent workflows.

    This class provides:
    - A single source of truth for workflow state
    - Versioning of all state changes
    - Atomic updates with history tracking
    - Deterministic snapshots at each step

    Example:
        state = SharedState()

        # Write state (typically by agents)
        state.set("research.results", [...], agent_id="researcher")
        state.set("summary", "...", agent_id="summarizer")

        # Read state (any agent can read)
        results = state.get("research.results")

        # Get versioned snapshot
        snapshot = state.snapshot()
        print(f"Version: {snapshot.version.version}")
    """

    def __init__(
        self,
        initial_data: dict[str, Any] | None = None,
        on_change: Callable[[str, Any, Any, str | None], None] | None = None,
    ):
        """Initialize the shared state.

        Args:
            initial_data: Initial state data.
            on_change: Callback on state change (field, old_value, new_value, agent_id).
        """
        self._data: dict[str, Any] = initial_data or {}
        self._version = 0
        self._history: list[StateSnapshot] = []
        self._on_change = on_change
        self._lock = threading.RLock()

        # Record initial snapshot
        if initial_data:
            self._record_snapshot(None, list(initial_data.keys()))

    @property
    def version(self) -> int:
        """Current state version."""
        with self._lock:
            return self._version

    def get(self, path: str, default: Any = None) -> Any:
        """Get a value from state using dot notation.

        Args:
            path: Dot-separated path (e.g., "research.results").
            default: Default value if path not found.

        Returns:
            The value at path or default.
        """
        with self._lock:
            return copy.deepcopy(_deep_get(self._data, path, default))

    def set(
        self,
        path: str,
        value: Any,
        agent_id: str | None = None,
    ) -> int:
        """Set a value in state.

        Args:
            path: Dot-separated path (e.g., "research.results").
            value: Value to set.
            agent_id: ID of agent making the change.

        Returns:
            New version number.
        """
        with self._lock:
            old_value = _deep_get(self._data, path)
            _deep_set(self._data, path, copy.deepcopy(value))

            self._version += 1
            self._record_snapshot(agent_id, [path])

            if self._on_change:
                self._on_change(path, old_value, value, agent_id)

            return self._version

    def update(
        self,
        updates: dict[str, Any],
        agent_id: str | None = None,
    ) -> int:
        """Update multiple fields atomically.

        Args:
            updates: Dict of path -> value updates.
            agent_id: ID of agent making the changes.

        Returns:
            New version number.
        """
        with self._lock:
            changed_fields = []
            for path, value in updates.items():
                old_value = _deep_get(self._data, path)
                _deep_set(self._data, path, copy.deepcopy(value))
                changed_fields.append(path)

                if self._on_change:
                    self._on_change(path, old_value, value, agent_id)

            self._version += 1
            self._record_snapshot(agent_id, changed_fields)

            return self._version

    def delete(self, path: str, agent_id: str | None = None) -> bool:
        """Delete a field from state.

        Args:
            path: Dot-separated path to delete.
            agent_id: ID of agent making the change.

        Returns:
            True if field existed and was deleted.
        """
        with self._lock:
            old_value = _deep_get(self._data, path)
            if _deep_delete(self._data, path):
                self._version += 1
                self._record_snapshot(agent_id, [f"-{path}"])

                if self._on_change:
                    self._on_change(path, old_value, None, agent_id)

                return True
            return False

    def exists(self, path: str) -> bool:
        """Check if a path exists in state.

        Args:
            path: Dot-separated path to check.

        Returns:
            True if path exists.
        """
        with self._lock:
            return _deep_get(self._data, path) is not None

    def keys(self, prefix: str | None = None) -> list[str]:
        """Get all top-level keys or keys under a prefix.

        Args:
            prefix: Optional prefix to filter keys.

        Returns:
            List of keys.
        """
        with self._lock:
            if prefix:
                data = _deep_get(self._data, prefix, {})
                if isinstance(data, dict):
                    return [f"{prefix}.{k}" for k in data.keys()]
                return []
            return list(self._data.keys())

    def snapshot(self) -> StateSnapshot:
        """Get a snapshot of current state.

        Returns:
            Immutable snapshot with version info.
        """
        with self._lock:
            return StateSnapshot(
                data=copy.deepcopy(self._data),
                version=StateVersion(
                    version=self._version,
                    timestamp=datetime.now(),
                ),
            )

    def get_hash(self) -> str:
        """Get deterministic hash of current state.

        Returns:
            Hash string for state comparison.
        """
        with self._lock:
            return _compute_hash(self._data)

    def _record_snapshot(self, agent_id: str | None, changed_fields: list[str]) -> None:
        """Record a snapshot in history."""
        snapshot = StateSnapshot(
            data=copy.deepcopy(self._data),
            version=StateVersion(
                version=self._version,
                timestamp=datetime.now(),
                agent_id=agent_id,
                changed_fields=changed_fields,
            ),
        )
        self._history.append(snapshot)

    def get_history(self, limit: int | None = None) -> list[StateSnapshot]:
        """Get state history.

        Args:
            limit: Maximum number of snapshots to return.

        Returns:
            List of historical snapshots, newest first.
        """
        with self._lock:
            history = list(reversed(self._history))
            if limit:
                history = history[:limit]
            return history

    def get_version_at(self, version: int) -> StateSnapshot | None:
        """Get state at a specific version.

        Args:
            version: Version number to retrieve.

        Returns:
            Snapshot at that version or None.
        """
        with self._lock:
            for snapshot in self._history:
                if snapshot.version.version == version:
                    return snapshot
            return None

    def diff(self, from_version: int, to_version: int | None = None) -> dict[str, Any]:
        """Get changes between two versions.

        Args:
            from_version: Starting version.
            to_version: Ending version (default: current).

        Returns:
            Dict describing changes.
        """
        with self._lock:
            to_version = to_version or self._version

            from_snapshot = self.get_version_at(from_version)
            to_snapshot = self.get_version_at(to_version)

            if not from_snapshot or not to_snapshot:
                return {"error": "Version not found"}

            return self._compute_diff(from_snapshot.data, to_snapshot.data)

    def _compute_diff(
        self, old: dict[str, Any], new: dict[str, Any], prefix: str = ""
    ) -> dict[str, Any]:
        """Compute diff between two state dicts."""
        changes: dict[str, Any] = {"added": {}, "removed": {}, "modified": {}}

        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)

            if key not in old:
                changes["added"][path] = new_val
            elif key not in new:
                changes["removed"][path] = old_val
            elif old_val != new_val:
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    nested = self._compute_diff(old_val, new_val, path)
                    for change_type in ["added", "removed", "modified"]:
                        changes[change_type].update(nested[change_type])
                else:
                    changes["modified"][path] = {"old": old_val, "new": new_val}

        return changes

    def restore(self, version: int) -> bool:
        """Restore state to a previous version.

        Args:
            version: Version to restore to.

        Returns:
            True if restore succeeded.
        """
        with self._lock:
            snapshot = self.get_version_at(version)
            if not snapshot:
                return False

            self._data = copy.deepcopy(snapshot.data)
            self._version += 1
            self._record_snapshot(None, ["__restore__"])
            return True

    def clear(self) -> None:
        """Clear all state data."""
        with self._lock:
            self._data = {}
            self._version += 1
            self._record_snapshot(None, ["__clear__"])

    def to_dict(self) -> dict[str, Any]:
        """Export state as a plain dict.

        Returns:
            Deep copy of state data.
        """
        with self._lock:
            return copy.deepcopy(self._data)

    def merge(
        self,
        data: dict[str, Any],
        agent_id: str | None = None,
        deep: bool = True,
    ) -> int:
        """Merge data into state.

        Args:
            data: Data to merge.
            agent_id: ID of agent making the change.
            deep: If True, merge nested dicts; if False, overwrite.

        Returns:
            New version number.
        """
        with self._lock:
            if deep:
                self._deep_merge(self._data, data)
            else:
                self._data.update(data)

            self._version += 1
            self._record_snapshot(agent_id, list(data.keys()))
            return self._version

    def _deep_merge(self, base: dict[str, Any], updates: dict[str, Any]) -> None:
        """Deep merge updates into base dict."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
