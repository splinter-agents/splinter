"""State ownership and write boundaries for Splinter.

This module provides rules defining:
- Which agent can write which fields
- Others can only read

Shared state without ownership = corruption. This enforces responsibility.
"""

import fnmatch
import threading
from typing import Any, Callable

from ..exceptions import StateOwnershipError
from ..schemas import FieldOwnership


class StateOwnershipManager:
    """Manages state field ownership and write permissions.

    This class enforces which agents can write to which state fields,
    preventing corruption from unauthorized writes.

    Example:
        ownership = StateOwnershipManager()

        # Define ownership rules
        ownership.register("research.*", owner="researcher")
        ownership.register("summary", owner="summarizer")
        ownership.register("config.*", owner="system", read_only_for=["*"])

        # Check before writes
        ownership.check_write("researcher", "research.results")  # OK
        ownership.check_write("summarizer", "research.results")  # Raises

        # Wrap SharedState for automatic enforcement
        protected_state = ownership.protect(state)
    """

    def __init__(
        self,
        default_policy: str = "allow",  # "allow" or "deny"
        on_violation: Callable[[str, str, str | None], None] | None = None,
    ):
        """Initialize the ownership manager.

        Args:
            default_policy: Default when no ownership defined ("allow" or "deny").
            on_violation: Callback on ownership violation (agent_id, field, owner).
        """
        self._default_policy = default_policy
        self._on_violation = on_violation
        self._ownership_rules: list[FieldOwnership] = []
        self._lock = threading.RLock()

    def register(
        self,
        field_pattern: str,
        owner: str,
        read_only_for: list[str] | None = None,
    ) -> None:
        """Register ownership for a field pattern.

        Args:
            field_pattern: Field pattern (supports wildcards like "research.*").
            owner: Agent ID that owns this field.
            read_only_for: Agents that can only read (empty = all can read).
        """
        with self._lock:
            # Remove any existing rule for this pattern
            self._ownership_rules = [
                r for r in self._ownership_rules if r.field_pattern != field_pattern
            ]

            self._ownership_rules.append(
                FieldOwnership(
                    field_pattern=field_pattern,
                    owner_agent_id=owner,
                    read_only_for=read_only_for or [],
                )
            )

    def register_rules(self, rules: list[FieldOwnership]) -> None:
        """Register multiple ownership rules.

        Args:
            rules: List of field ownership rules.
        """
        with self._lock:
            for rule in rules:
                self.register(
                    field_pattern=rule.field_pattern,
                    owner=rule.owner_agent_id,
                    read_only_for=rule.read_only_for,
                )

    def set_agent_ownership(self, agent_id: str, fields: list[str]) -> None:
        """Set fields owned by an agent.

        Args:
            agent_id: The agent ID.
            fields: List of field patterns this agent owns.
        """
        with self._lock:
            for field in fields:
                self.register(field, owner=agent_id)

    def check_write(self, agent_id: str, field: str) -> bool:
        """Check if an agent can write to a field.

        Args:
            agent_id: Agent attempting to write.
            field: Field being written.

        Returns:
            True if write is allowed.

        Raises:
            StateOwnershipError: If write is not allowed.
        """
        with self._lock:
            rule = self._find_rule(field)

            if rule is None:
                # No rule defined - use default policy
                if self._default_policy == "deny":
                    self._record_violation(agent_id, field, None)
                    raise StateOwnershipError(agent_id, field, None)
                return True

            # Check if agent is the owner
            if not self._matches_pattern(agent_id, rule.owner_agent_id):
                self._record_violation(agent_id, field, rule.owner_agent_id)
                raise StateOwnershipError(agent_id, field, rule.owner_agent_id)

            return True

    def check_read(self, agent_id: str, field: str) -> bool:
        """Check if an agent can read a field.

        Note: By default all agents can read all fields. This only
        restricts when read_only_for is explicitly set.

        Args:
            agent_id: Agent attempting to read.
            field: Field being read.

        Returns:
            True (reads are generally allowed).
        """
        # Currently all reads are allowed
        # Could be extended to support read restrictions
        return True

    def _find_rule(self, field: str) -> FieldOwnership | None:
        """Find the most specific ownership rule for a field."""
        # Sort rules by specificity (more specific = longer pattern without wildcards)
        matching_rules = []
        for rule in self._ownership_rules:
            if self._field_matches_pattern(field, rule.field_pattern):
                # Calculate specificity score
                specificity = len(rule.field_pattern.replace("*", ""))
                matching_rules.append((specificity, rule))

        if not matching_rules:
            return None

        # Return most specific rule
        matching_rules.sort(key=lambda x: x[0], reverse=True)
        return matching_rules[0][1]

    def _field_matches_pattern(self, field: str, pattern: str) -> bool:
        """Check if a field matches a pattern."""
        return fnmatch.fnmatch(field, pattern)

    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if a value matches a pattern (for agent IDs)."""
        if pattern == "*":
            return True
        return fnmatch.fnmatch(value, pattern)

    def _record_violation(self, agent_id: str, field: str, owner: str | None) -> None:
        """Record an ownership violation."""
        if self._on_violation:
            self._on_violation(agent_id, field, owner)

    def get_owner(self, field: str) -> str | None:
        """Get the owner of a field.

        Args:
            field: The field to check.

        Returns:
            Owner agent ID or None.
        """
        with self._lock:
            rule = self._find_rule(field)
            return rule.owner_agent_id if rule else None

    def get_owned_fields(self, agent_id: str) -> list[str]:
        """Get field patterns owned by an agent.

        Args:
            agent_id: The agent ID.

        Returns:
            List of field patterns.
        """
        with self._lock:
            return [
                rule.field_pattern
                for rule in self._ownership_rules
                if self._matches_pattern(agent_id, rule.owner_agent_id)
            ]

    def get_all_rules(self) -> list[FieldOwnership]:
        """Get all ownership rules.

        Returns:
            List of all rules.
        """
        with self._lock:
            return list(self._ownership_rules)

    def protect(self, state: "SharedState") -> "ProtectedState":
        """Wrap a SharedState with ownership enforcement.

        Args:
            state: The SharedState to protect.

        Returns:
            ProtectedState wrapper.
        """
        return ProtectedState(state, self)

    def clear(self) -> None:
        """Clear all ownership rules."""
        with self._lock:
            self._ownership_rules.clear()


# Import here to avoid circular import
from .state import SharedState


class ProtectedState:
    """SharedState wrapper that enforces ownership on writes.

    This class wraps a SharedState and checks ownership rules
    before allowing any writes.

    Example:
        state = SharedState()
        ownership = StateOwnershipManager()
        ownership.register("research.*", owner="researcher")

        protected = ProtectedState(state, ownership)

        # Must specify agent_id for writes
        protected.set("research.results", data, agent_id="researcher")  # OK
        protected.set("research.results", data, agent_id="other")  # Raises
    """

    def __init__(self, state: SharedState, ownership: StateOwnershipManager):
        """Initialize protected state.

        Args:
            state: The underlying SharedState.
            ownership: Ownership manager for enforcement.
        """
        self._state = state
        self._ownership = ownership

    @property
    def version(self) -> int:
        """Current state version."""
        return self._state.version

    def get(self, path: str, default: Any = None) -> Any:
        """Get a value (no ownership check needed for reads)."""
        return self._state.get(path, default)

    def set(self, path: str, value: Any, agent_id: str) -> int:
        """Set a value with ownership check.

        Args:
            path: Dot-separated path.
            value: Value to set.
            agent_id: Agent making the write (required).

        Returns:
            New version number.

        Raises:
            StateOwnershipError: If agent doesn't own this field.
        """
        self._ownership.check_write(agent_id, path)
        return self._state.set(path, value, agent_id=agent_id)

    def update(self, updates: dict[str, Any], agent_id: str) -> int:
        """Update multiple fields with ownership check.

        Args:
            updates: Dict of path -> value.
            agent_id: Agent making the writes.

        Returns:
            New version number.

        Raises:
            StateOwnershipError: If agent doesn't own any field.
        """
        # Check all fields first
        for path in updates.keys():
            self._ownership.check_write(agent_id, path)

        return self._state.update(updates, agent_id=agent_id)

    def delete(self, path: str, agent_id: str) -> bool:
        """Delete a field with ownership check.

        Args:
            path: Path to delete.
            agent_id: Agent making the delete.

        Returns:
            True if deleted.

        Raises:
            StateOwnershipError: If agent doesn't own this field.
        """
        self._ownership.check_write(agent_id, path)
        return self._state.delete(path, agent_id=agent_id)

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return self._state.exists(path)

    def keys(self, prefix: str | None = None) -> list[str]:
        """Get keys."""
        return self._state.keys(prefix)

    def snapshot(self):
        """Get state snapshot."""
        return self._state.snapshot()

    def get_hash(self) -> str:
        """Get state hash."""
        return self._state.get_hash()

    def to_dict(self) -> dict[str, Any]:
        """Export as dict."""
        return self._state.to_dict()

    @property
    def underlying_state(self) -> SharedState:
        """Access the underlying SharedState (use with caution)."""
        return self._state
