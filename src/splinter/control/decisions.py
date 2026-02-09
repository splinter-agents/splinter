"""Decision enforcement for Splinter.

Once an agent makes an important decision, it cannot change its mind.
This prevents flip-flopping and ensures consistent execution.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from ..exceptions import SplinterError


class DecisionLockError(SplinterError):
    """Raised when trying to change a locked decision."""

    def __init__(self, decision_id: str, agent_id: str, locked_by: str, locked_at: datetime):
        self.decision_id = decision_id
        self.agent_id = agent_id
        self.locked_by = locked_by
        self.locked_at = locked_at
        super().__init__(
            f"Decision '{decision_id}' is locked by agent '{locked_by}' "
            f"(at {locked_at}). Agent '{agent_id}' cannot change it."
        )


class DecisionType(str, Enum):
    """Types of decisions that can be enforced."""

    STRATEGY = "strategy"  # High-level approach decision
    CHOICE = "choice"  # Selection from options
    COMMIT = "commit"  # Commitment to action
    APPROVAL = "approval"  # Approval/rejection decision
    DELEGATION = "delegation"  # Task delegation decision


@dataclass
class Decision:
    """A recorded decision."""

    decision_id: str
    decision_type: DecisionType
    agent_id: str
    value: Any
    reasoning: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class DecisionEnforcer:
    """Enforces decisions made by agents.

    Example:
        enforcer = DecisionEnforcer()

        # Agent makes a decision
        enforcer.record_decision(
            decision_id="approach",
            agent_id="planner",
            decision_type=DecisionType.STRATEGY,
            value="parallel_processing",
            reasoning="Faster for this workload",
            lock=True,  # Lock the decision
        )

        # Later, trying to change it fails
        try:
            enforcer.update_decision("approach", "researcher", "sequential")
        except DecisionLockError:
            print("Cannot change locked decision!")

        # Check if decision exists
        if enforcer.has_decision("approach"):
            value = enforcer.get_decision("approach")
    """

    def __init__(
        self,
        auto_lock: bool = True,
        on_decision: Callable[[Decision], None] | None = None,
        on_lock_violation: Callable[[str, str, str], None] | None = None,
    ):
        """Initialize decision enforcer.

        Args:
            auto_lock: If True, decisions are locked by default.
            on_decision: Callback when decision is made.
            on_lock_violation: Callback on violation (decision_id, agent_id, locked_by).
        """
        self._auto_lock = auto_lock
        self._on_decision = on_decision
        self._on_lock_violation = on_lock_violation
        self._decisions: dict[str, Decision] = {}
        self._lock = threading.RLock()

    def record_decision(
        self,
        decision_id: str,
        agent_id: str,
        decision_type: DecisionType,
        value: Any,
        reasoning: str | None = None,
        lock: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Decision:
        """Record a decision.

        Args:
            decision_id: Unique identifier for the decision.
            agent_id: Agent making the decision.
            decision_type: Type of decision.
            value: The decision value.
            reasoning: Optional reasoning.
            lock: Whether to lock (default: auto_lock setting).
            metadata: Additional metadata.

        Returns:
            The recorded decision.

        Raises:
            DecisionLockError: If decision exists and is locked.
        """
        with self._lock:
            # Check if decision already exists and is locked
            if decision_id in self._decisions:
                existing = self._decisions[decision_id]
                if existing.locked:
                    if self._on_lock_violation:
                        self._on_lock_violation(
                            decision_id, agent_id, existing.agent_id
                        )
                    raise DecisionLockError(
                        decision_id, agent_id, existing.agent_id, existing.timestamp
                    )

            should_lock = lock if lock is not None else self._auto_lock

            decision = Decision(
                decision_id=decision_id,
                decision_type=decision_type,
                agent_id=agent_id,
                value=value,
                reasoning=reasoning,
                locked=should_lock,
                metadata=metadata or {},
            )

            self._decisions[decision_id] = decision

            if self._on_decision:
                self._on_decision(decision)

            return decision

    def get_decision(self, decision_id: str) -> Any:
        """Get a decision value.

        Args:
            decision_id: Decision identifier.

        Returns:
            The decision value, or None if not found.
        """
        with self._lock:
            decision = self._decisions.get(decision_id)
            return decision.value if decision else None

    def get_decision_full(self, decision_id: str) -> Decision | None:
        """Get full decision object."""
        with self._lock:
            return self._decisions.get(decision_id)

    def has_decision(self, decision_id: str) -> bool:
        """Check if a decision exists."""
        with self._lock:
            return decision_id in self._decisions

    def is_locked(self, decision_id: str) -> bool:
        """Check if a decision is locked."""
        with self._lock:
            decision = self._decisions.get(decision_id)
            return decision.locked if decision else False

    def update_decision(
        self,
        decision_id: str,
        agent_id: str,
        new_value: Any,
        reasoning: str | None = None,
    ) -> Decision:
        """Update an existing decision.

        Args:
            decision_id: Decision to update.
            agent_id: Agent making the update.
            new_value: New decision value.
            reasoning: Reasoning for change.

        Returns:
            Updated decision.

        Raises:
            DecisionLockError: If decision is locked.
        """
        with self._lock:
            if decision_id not in self._decisions:
                raise ValueError(f"Decision '{decision_id}' not found")

            existing = self._decisions[decision_id]
            if existing.locked:
                if self._on_lock_violation:
                    self._on_lock_violation(decision_id, agent_id, existing.agent_id)
                raise DecisionLockError(
                    decision_id, agent_id, existing.agent_id, existing.timestamp
                )

            # Create new decision with updated value
            return self.record_decision(
                decision_id=decision_id,
                agent_id=agent_id,
                decision_type=existing.decision_type,
                value=new_value,
                reasoning=reasoning,
                lock=existing.locked,
                metadata=existing.metadata,
            )

    def lock_decision(self, decision_id: str) -> bool:
        """Lock a decision to prevent changes.

        Returns:
            True if locked, False if not found.
        """
        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision:
                decision.locked = True
                return True
            return False

    def unlock_decision(self, decision_id: str, force: bool = False) -> bool:
        """Unlock a decision.

        Args:
            decision_id: Decision to unlock.
            force: If True, unlock even if locked.

        Returns:
            True if unlocked.
        """
        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision:
                if force or not decision.locked:
                    decision.locked = False
                    return True
            return False

    def get_decisions_by_agent(self, agent_id: str) -> list[Decision]:
        """Get all decisions made by an agent."""
        with self._lock:
            return [
                d for d in self._decisions.values()
                if d.agent_id == agent_id
            ]

    def get_decisions_by_type(self, decision_type: DecisionType) -> list[Decision]:
        """Get all decisions of a type."""
        with self._lock:
            return [
                d for d in self._decisions.values()
                if d.decision_type == decision_type
            ]

    def get_all_decisions(self) -> list[Decision]:
        """Get all decisions."""
        with self._lock:
            return list(self._decisions.values())

    def clear(self, decision_id: str | None = None, force: bool = False) -> int:
        """Clear decisions.

        Args:
            decision_id: Specific decision to clear, or None for all.
            force: If True, clear even locked decisions.

        Returns:
            Number of decisions cleared.
        """
        with self._lock:
            if decision_id:
                decision = self._decisions.get(decision_id)
                if decision and (force or not decision.locked):
                    del self._decisions[decision_id]
                    return 1
                return 0

            if force:
                count = len(self._decisions)
                self._decisions.clear()
                return count

            # Only clear unlocked decisions
            unlocked = [k for k, v in self._decisions.items() if not v.locked]
            for k in unlocked:
                del self._decisions[k]
            return len(unlocked)
