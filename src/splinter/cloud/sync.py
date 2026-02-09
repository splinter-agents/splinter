"""
StateSync - Sync local state to Splinter Cloud.

Enables paid features:
- Live agent dashboard
- Coordination status view
- Bottleneck detection
- Deadlock detection
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import deque


class SyncEventType(Enum):
    """Types of events synced to cloud."""
    # Agent events
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_PAUSED = "agent_paused"
    AGENT_RESUMED = "agent_resumed"

    # Control events
    BUDGET_UPDATE = "budget_update"
    RATE_LIMIT_HIT = "rate_limit_hit"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSED = "circuit_closed"
    LOOP_DETECTED = "loop_detected"
    RULE_TRIGGERED = "rule_triggered"
    DECISION_LOCKED = "decision_locked"

    # Coordination events
    STATE_UPDATE = "state_update"
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    HANDOFF_STARTED = "handoff_started"
    HANDOFF_COMPLETED = "handoff_completed"
    GOAL_PROGRESS = "goal_progress"
    GOAL_COMPLETED = "goal_completed"
    AGENT_WAITING = "agent_waiting"
    AGENT_ELIGIBLE = "agent_eligible"
    DEADLOCK_DETECTED = "deadlock_detected"
    BOTTLENECK_DETECTED = "bottleneck_detected"


@dataclass
class SyncEvent:
    """Event to sync to cloud."""
    type: SyncEventType
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "payload": self.payload,
        }


@dataclass
class AgentSnapshot:
    """Snapshot of agent state for dashboard."""
    agent_id: str
    status: str  # running, paused, completed, failed, waiting
    provider: str
    model: str
    started_at: float
    steps_completed: int = 0
    budget_used: float = 0.0
    current_action: Optional[str] = None
    waiting_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CoordinationSnapshot:
    """Snapshot of coordination state for dashboard."""
    shared_state: dict
    ownership: dict[str, str]  # field -> owner
    checkpoints: list[str]
    active_handoffs: list[dict]
    goals: list[dict]
    waiting_agents: dict[str, str]  # agent_id -> reason
    eligible_agents: list[str]


class StateSync:
    """
    Syncs local state to Splinter Cloud.

    Batches events and sends them periodically to reduce overhead.
    Also takes periodic snapshots for the live dashboard.
    """

    def __init__(self, config: "CloudConfig"):
        self.config = config
        self._event_queue: deque[SyncEvent] = deque(maxlen=10000)
        self._registered_objects: dict[str, Any] = {}
        self._running = False
        self._last_snapshot: Optional[dict] = None

    def register(self, name: str, obj: Any) -> None:
        """Register object for state sync."""
        self._registered_objects[name] = obj

    def queue_event(self, event: SyncEvent) -> None:
        """Queue event for sync."""
        self._event_queue.append(event)

    async def start(self) -> None:
        """Start sync loop."""
        self._running = True
        while self._running:
            try:
                await self._sync_cycle()
                await asyncio.sleep(self.config.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but keep running
                await asyncio.sleep(self.config.sync_interval)

    async def stop(self) -> None:
        """Stop sync loop."""
        self._running = False

    async def _sync_cycle(self) -> None:
        """Run one sync cycle."""
        # Collect pending events
        events = []
        while self._event_queue:
            try:
                events.append(self._event_queue.popleft())
            except IndexError:
                break

        # Take state snapshot
        snapshot = self._take_snapshot()

        # Send to cloud
        if events or snapshot != self._last_snapshot:
            await self._send_to_cloud(events, snapshot)
            self._last_snapshot = snapshot

    def _take_snapshot(self) -> dict:
        """Take snapshot of current state."""
        snapshot = {
            "agents": [],
            "coordination": None,
            "control": {},
        }

        # Agent snapshots
        workflow = self._registered_objects.get("workflow")
        if workflow and hasattr(workflow, "get_agent_statuses"):
            snapshot["agents"] = workflow.get_agent_statuses()

        # Coordination snapshot
        shared_state = self._registered_objects.get("shared_state")
        if shared_state:
            snapshot["coordination"] = {
                "state": dict(shared_state._state) if hasattr(shared_state, "_state") else {},
                "ownership": dict(shared_state._ownership) if hasattr(shared_state, "_ownership") else {},
            }

        # Add checkpoint info
        checkpoint_manager = self._registered_objects.get("checkpoint_manager")
        if checkpoint_manager and hasattr(checkpoint_manager, "list_checkpoints"):
            snapshot["coordination"] = snapshot.get("coordination") or {}
            snapshot["coordination"]["checkpoints"] = checkpoint_manager.list_checkpoints()

        # Add wait tracking
        wait_tracker = self._registered_objects.get("wait_tracker")
        if wait_tracker and hasattr(wait_tracker, "get_all_waiting"):
            snapshot["coordination"] = snapshot.get("coordination") or {}
            snapshot["coordination"]["waiting"] = wait_tracker.get_all_waiting()

        # Add goal tracking
        goal_tracker = self._registered_objects.get("goal_tracker")
        if goal_tracker and hasattr(goal_tracker, "get_all_goals"):
            snapshot["coordination"] = snapshot.get("coordination") or {}
            snapshot["coordination"]["goals"] = goal_tracker.get_all_goals()

        # Control state
        execution_limits = self._registered_objects.get("execution_limits")
        if execution_limits:
            snapshot["control"]["limits"] = {
                "max_budget": getattr(execution_limits, "max_budget", None),
                "max_steps": getattr(execution_limits, "max_steps", None),
                "budget_used": getattr(execution_limits, "budget_used", 0),
                "steps_used": getattr(execution_limits, "steps_used", 0),
            }

        circuit_registry = self._registered_objects.get("circuit_breaker_registry")
        if circuit_registry and hasattr(circuit_registry, "get_all_states"):
            snapshot["control"]["circuits"] = circuit_registry.get_all_states()

        return snapshot

    async def _send_to_cloud(self, events: list[SyncEvent], snapshot: dict) -> None:
        """Send events and snapshot to cloud API."""
        # In production, this would make an HTTP POST request
        # For now, this is a stub that could be connected to real API
        payload = {
            "api_key": self.config.api_key,
            "project_id": self.config.project_id,
            "events": [e.to_dict() for e in events],
            "snapshot": snapshot,
            "timestamp": time.time(),
        }

        # Would send to: POST {self.config.endpoint}/v1/sync
        # await httpx.post(f"{self.config.endpoint}/v1/sync", json=payload)
        _ = payload  # Stub for now

    # Convenience methods for common events

    def agent_started(self, agent_id: str, provider: str, model: str) -> None:
        """Record agent start."""
        self.queue_event(SyncEvent(
            type=SyncEventType.AGENT_STARTED,
            agent_id=agent_id,
            payload={"provider": provider, "model": model},
        ))

    def agent_completed(self, agent_id: str, result: Any) -> None:
        """Record agent completion."""
        self.queue_event(SyncEvent(
            type=SyncEventType.AGENT_COMPLETED,
            agent_id=agent_id,
            payload={"result": str(result)[:1000]},  # Truncate large results
        ))

    def agent_failed(self, agent_id: str, error: str) -> None:
        """Record agent failure."""
        self.queue_event(SyncEvent(
            type=SyncEventType.AGENT_FAILED,
            agent_id=agent_id,
            payload={"error": error},
        ))

    def loop_detected(self, agent_id: str, pattern: str) -> None:
        """Record loop detection."""
        self.queue_event(SyncEvent(
            type=SyncEventType.LOOP_DETECTED,
            agent_id=agent_id,
            payload={"pattern": pattern},
        ))

    def deadlock_detected(self, agents: list[str], reason: str) -> None:
        """Record deadlock detection."""
        self.queue_event(SyncEvent(
            type=SyncEventType.DEADLOCK_DETECTED,
            payload={"agents": agents, "reason": reason},
        ))

    def bottleneck_detected(self, agent_id: str, wait_time: float, reason: str) -> None:
        """Record bottleneck detection."""
        self.queue_event(SyncEvent(
            type=SyncEventType.BOTTLENECK_DETECTED,
            agent_id=agent_id,
            payload={"wait_time": wait_time, "reason": reason},
        ))
