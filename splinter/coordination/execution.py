"""Execution context and coordination for Splinter.

Provides:
- Chain Awareness: Agents know about other agents and execution history
- Goal Awareness: Agents understand the overall goal and their role
- Action Eligibility: Rules for which agent can act when
- Completion Signals: Explicit step completion declarations
- Waiting Reasons: Track why agents are idle
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from ..exceptions import SplinterError


class AgentNotEligibleError(SplinterError):
    """Raised when agent is not eligible to act."""

    def __init__(self, agent_id: str, reason: str):
        self.agent_id = agent_id
        self.reason = reason
        super().__init__(f"Agent '{agent_id}' is not eligible to act: {reason}")


class CompletionNotDeclaredError(SplinterError):
    """Raised when step completes without explicit declaration."""

    def __init__(self, agent_id: str, step: int):
        self.agent_id = agent_id
        self.step = step
        super().__init__(
            f"Agent '{agent_id}' did not declare completion for step {step}"
        )


class WaitReason(str, Enum):
    """Reasons why an agent is waiting."""

    DEPENDENCY = "dependency"  # Waiting for another agent
    RESOURCE = "resource"  # Waiting for resource
    APPROVAL = "approval"  # Waiting for human approval
    RATE_LIMIT = "rate_limit"  # Waiting due to rate limit
    CIRCUIT_OPEN = "circuit_open"  # Waiting for circuit breaker
    SCHEDULED = "scheduled"  # Waiting for scheduled time
    CONDITION = "condition"  # Waiting for condition to be met
    UNKNOWN = "unknown"


@dataclass
class WaitState:
    """State of a waiting agent."""

    agent_id: str
    reason: WaitReason
    waiting_for: str | None = None  # What specifically they're waiting for
    since: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionSignal:
    """Explicit completion signal from an agent."""

    agent_id: str
    step: int
    success: bool
    output_summary: str | None = None
    next_agent: str | None = None  # Suggested next agent
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionRecord:
    """Record of an agent's execution."""

    agent_id: str
    step: int
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    input_summary: str | None = None
    output_summary: str | None = None
    error: str | None = None


@dataclass
class Goal:
    """A workflow goal."""

    goal_id: str
    description: str
    success_criteria: list[str] = field(default_factory=list)
    assigned_agents: list[str] = field(default_factory=list)
    achieved: bool = False
    progress: float = 0.0  # 0.0 to 1.0


class ChainContext:
    """Provides chain awareness to agents.

    Agents can query:
    - Who else is in the chain
    - What has happened so far
    - What the overall goal is

    Example:
        context = ChainContext(workflow_id="wf-123")

        # Register agents
        context.register_agent("researcher", role="Find information")
        context.register_agent("writer", role="Write content")

        # Record execution
        context.start_execution("researcher", step=0, input_summary="Research AI trends")
        # ... agent executes ...
        context.complete_execution("researcher", step=0, output_summary="Found 5 trends")

        # Query from another agent
        history = context.get_execution_history()
        other_agents = context.get_other_agents("writer")
    """

    def __init__(
        self,
        workflow_id: str,
        on_execution_complete: Callable[[ExecutionRecord], None] | None = None,
    ):
        """Initialize chain context.

        Args:
            workflow_id: Workflow identifier.
            on_execution_complete: Callback when agent completes.
        """
        self._workflow_id = workflow_id
        self._on_execution_complete = on_execution_complete

        self._agents: dict[str, dict[str, Any]] = {}  # agent_id -> info
        self._history: list[ExecutionRecord] = []
        self._current_executions: dict[str, ExecutionRecord] = {}
        self._lock = threading.RLock()

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    def register_agent(
        self,
        agent_id: str,
        role: str | None = None,
        description: str | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        """Register an agent in the chain."""
        with self._lock:
            self._agents[agent_id] = {
                "agent_id": agent_id,
                "role": role,
                "description": description,
                "capabilities": capabilities or [],
                "registered_at": datetime.now(),
            }

    def get_agent_info(self, agent_id: str) -> dict[str, Any] | None:
        """Get info about an agent."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_all_agents(self) -> list[dict[str, Any]]:
        """Get info about all agents."""
        with self._lock:
            return list(self._agents.values())

    def get_other_agents(self, current_agent_id: str) -> list[dict[str, Any]]:
        """Get info about other agents (not the current one)."""
        with self._lock:
            return [
                info for agent_id, info in self._agents.items()
                if agent_id != current_agent_id
            ]

    def start_execution(
        self,
        agent_id: str,
        step: int,
        input_summary: str | None = None,
    ) -> ExecutionRecord:
        """Record start of agent execution."""
        with self._lock:
            record = ExecutionRecord(
                agent_id=agent_id,
                step=step,
                started_at=datetime.now(),
                input_summary=input_summary,
            )
            self._current_executions[agent_id] = record
            return record

    def complete_execution(
        self,
        agent_id: str,
        step: int,
        success: bool = True,
        output_summary: str | None = None,
        error: str | None = None,
    ) -> ExecutionRecord:
        """Record completion of agent execution."""
        with self._lock:
            record = self._current_executions.pop(agent_id, None)
            if record is None:
                record = ExecutionRecord(
                    agent_id=agent_id,
                    step=step,
                    started_at=datetime.now(),
                )

            record.completed_at = datetime.now()
            record.success = success
            record.output_summary = output_summary
            record.error = error

            self._history.append(record)

            if self._on_execution_complete:
                self._on_execution_complete(record)

            return record

    def get_execution_history(
        self,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[ExecutionRecord]:
        """Get execution history."""
        with self._lock:
            history = list(self._history)
            if agent_id:
                history = [r for r in history if r.agent_id == agent_id]
            if limit:
                history = history[-limit:]
            return history

    def get_last_execution(self, agent_id: str | None = None) -> ExecutionRecord | None:
        """Get the last execution record."""
        history = self.get_execution_history(agent_id=agent_id, limit=1)
        return history[0] if history else None

    def is_agent_executing(self, agent_id: str) -> bool:
        """Check if an agent is currently executing."""
        with self._lock:
            return agent_id in self._current_executions

    def get_context_for_agent(self, agent_id: str) -> dict[str, Any]:
        """Get full context for an agent to understand the chain."""
        with self._lock:
            return {
                "workflow_id": self._workflow_id,
                "current_agent": agent_id,
                "all_agents": self.get_all_agents(),
                "other_agents": self.get_other_agents(agent_id),
                "execution_history": [
                    {
                        "agent_id": r.agent_id,
                        "step": r.step,
                        "success": r.success,
                        "output_summary": r.output_summary,
                    }
                    for r in self._history[-10:]  # Last 10
                ],
                "currently_executing": list(self._current_executions.keys()),
            }


class GoalTracker:
    """Tracks workflow goals and agent progress.

    Example:
        tracker = GoalTracker()

        # Set the main goal
        tracker.set_goal(Goal(
            goal_id="main",
            description="Write a blog post about AI",
            success_criteria=[
                "Research completed",
                "Outline created",
                "Draft written",
                "Review completed",
            ],
            assigned_agents=["researcher", "writer", "reviewer"],
        ))

        # Update progress
        tracker.update_progress("main", 0.25)  # 25% complete
        tracker.mark_criterion_met("main", "Research completed")

        # Check if goal is achieved
        if tracker.is_achieved("main"):
            print("Goal completed!")
    """

    def __init__(
        self,
        on_goal_achieved: Callable[[Goal], None] | None = None,
    ):
        """Initialize goal tracker.

        Args:
            on_goal_achieved: Callback when goal is achieved.
        """
        self._on_goal_achieved = on_goal_achieved
        self._goals: dict[str, Goal] = {}
        self._criteria_met: dict[str, set[str]] = {}  # goal_id -> set of met criteria
        self._lock = threading.RLock()

    def set_goal(self, goal: Goal) -> None:
        """Set a goal."""
        with self._lock:
            self._goals[goal.goal_id] = goal
            self._criteria_met[goal.goal_id] = set()

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        with self._lock:
            return self._goals.get(goal_id)

    def get_all_goals(self) -> list[Goal]:
        """Get all goals."""
        with self._lock:
            return list(self._goals.values())

    def update_progress(self, goal_id: str, progress: float) -> None:
        """Update goal progress (0.0 to 1.0)."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal:
                goal.progress = min(1.0, max(0.0, progress))
                if goal.progress >= 1.0:
                    self._check_achievement(goal_id)

    def mark_criterion_met(self, goal_id: str, criterion: str) -> None:
        """Mark a success criterion as met."""
        with self._lock:
            if goal_id not in self._criteria_met:
                self._criteria_met[goal_id] = set()
            self._criteria_met[goal_id].add(criterion)
            self._check_achievement(goal_id)

    def _check_achievement(self, goal_id: str) -> None:
        """Check if goal is achieved and trigger callback."""
        goal = self._goals.get(goal_id)
        if not goal or goal.achieved:
            return

        met = self._criteria_met.get(goal_id, set())
        if all(c in met for c in goal.success_criteria):
            goal.achieved = True
            goal.progress = 1.0
            if self._on_goal_achieved:
                self._on_goal_achieved(goal)

    def is_achieved(self, goal_id: str) -> bool:
        """Check if goal is achieved."""
        with self._lock:
            goal = self._goals.get(goal_id)
            return goal.achieved if goal else False

    def get_progress(self, goal_id: str) -> float:
        """Get goal progress."""
        with self._lock:
            goal = self._goals.get(goal_id)
            return goal.progress if goal else 0.0

    def get_unmet_criteria(self, goal_id: str) -> list[str]:
        """Get list of unmet success criteria."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return []
            met = self._criteria_met.get(goal_id, set())
            return [c for c in goal.success_criteria if c not in met]


class ActionEligibility:
    """Manages which agents are allowed to act at any moment.

    Example:
        eligibility = ActionEligibility()

        # Set eligibility rules
        eligibility.set_eligible("researcher")  # Only researcher can act

        # Check before acting
        eligibility.check("researcher")  # OK
        eligibility.check("writer")  # Raises AgentNotEligibleError

        # Transfer eligibility
        eligibility.transfer("researcher", "writer")
    """

    def __init__(
        self,
        on_eligibility_change: Callable[[str | None, str | None], None] | None = None,
    ):
        """Initialize eligibility tracker.

        Args:
            on_eligibility_change: Callback on change (old_agent, new_agent).
        """
        self._on_change = on_eligibility_change
        self._eligible_agents: set[str] = set()
        self._blocked_agents: set[str] = set()
        self._lock = threading.RLock()

    def set_eligible(self, *agent_ids: str) -> None:
        """Set which agents are eligible to act."""
        with self._lock:
            old = self._eligible_agents.copy()
            self._eligible_agents = set(agent_ids)
            if self._on_change and old != self._eligible_agents:
                self._on_change(
                    next(iter(old), None),
                    next(iter(self._eligible_agents), None),
                )

    def add_eligible(self, agent_id: str) -> None:
        """Add an agent to eligible set."""
        with self._lock:
            self._eligible_agents.add(agent_id)
            self._blocked_agents.discard(agent_id)

    def remove_eligible(self, agent_id: str) -> None:
        """Remove an agent from eligible set."""
        with self._lock:
            self._eligible_agents.discard(agent_id)

    def block(self, agent_id: str) -> None:
        """Explicitly block an agent."""
        with self._lock:
            self._blocked_agents.add(agent_id)
            self._eligible_agents.discard(agent_id)

    def unblock(self, agent_id: str) -> None:
        """Remove explicit block on agent."""
        with self._lock:
            self._blocked_agents.discard(agent_id)

    def transfer(self, from_agent: str, to_agent: str) -> None:
        """Transfer eligibility from one agent to another."""
        with self._lock:
            self._eligible_agents.discard(from_agent)
            self._eligible_agents.add(to_agent)
            if self._on_change:
                self._on_change(from_agent, to_agent)

    def check(self, agent_id: str) -> bool:
        """Check if agent is eligible to act.

        Raises:
            AgentNotEligibleError: If agent is not eligible.
        """
        with self._lock:
            # Check explicit blocks
            if agent_id in self._blocked_agents:
                raise AgentNotEligibleError(agent_id, "Agent is blocked")

            # If no agents are set as eligible, all are eligible
            if not self._eligible_agents:
                return True

            if agent_id not in self._eligible_agents:
                eligible_list = ", ".join(self._eligible_agents)
                raise AgentNotEligibleError(
                    agent_id,
                    f"Only these agents can act: {eligible_list}",
                )

            return True

    def is_eligible(self, agent_id: str) -> bool:
        """Check eligibility without throwing."""
        try:
            return self.check(agent_id)
        except AgentNotEligibleError:
            return False

    def get_eligible_agents(self) -> list[str]:
        """Get list of eligible agents."""
        with self._lock:
            return list(self._eligible_agents)

    def clear(self) -> None:
        """Clear all eligibility rules (all agents become eligible)."""
        with self._lock:
            self._eligible_agents.clear()
            self._blocked_agents.clear()


class CompletionTracker:
    """Tracks explicit completion signals from agents.

    Requires agents to explicitly declare when they are done.

    Example:
        tracker = CompletionTracker(require_explicit=True)

        # Agent declares completion
        tracker.declare_complete(
            agent_id="researcher",
            step=0,
            success=True,
            output_summary="Found 5 relevant articles",
        )

        # Check completion
        if tracker.is_complete("researcher", 0):
            print("Step 0 complete!")
    """

    def __init__(
        self,
        require_explicit: bool = True,
        on_completion: Callable[[CompletionSignal], None] | None = None,
    ):
        """Initialize completion tracker.

        Args:
            require_explicit: If True, steps must explicitly declare completion.
            on_completion: Callback when completion is declared.
        """
        self._require_explicit = require_explicit
        self._on_completion = on_completion
        self._completions: dict[tuple[str, int], CompletionSignal] = {}
        self._lock = threading.RLock()

    def declare_complete(
        self,
        agent_id: str,
        step: int,
        success: bool = True,
        output_summary: str | None = None,
        next_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionSignal:
        """Declare that an agent has completed a step."""
        with self._lock:
            signal = CompletionSignal(
                agent_id=agent_id,
                step=step,
                success=success,
                output_summary=output_summary,
                next_agent=next_agent,
                metadata=metadata or {},
            )
            self._completions[(agent_id, step)] = signal

            if self._on_completion:
                self._on_completion(signal)

            return signal

    def is_complete(self, agent_id: str, step: int) -> bool:
        """Check if agent has declared completion for step."""
        with self._lock:
            return (agent_id, step) in self._completions

    def get_completion(self, agent_id: str, step: int) -> CompletionSignal | None:
        """Get completion signal for a step."""
        with self._lock:
            return self._completions.get((agent_id, step))

    def require_completion(self, agent_id: str, step: int) -> None:
        """Require that completion was declared.

        Raises:
            CompletionNotDeclaredError: If completion not declared.
        """
        if self._require_explicit and not self.is_complete(agent_id, step):
            raise CompletionNotDeclaredError(agent_id, step)

    def get_all_completions(self) -> list[CompletionSignal]:
        """Get all completion signals."""
        with self._lock:
            return list(self._completions.values())


class WaitTracker:
    """Tracks waiting states and reasons for agents.

    Example:
        tracker = WaitTracker()

        # Agent starts waiting
        tracker.start_waiting(
            agent_id="writer",
            reason=WaitReason.DEPENDENCY,
            waiting_for="researcher",
        )

        # Check waiting status
        if tracker.is_waiting("writer"):
            state = tracker.get_wait_state("writer")
            print(f"Writer is waiting for: {state.waiting_for}")

        # Agent stops waiting
        tracker.stop_waiting("writer")
    """

    def __init__(
        self,
        on_wait_start: Callable[[WaitState], None] | None = None,
        on_wait_end: Callable[[str, WaitState], None] | None = None,
    ):
        """Initialize wait tracker.

        Args:
            on_wait_start: Callback when agent starts waiting.
            on_wait_end: Callback when agent stops waiting.
        """
        self._on_wait_start = on_wait_start
        self._on_wait_end = on_wait_end
        self._waiting: dict[str, WaitState] = {}
        self._lock = threading.RLock()

    def start_waiting(
        self,
        agent_id: str,
        reason: WaitReason,
        waiting_for: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WaitState:
        """Record that an agent is waiting."""
        with self._lock:
            state = WaitState(
                agent_id=agent_id,
                reason=reason,
                waiting_for=waiting_for,
                metadata=metadata or {},
            )
            self._waiting[agent_id] = state

            if self._on_wait_start:
                self._on_wait_start(state)

            return state

    def stop_waiting(self, agent_id: str) -> WaitState | None:
        """Record that an agent stopped waiting."""
        with self._lock:
            state = self._waiting.pop(agent_id, None)
            if state and self._on_wait_end:
                self._on_wait_end(agent_id, state)
            return state

    def is_waiting(self, agent_id: str) -> bool:
        """Check if agent is waiting."""
        with self._lock:
            return agent_id in self._waiting

    def get_wait_state(self, agent_id: str) -> WaitState | None:
        """Get wait state for an agent."""
        with self._lock:
            return self._waiting.get(agent_id)

    def get_all_waiting(self) -> list[WaitState]:
        """Get all waiting agents."""
        with self._lock:
            return list(self._waiting.values())

    def get_waiting_for(self, target: str) -> list[WaitState]:
        """Get agents waiting for a specific target."""
        with self._lock:
            return [
                s for s in self._waiting.values()
                if s.waiting_for == target
            ]

    def get_by_reason(self, reason: WaitReason) -> list[WaitState]:
        """Get agents waiting for a specific reason."""
        with self._lock:
            return [
                s for s in self._waiting.values()
                if s.reason == reason
            ]
