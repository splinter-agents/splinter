"""Resumable execution and checkpointing for Splinter.

This module provides checkpointing of:
- Current step
- Shared state snapshot
- Execution position

Production workflows will fail. Restarting from scratch is unacceptable.
This enables resume from last completed step with no duplicate tool calls.
"""

import json
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from ..exceptions import CheckpointError, CheckpointNotFoundError
from ..schemas import AgentStatus, Checkpoint, ExecutionMetrics, StateSnapshot, StateVersion


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends."""

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        pass

    @abstractmethod
    def load(self, workflow_id: str, step: int | None = None) -> Checkpoint | None:
        """Load a checkpoint. If step is None, load the latest."""
        pass

    @abstractmethod
    def list_checkpoints(self, workflow_id: str) -> list[Checkpoint]:
        """List all checkpoints for a workflow."""
        pass

    @abstractmethod
    def delete(self, workflow_id: str, step: int | None = None) -> bool:
        """Delete checkpoint(s). If step is None, delete all for workflow."""
        pass

    @abstractmethod
    def exists(self, workflow_id: str, step: int | None = None) -> bool:
        """Check if checkpoint exists."""
        pass


class InMemoryCheckpointStorage(CheckpointStorage):
    """In-memory checkpoint storage for testing and simple use cases."""

    def __init__(self):
        self._checkpoints: dict[str, dict[int, Checkpoint]] = {}
        self._lock = threading.RLock()

    def save(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            if checkpoint.workflow_id not in self._checkpoints:
                self._checkpoints[checkpoint.workflow_id] = {}
            self._checkpoints[checkpoint.workflow_id][checkpoint.step] = checkpoint

    def load(self, workflow_id: str, step: int | None = None) -> Checkpoint | None:
        with self._lock:
            if workflow_id not in self._checkpoints:
                return None

            workflow_checkpoints = self._checkpoints[workflow_id]
            if not workflow_checkpoints:
                return None

            if step is not None:
                return workflow_checkpoints.get(step)

            # Return latest
            latest_step = max(workflow_checkpoints.keys())
            return workflow_checkpoints[latest_step]

    def list_checkpoints(self, workflow_id: str) -> list[Checkpoint]:
        with self._lock:
            if workflow_id not in self._checkpoints:
                return []
            return sorted(
                self._checkpoints[workflow_id].values(), key=lambda c: c.step
            )

    def delete(self, workflow_id: str, step: int | None = None) -> bool:
        with self._lock:
            if workflow_id not in self._checkpoints:
                return False

            if step is not None:
                if step in self._checkpoints[workflow_id]:
                    del self._checkpoints[workflow_id][step]
                    return True
                return False

            del self._checkpoints[workflow_id]
            return True

    def exists(self, workflow_id: str, step: int | None = None) -> bool:
        with self._lock:
            if workflow_id not in self._checkpoints:
                return False
            if step is not None:
                return step in self._checkpoints[workflow_id]
            return len(self._checkpoints[workflow_id]) > 0


class FileCheckpointStorage(CheckpointStorage):
    """File-based checkpoint storage for persistence across restarts."""

    def __init__(self, base_dir: str | Path):
        """Initialize file storage.

        Args:
            base_dir: Directory to store checkpoint files.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _workflow_dir(self, workflow_id: str) -> Path:
        """Get directory for a workflow's checkpoints."""
        # Sanitize workflow_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in workflow_id)
        return self._base_dir / safe_id

    def _checkpoint_path(self, workflow_id: str, step: int) -> Path:
        """Get path for a specific checkpoint file."""
        return self._workflow_dir(workflow_id) / f"checkpoint_{step:06d}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            workflow_dir = self._workflow_dir(checkpoint.workflow_id)
            workflow_dir.mkdir(parents=True, exist_ok=True)

            path = self._checkpoint_path(checkpoint.workflow_id, checkpoint.step)
            data = checkpoint.model_dump(mode="json")

            # Write atomically
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            temp_path.rename(path)

    def load(self, workflow_id: str, step: int | None = None) -> Checkpoint | None:
        with self._lock:
            workflow_dir = self._workflow_dir(workflow_id)
            if not workflow_dir.exists():
                return None

            if step is not None:
                path = self._checkpoint_path(workflow_id, step)
                if not path.exists():
                    return None
                with open(path) as f:
                    data = json.load(f)
                return Checkpoint.model_validate(data)

            # Find latest checkpoint
            checkpoints = list(workflow_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                return None

            latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
            with open(latest) as f:
                data = json.load(f)
            return Checkpoint.model_validate(data)

    def list_checkpoints(self, workflow_id: str) -> list[Checkpoint]:
        with self._lock:
            workflow_dir = self._workflow_dir(workflow_id)
            if not workflow_dir.exists():
                return []

            checkpoints = []
            for path in sorted(workflow_dir.glob("checkpoint_*.json")):
                with open(path) as f:
                    data = json.load(f)
                checkpoints.append(Checkpoint.model_validate(data))
            return checkpoints

    def delete(self, workflow_id: str, step: int | None = None) -> bool:
        with self._lock:
            workflow_dir = self._workflow_dir(workflow_id)
            if not workflow_dir.exists():
                return False

            if step is not None:
                path = self._checkpoint_path(workflow_id, step)
                if path.exists():
                    path.unlink()
                    return True
                return False

            # Delete all checkpoints
            import shutil

            shutil.rmtree(workflow_dir)
            return True

    def exists(self, workflow_id: str, step: int | None = None) -> bool:
        with self._lock:
            workflow_dir = self._workflow_dir(workflow_id)
            if not workflow_dir.exists():
                return False

            if step is not None:
                return self._checkpoint_path(workflow_id, step).exists()

            return any(workflow_dir.glob("checkpoint_*.json"))


class CheckpointManager:
    """Manages workflow checkpointing for resumable execution.

    This class provides:
    - Automatic checkpointing after each agent completion
    - Resume from any checkpoint
    - No duplicate tool calls on resume
    - State recovery after crashes

    Example:
        manager = CheckpointManager(storage=FileCheckpointStorage("./checkpoints"))

        # During workflow execution
        manager.create_checkpoint(
            workflow_id="wf-123",
            step=2,
            agent_id="researcher",
            status=AgentStatus.COMPLETED,
            state=current_state,
            metrics=current_metrics,
        )

        # After crash, resume
        checkpoint = manager.get_latest_checkpoint("wf-123")
        if checkpoint:
            resume_from_step = checkpoint.step + 1
            state = checkpoint.state_snapshot
    """

    def __init__(
        self,
        storage: CheckpointStorage | None = None,
        auto_checkpoint: bool = True,
        max_checkpoints: int | None = None,
    ):
        """Initialize the checkpoint manager.

        Args:
            storage: Checkpoint storage backend. Defaults to in-memory.
            auto_checkpoint: If True, automatically create checkpoints.
            max_checkpoints: Max checkpoints to keep per workflow (oldest deleted).
        """
        self._storage = storage or InMemoryCheckpointStorage()
        self._auto_checkpoint = auto_checkpoint
        self._max_checkpoints = max_checkpoints
        self._lock = threading.RLock()

    def create_checkpoint(
        self,
        workflow_id: str,
        step: int,
        agent_id: str,
        status: AgentStatus,
        state: "SharedState | StateSnapshot | dict[str, Any]",
        metrics: ExecutionMetrics,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint.

        Args:
            workflow_id: Workflow identifier.
            step: Current step number.
            agent_id: Agent that completed.
            status: Agent completion status.
            state: Current shared state.
            metrics: Current execution metrics.
            metadata: Optional additional metadata.

        Returns:
            The created checkpoint.
        """
        with self._lock:
            # Convert state to snapshot if needed
            if isinstance(state, dict):
                state_snapshot = StateSnapshot(
                    data=state,
                    version=StateVersion(version=step, timestamp=datetime.now()),
                )
            elif hasattr(state, "snapshot"):
                state_snapshot = state.snapshot()
            else:
                state_snapshot = state

            checkpoint = Checkpoint(
                workflow_id=workflow_id,
                step=step,
                agent_id=agent_id,
                status=status,
                state_snapshot=state_snapshot,
                metrics=metrics,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            self._storage.save(checkpoint)

            # Cleanup old checkpoints if limit set
            if self._max_checkpoints:
                self._cleanup_old_checkpoints(workflow_id)

            return checkpoint

    def _cleanup_old_checkpoints(self, workflow_id: str) -> None:
        """Remove old checkpoints beyond the limit."""
        checkpoints = self._storage.list_checkpoints(workflow_id)
        if len(checkpoints) > self._max_checkpoints:
            # Delete oldest checkpoints
            to_delete = len(checkpoints) - self._max_checkpoints
            for cp in checkpoints[:to_delete]:
                self._storage.delete(workflow_id, cp.step)

    def get_checkpoint(self, workflow_id: str, step: int) -> Checkpoint:
        """Get a specific checkpoint.

        Args:
            workflow_id: Workflow identifier.
            step: Step number.

        Returns:
            The checkpoint.

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist.
        """
        checkpoint = self._storage.load(workflow_id, step)
        if checkpoint is None:
            raise CheckpointNotFoundError(workflow_id, step)
        return checkpoint

    def get_latest_checkpoint(self, workflow_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Latest checkpoint or None if no checkpoints.
        """
        return self._storage.load(workflow_id)

    def has_checkpoint(self, workflow_id: str, step: int | None = None) -> bool:
        """Check if checkpoint exists.

        Args:
            workflow_id: Workflow identifier.
            step: Optional specific step.

        Returns:
            True if checkpoint exists.
        """
        return self._storage.exists(workflow_id, step)

    def list_checkpoints(self, workflow_id: str) -> list[Checkpoint]:
        """List all checkpoints for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            List of checkpoints ordered by step.
        """
        return self._storage.list_checkpoints(workflow_id)

    def delete_checkpoints(self, workflow_id: str, step: int | None = None) -> bool:
        """Delete checkpoint(s).

        Args:
            workflow_id: Workflow identifier.
            step: Specific step to delete, or None for all.

        Returns:
            True if deleted.
        """
        return self._storage.delete(workflow_id, step)

    def get_resume_point(self, workflow_id: str) -> tuple[int, StateSnapshot, ExecutionMetrics] | None:
        """Get the resume point for a workflow.

        Returns the step to resume from and the state/metrics to restore.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Tuple of (resume_step, state_snapshot, metrics) or None.
        """
        checkpoint = self.get_latest_checkpoint(workflow_id)
        if checkpoint is None:
            return None

        # Resume from next step if last completed successfully
        if checkpoint.status == AgentStatus.COMPLETED:
            resume_step = checkpoint.step + 1
        else:
            # Retry failed/blocked step
            resume_step = checkpoint.step

        return (resume_step, checkpoint.state_snapshot, checkpoint.metrics)


# Import here to avoid circular import
from .state import SharedState


class ResumableWorkflow:
    """Mixin for making workflows resumable.

    This class can be used to wrap workflow execution with
    automatic checkpointing and resume capability.

    Example:
        resumable = ResumableWorkflow(
            workflow_id="wf-123",
            checkpoint_manager=manager,
        )

        # Start or resume workflow
        async with resumable.execution_context() as ctx:
            if ctx.is_resumed:
                print(f"Resuming from step {ctx.resume_step}")

            for step, agent in enumerate(agents):
                if step < ctx.resume_step:
                    continue  # Skip already completed steps

                result = await agent.run(ctx.state)

                # Checkpoint after each agent
                resumable.checkpoint(
                    step=step,
                    agent_id=agent.id,
                    status=AgentStatus.COMPLETED,
                )
    """

    def __init__(
        self,
        workflow_id: str,
        checkpoint_manager: CheckpointManager,
        initial_state: dict[str, Any] | None = None,
    ):
        """Initialize resumable workflow.

        Args:
            workflow_id: Unique workflow identifier.
            checkpoint_manager: Checkpoint manager instance.
            initial_state: Initial state data.
        """
        self._workflow_id = workflow_id
        self._checkpoint_manager = checkpoint_manager
        self._state = SharedState(initial_data=initial_state)
        self._metrics = ExecutionMetrics()
        self._current_step = 0
        self._is_resumed = False

    @property
    def workflow_id(self) -> str:
        """Workflow identifier."""
        return self._workflow_id

    @property
    def state(self) -> SharedState:
        """Current shared state."""
        return self._state

    @property
    def metrics(self) -> ExecutionMetrics:
        """Current execution metrics."""
        return self._metrics

    @property
    def current_step(self) -> int:
        """Current step number."""
        return self._current_step

    @property
    def is_resumed(self) -> bool:
        """Whether this workflow was resumed from checkpoint."""
        return self._is_resumed

    def try_resume(self) -> int:
        """Attempt to resume from checkpoint.

        Returns:
            Step to resume from (0 if no checkpoint).
        """
        resume_point = self._checkpoint_manager.get_resume_point(self._workflow_id)

        if resume_point is None:
            return 0

        resume_step, state_snapshot, metrics = resume_point

        # Restore state
        self._state = SharedState(initial_data=state_snapshot.data)
        self._metrics = metrics
        self._current_step = resume_step
        self._is_resumed = True

        return resume_step

    def checkpoint(
        self,
        step: int,
        agent_id: str,
        status: AgentStatus,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint at current position.

        Args:
            step: Step number.
            agent_id: Agent that completed.
            status: Completion status.
            metadata: Optional metadata.

        Returns:
            Created checkpoint.
        """
        self._current_step = step
        return self._checkpoint_manager.create_checkpoint(
            workflow_id=self._workflow_id,
            step=step,
            agent_id=agent_id,
            status=status,
            state=self._state,
            metrics=self._metrics,
            metadata=metadata,
        )

    def clear_checkpoints(self) -> bool:
        """Clear all checkpoints for this workflow."""
        return self._checkpoint_manager.delete_checkpoints(self._workflow_id)
