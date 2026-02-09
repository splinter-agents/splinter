"""Splinter Coordination Layer.

The Coordination Layer makes multiple agents behave like a system, not scripts.

Components:
- state: Shared state store (canonical state)
- ownership: State ownership and write boundaries
- schema: Schema-enforced handoffs
- checkpoint: Resumable execution with checkpointing
- execution: Chain awareness, goals, eligibility, completion signals, wait reasons
"""

from .checkpoint import (
    CheckpointManager,
    CheckpointStorage,
    FileCheckpointStorage,
    InMemoryCheckpointStorage,
    ResumableWorkflow,
)
from .ownership import ProtectedState, StateOwnershipManager
from .schema import HandoffManager, SchemaValidator, create_schema_from_example
from .state import SharedState
from .execution import (
    ChainContext,
    GoalTracker,
    Goal,
    ActionEligibility,
    CompletionTracker,
    CompletionSignal,
    WaitTracker,
    WaitState,
    WaitReason,
    ExecutionRecord,
)

__all__ = [
    # State
    "SharedState",
    # Ownership
    "StateOwnershipManager",
    "ProtectedState",
    # Schema
    "SchemaValidator",
    "HandoffManager",
    "create_schema_from_example",
    # Checkpoint
    "CheckpointManager",
    "CheckpointStorage",
    "InMemoryCheckpointStorage",
    "FileCheckpointStorage",
    "ResumableWorkflow",
    # Execution context
    "ChainContext",
    "ExecutionRecord",
    # Goals
    "GoalTracker",
    "Goal",
    # Eligibility
    "ActionEligibility",
    # Completion
    "CompletionTracker",
    "CompletionSignal",
    # Wait tracking
    "WaitTracker",
    "WaitState",
    "WaitReason",
]
