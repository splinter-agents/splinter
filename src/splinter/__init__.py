"""Splinter - Multi-agent control and coordination.

Simple usage:
    from splinter import Splinter

    s = Splinter(openai_key="sk-...")
    result = await s.run("helper", "What is 2+2?")

Advanced usage:
    from splinter import Gateway, Workflow, AgentConfig
"""

__version__ = "0.1.0"

# =============================================================================
# SIMPLE API (start here)
# =============================================================================

from .client import Splinter, run_sync

# =============================================================================
# ADVANCED API
# =============================================================================

# Types
from .schemas import (
    AgentConfig,
    AgentStatus,
    Checkpoint,
    EvictionPolicy,
    ExecutionContext,
    ExecutionLimits,
    ExecutionMetrics,
    FieldOwnership,
    HandoffSchema,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LoopDetectionConfig,
    MemoryConfig,
    MemoryEntry,
    StateSnapshot,
    StateVersion,
    StepSignature,
    ToolCall,
    ToolPermission,
    WorkflowConfig,
    WorkflowStatus,
    WorkflowStep,
)

# Exceptions
from .exceptions import (
    AgentNotFoundError,
    BudgetExceededError,
    CheckpointError,
    CheckpointNotFoundError,
    ExecutionLimitError,
    GatewayError,
    LoopDetectedError,
    MemoryLimitError,
    ProviderError,
    ProviderNotFoundError,
    SchemaValidationError,
    SplinterError,
    StateError,
    StateOwnershipError,
    StepLimitExceededError,
    TimeLimitExceededError,
    ToolAccessDeniedError,
    WorkflowError,
    WorkflowExecutionError,
)

# Control Layer
from .control import (
    AgentMemory,
    LimitsEnforcer,
    LoopBreaker,
    LoopDetector,
    MemoryStore,
    PerAgentLimits,
    ToolAccessController,
    ToolRegistry,
)

# Coordination Layer
from .coordination import (
    CheckpointManager,
    CheckpointStorage,
    FileCheckpointStorage,
    HandoffManager,
    InMemoryCheckpointStorage,
    ProtectedState,
    ResumableWorkflow,
    SchemaValidator,
    SharedState,
    StateOwnershipManager,
    create_schema_from_example,
)

# Gateway Layer
from .gateway import (
    AnthropicProvider,
    BaseProvider,
    CallRecord,
    Gateway,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
    ProviderFactory,
    SmartMockProvider,
)

# Workflow Layer
from .workflow import (
    Agent,
    AgentBuilder,
    Workflow,
    WorkflowResult,
)

# Utils
from .utils import configure_logging, get_logger

# Cloud (Paid Features)
from .cloud import (
    CloudClient,
    CloudConfig,
    Command,
    CommandHandler,
    CommandType,
    StateSync,
    SyncEvent,
)

__all__ = [
    # Simple API
    "Splinter",
    "run_sync",
    # Version
    "__version__",
    # Types
    "AgentConfig",
    "AgentStatus",
    "Checkpoint",
    "EvictionPolicy",
    "ExecutionContext",
    "ExecutionLimits",
    "ExecutionMetrics",
    "FieldOwnership",
    "HandoffSchema",
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LoopDetectionConfig",
    "MemoryConfig",
    "MemoryEntry",
    "StateSnapshot",
    "StateVersion",
    "StepSignature",
    "ToolCall",
    "ToolPermission",
    "WorkflowConfig",
    "WorkflowStatus",
    "WorkflowStep",
    # Exceptions
    "AgentNotFoundError",
    "BudgetExceededError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "ExecutionLimitError",
    "GatewayError",
    "LoopDetectedError",
    "MemoryLimitError",
    "ProviderError",
    "ProviderNotFoundError",
    "SchemaValidationError",
    "SplinterError",
    "StateError",
    "StateOwnershipError",
    "StepLimitExceededError",
    "TimeLimitExceededError",
    "ToolAccessDeniedError",
    "WorkflowError",
    "WorkflowExecutionError",
    # Control
    "AgentMemory",
    "LimitsEnforcer",
    "LoopBreaker",
    "LoopDetector",
    "MemoryStore",
    "PerAgentLimits",
    "ToolAccessController",
    "ToolRegistry",
    # Coordination
    "CheckpointManager",
    "CheckpointStorage",
    "FileCheckpointStorage",
    "HandoffManager",
    "InMemoryCheckpointStorage",
    "ProtectedState",
    "ResumableWorkflow",
    "SchemaValidator",
    "SharedState",
    "StateOwnershipManager",
    "create_schema_from_example",
    # Gateway
    "AnthropicProvider",
    "BaseProvider",
    "CallRecord",
    "Gateway",
    "GeminiProvider",
    "MockProvider",
    "OpenAIProvider",
    "ProviderFactory",
    "SmartMockProvider",
    # Workflow
    "Agent",
    "AgentBuilder",
    "Workflow",
    "WorkflowResult",
    # Utils
    "configure_logging",
    "get_logger",
    # Cloud (Paid Features)
    "CloudClient",
    "CloudConfig",
    "Command",
    "CommandHandler",
    "CommandType",
    "StateSync",
    "SyncEvent",
]
