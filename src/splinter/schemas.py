"""Core types and data models for Splinter."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class AgentStatus(str, Enum):
    """Status of an agent in the workflow."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROK = "grok"
    MOCK = "mock"


class EvictionPolicy(str, Enum):
    """Memory eviction policies."""

    FIFO = "fifo"  # First in, first out
    LRU = "lru"  # Least recently used
    TTL = "ttl"  # Time-to-live based


# =============================================================================
# Execution Limits
# =============================================================================


class ExecutionLimits(BaseModel):
    """Limits for workflow execution."""

    max_budget: float | None = Field(default=None, description="Maximum spend in dollars")
    max_steps: int | None = Field(default=None, description="Maximum number of steps")
    max_time_seconds: float | None = Field(default=None, description="Maximum wall-clock time")

    # Per-agent limits (can override workflow limits)
    per_agent_max_steps: int | None = Field(default=None, description="Max steps per agent")
    per_agent_max_budget: float | None = Field(default=None, description="Max budget per agent")


class ExecutionMetrics(BaseModel):
    """Current execution metrics."""

    total_cost: float = Field(default=0.0, description="Total cost so far")
    total_steps: int = Field(default=0, description="Total steps executed")
    total_tokens: int = Field(default=0, description="Total tokens used")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    start_time: datetime | None = Field(default=None, description="Workflow start time")
    elapsed_seconds: float = Field(default=0.0, description="Elapsed time in seconds")

    def elapsed_time(self) -> float:
        """Calculate elapsed time from start."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()


# =============================================================================
# Loop Detection
# =============================================================================


class StepSignature(BaseModel):
    """Signature of an execution step for loop detection."""

    agent_id: str
    tool_name: str | None = None
    input_hash: str
    output_hash: str
    state_hash: str
    timestamp: datetime = Field(default_factory=datetime.now)


class LoopDetectionConfig(BaseModel):
    """Configuration for loop detection."""

    max_repeated_outputs: int = Field(default=3, description="Max identical outputs before flagging")
    max_no_state_change: int = Field(default=5, description="Max steps without state change")
    max_ping_pong: int = Field(default=4, description="Max A↔B ping-pong cycles")
    window_size: int = Field(default=20, description="Number of recent steps to analyze")


# =============================================================================
# Tool Access Control
# =============================================================================


class ToolPermission(BaseModel):
    """Permission definition for a tool."""

    tool_name: str
    allowed_agents: list[str] = Field(default_factory=list)
    denied_agents: list[str] = Field(default_factory=list)
    require_approval: bool = Field(default=False)


class ToolCall(BaseModel):
    """Record of a tool call."""

    tool_name: str
    agent_id: str
    arguments: dict[str, Any]
    result: Any | None = None
    success: bool = False
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float | None = None


# =============================================================================
# Memory
# =============================================================================


class MemoryConfig(BaseModel):
    """Configuration for memory limits."""

    max_size_bytes: int = Field(default=10 * 1024 * 1024, description="Max memory size (10MB default)")
    ttl_seconds: float | None = Field(default=3600, description="Time-to-live for entries (1hr default)")
    eviction_policy: EvictionPolicy = Field(default=EvictionPolicy.FIFO)
    max_entries: int | None = Field(default=1000, description="Max number of entries")


class MemoryEntry(BaseModel):
    """A single memory entry."""

    key: str
    value: Any
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: datetime = Field(default_factory=datetime.now)
    ttl_seconds: float | None = None
    agent_id: str | None = None


# =============================================================================
# Shared State
# =============================================================================


class StateVersion(BaseModel):
    """Version information for state."""

    version: int
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_id: str | None = None
    changed_fields: list[str] = Field(default_factory=list)


class StateSnapshot(BaseModel):
    """A snapshot of the shared state."""

    data: dict[str, Any] = Field(default_factory=dict)
    version: StateVersion
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# State Ownership
# =============================================================================


class FieldOwnership(BaseModel):
    """Ownership definition for a state field."""

    field_pattern: str  # Supports wildcards like "research.*"
    owner_agent_id: str
    read_only_for: list[str] = Field(default_factory=list)  # Empty = all can read


# =============================================================================
# Schema Handoffs
# =============================================================================


class HandoffSchema(BaseModel):
    """Schema definition for agent handoffs."""

    source_agent_id: str
    target_agent_id: str
    output_schema: dict[str, Any]  # JSON Schema
    required_fields: list[str] = Field(default_factory=list)
    strict: bool = Field(default=True, description="Reject on any validation failure")


# =============================================================================
# Checkpointing
# =============================================================================


class Checkpoint(BaseModel):
    """A checkpoint of workflow execution."""

    workflow_id: str
    step: int
    agent_id: str
    status: AgentStatus
    state_snapshot: StateSnapshot
    metrics: ExecutionMetrics
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# LLM Calls
# =============================================================================


class LLMMessage(BaseModel):
    """A message in an LLM conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None


class LLMRequest(BaseModel):
    """A request to an LLM provider."""

    provider: LLMProvider
    model: str
    messages: list[LLMMessage]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A response from an LLM provider."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    model: str
    provider: LLMProvider
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = None


# =============================================================================
# Agent Definition
# =============================================================================


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    agent_id: str
    name: str | None = None
    description: str | None = None
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4"
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] | None = None  # None = all tools allowed
    state_ownership: list[str] = Field(default_factory=list)  # Fields this agent owns
    output_schema: dict[str, Any] | None = None
    max_steps: int | None = None
    max_budget: float | None = None


# =============================================================================
# Workflow Definition
# =============================================================================


class WorkflowStep(BaseModel):
    """A step in a workflow."""

    agent_id: str
    depends_on: list[str] = Field(default_factory=list)
    input_mapping: dict[str, str] = Field(default_factory=dict)  # Map state fields to agent input
    condition: str | None = None  # Optional condition expression


class WorkflowConfig(BaseModel):
    """Configuration for a workflow."""

    workflow_id: str
    name: str | None = None
    description: str | None = None
    agents: list[AgentConfig]
    steps: list[WorkflowStep]
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    loop_detection: LoopDetectionConfig = Field(default_factory=LoopDetectionConfig)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    handoff_schemas: list[HandoffSchema] = Field(default_factory=list)
    field_ownership: list[FieldOwnership] = Field(default_factory=list)
    checkpoint_enabled: bool = Field(default=True)


# =============================================================================
# Execution Context
# =============================================================================


class ExecutionContext(BaseModel):
    """Context passed through workflow execution."""

    workflow_id: str
    run_id: str
    current_step: int = 0
    current_agent_id: str | None = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Auth tokens, user info, etc. that should be injected into tool calls
    execution_context: dict[str, Any] = Field(default_factory=dict)
