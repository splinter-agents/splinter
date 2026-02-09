"""Custom exceptions for Splinter."""

from typing import Any


class SplinterError(Exception):
    """Base exception for all Splinter errors."""

    pass


# =============================================================================
# Control Layer Exceptions
# =============================================================================


class ExecutionLimitError(SplinterError):
    """Raised when an execution limit is exceeded."""

    def __init__(self, limit_type: str, current: float, limit: float, message: str | None = None):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        super().__init__(message or f"{limit_type} limit exceeded: {current} >= {limit}")


class BudgetExceededError(ExecutionLimitError):
    """Raised when budget limit is exceeded."""

    def __init__(self, current: float, limit: float):
        super().__init__("budget", current, limit, f"Budget exceeded: ${current:.4f} >= ${limit:.4f}")


class StepLimitExceededError(ExecutionLimitError):
    """Raised when step limit is exceeded."""

    def __init__(self, current: int, limit: int):
        super().__init__("steps", current, limit, f"Step limit exceeded: {current} >= {limit}")


class TimeLimitExceededError(ExecutionLimitError):
    """Raised when time limit is exceeded."""

    def __init__(self, elapsed: float, limit: float):
        super().__init__(
            "time", elapsed, limit, f"Time limit exceeded: {elapsed:.2f}s >= {limit:.2f}s"
        )


class LoopDetectedError(SplinterError):
    """Raised when a loop is detected in agent execution."""

    def __init__(self, pattern: str, occurrences: int):
        self.pattern = pattern
        self.occurrences = occurrences
        super().__init__(f"Loop detected: {pattern} repeated {occurrences} times")


class ToolAccessDeniedError(SplinterError):
    """Raised when an agent tries to use a tool it's not authorized for."""

    def __init__(self, agent_id: str, tool_name: str):
        self.agent_id = agent_id
        self.tool_name = tool_name
        super().__init__(f"Agent '{agent_id}' is not authorized to use tool '{tool_name}'")


class MemoryLimitError(SplinterError):
    """Raised when memory limits are exceeded."""

    def __init__(self, current_size: int, max_size: int):
        self.current_size = current_size
        self.max_size = max_size
        super().__init__(f"Memory limit exceeded: {current_size} bytes >= {max_size} bytes")


class RateLimitExceededError(SplinterError):
    """Raised when rate limit is exceeded."""

    def __init__(self, entity_type: str, entity_id: str, limit: int, window: float):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded for {entity_type} '{entity_id}': "
            f"{limit} calls per {window}s"
        )


class RetryExhaustedError(SplinterError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_error}")


class CircuitOpenError(SplinterError):
    """Raised when circuit breaker is open."""

    def __init__(self, breaker_id: str, reason: str):
        self.breaker_id = breaker_id
        self.reason = reason
        super().__init__(f"Circuit breaker '{breaker_id}' is OPEN: {reason}")


class DecisionLockError(SplinterError):
    """Raised when trying to change a locked decision."""

    def __init__(self, decision_id: str, agent_id: str, locked_by: str):
        self.decision_id = decision_id
        self.agent_id = agent_id
        self.locked_by = locked_by
        super().__init__(
            f"Decision '{decision_id}' is locked by agent '{locked_by}'. "
            f"Agent '{agent_id}' cannot change it."
        )


class RuleViolationError(SplinterError):
    """Raised when a rule is violated."""

    def __init__(self, rule_id: str, rule_name: str, message: str):
        self.rule_id = rule_id
        self.rule_name = rule_name
        super().__init__(f"Rule violation [{rule_name}]: {message}")


# =============================================================================
# Coordination Layer Exceptions
# =============================================================================


class StateError(SplinterError):
    """Base exception for state-related errors."""

    pass


class StateOwnershipError(StateError):
    """Raised when an agent tries to write to a field it doesn't own."""

    def __init__(self, agent_id: str, field: str, owner: str | None):
        self.agent_id = agent_id
        self.field = field
        self.owner = owner
        owner_msg = f"owned by '{owner}'" if owner else "has no assigned owner"
        super().__init__(f"Agent '{agent_id}' cannot write to field '{field}' ({owner_msg})")


class SchemaValidationError(SplinterError):
    """Raised when output doesn't match expected schema."""

    def __init__(self, agent_id: str, errors: list[str]):
        self.agent_id = agent_id
        self.errors = errors
        error_list = "; ".join(errors[:5])
        if len(errors) > 5:
            error_list += f" ... and {len(errors) - 5} more"
        super().__init__(f"Schema validation failed for agent '{agent_id}': {error_list}")


class CheckpointError(SplinterError):
    """Raised when checkpoint operations fail."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""

    def __init__(self, workflow_id: str, step: int | None = None):
        self.workflow_id = workflow_id
        self.step = step
        if step is not None:
            super().__init__(f"Checkpoint not found for workflow '{workflow_id}' at step {step}")
        else:
            super().__init__(f"No checkpoints found for workflow '{workflow_id}'")


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


# =============================================================================
# Gateway Exceptions
# =============================================================================


class GatewayError(SplinterError):
    """Base exception for gateway errors."""

    pass


class ProviderError(GatewayError):
    """Raised when a provider call fails."""

    def __init__(self, provider: str, message: str, original_error: Exception | None = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Provider '{provider}' error: {message}")


class ProviderNotFoundError(GatewayError):
    """Raised when a requested provider is not registered."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Provider '{provider}' not found. Available providers: openai, anthropic, gemini")


# =============================================================================
# Workflow Exceptions
# =============================================================================


class WorkflowError(SplinterError):
    """Base exception for workflow errors."""

    pass


class AgentNotFoundError(WorkflowError):
    """Raised when a referenced agent doesn't exist."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent '{agent_id}' not found in workflow")


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""

    def __init__(self, message: str, step: int | None = None, agent_id: str | None = None):
        self.step = step
        self.agent_id = agent_id
        prefix = ""
        if step is not None:
            prefix = f"[Step {step}] "
        if agent_id:
            prefix += f"[Agent: {agent_id}] "
        super().__init__(f"{prefix}{message}")
