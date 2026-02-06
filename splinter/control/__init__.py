"""Splinter Control Layer.

The Control Layer provides safety mechanisms that stop agents from doing damage.
These are enforced by the system, not prompts.

Components:
- limits: Execution limits (cost/time/steps)
- loop_detection: State-based loop detection
- tool_access: Per-agent tool access control
- memory: Memory limits with TTL and eviction
- rate_limit: Rate limiting per agent/tool
- retry: Retry strategies (basic, fail-closed)
- circuit_breaker: Global safety triggers
- decisions: Decision enforcement (no flip-flopping)
- rules: Custom rules engine
"""

from .limits import LimitsEnforcer, PerAgentLimits
from .loop_detection import LoopBreaker, LoopDetector
from .memory import AgentMemory, MemoryStore
from .tool_access import ToolAccessController, ToolRegistry
from .rate_limit import RateLimiter, RateLimitConfig
from .retry import RetryStrategy, RetryConfig, RetryMode, with_retry
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from .decisions import DecisionEnforcer, Decision, DecisionType
from .rules import RulesEngine, Rule, RuleAction, RulePriority

__all__ = [
    # Limits
    "LimitsEnforcer",
    "PerAgentLimits",
    # Loop detection
    "LoopDetector",
    "LoopBreaker",
    # Tool access
    "ToolAccessController",
    "ToolRegistry",
    # Memory
    "MemoryStore",
    "AgentMemory",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    # Retry
    "RetryStrategy",
    "RetryConfig",
    "RetryMode",
    "with_retry",
    # Circuit breakers
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitState",
    # Decisions
    "DecisionEnforcer",
    "Decision",
    "DecisionType",
    # Rules
    "RulesEngine",
    "Rule",
    "RuleAction",
    "RulePriority",
]
