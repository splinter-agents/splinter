"""Custom rules engine for Splinter.

Allows defining custom rules that are enforced during execution.
Rules can block actions, modify behavior, or trigger alerts.
"""

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from ..exceptions import SplinterError


class RuleViolationError(SplinterError):
    """Raised when a rule is violated."""

    def __init__(self, rule_id: str, rule_name: str, message: str):
        self.rule_id = rule_id
        self.rule_name = rule_name
        super().__init__(f"Rule violation [{rule_name}]: {message}")


class RuleAction(str, Enum):
    """Action to take when rule matches."""

    BLOCK = "block"  # Block the operation
    WARN = "warn"  # Log warning but allow
    LOG = "log"  # Just log
    MODIFY = "modify"  # Modify the context


class RulePriority(int, Enum):
    """Rule priority (higher = evaluated first)."""

    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25


@dataclass
class Rule:
    """A custom rule definition."""

    rule_id: str
    name: str
    description: str
    condition: Callable[[dict[str, Any]], bool]  # Returns True if rule matches
    action: RuleAction = RuleAction.BLOCK
    priority: RulePriority = RulePriority.MEDIUM
    message: str | None = None  # Custom message
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleMatch:
    """Record of a rule match."""

    rule_id: str
    rule_name: str
    action: RuleAction
    context: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message: str | None = None


class RulesEngine:
    """Custom rules engine for enforcing policies.

    Example:
        engine = RulesEngine()

        # Add a rule that blocks expensive operations
        engine.add_rule(Rule(
            rule_id="budget_guard",
            name="Budget Guard",
            description="Block operations when budget is low",
            condition=lambda ctx: ctx.get("remaining_budget", 0) < 1.0,
            action=RuleAction.BLOCK,
            message="Budget too low for this operation",
        ))

        # Add a rule that warns about long-running agents
        engine.add_rule(Rule(
            rule_id="long_running",
            name="Long Running Warning",
            description="Warn when agent runs too long",
            condition=lambda ctx: ctx.get("elapsed_seconds", 0) > 60,
            action=RuleAction.WARN,
        ))

        # Evaluate rules
        context = {"remaining_budget": 0.5, "agent_id": "researcher"}
        engine.evaluate(context)  # Raises RuleViolationError
    """

    def __init__(
        self,
        on_match: Callable[[RuleMatch], None] | None = None,
        on_block: Callable[[Rule, dict[str, Any]], None] | None = None,
    ):
        """Initialize rules engine.

        Args:
            on_match: Callback when any rule matches.
            on_block: Callback when rule blocks operation.
        """
        self._on_match = on_match
        self._on_block = on_block
        self._rules: dict[str, Rule] = {}
        self._matches: list[RuleMatch] = []
        self._lock = threading.RLock()

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        with self._lock:
            self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                rule.enabled = True
                return True
            return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                rule.enabled = False
                return True
            return False

    def evaluate(self, context: dict[str, Any]) -> list[RuleMatch]:
        """Evaluate all rules against context.

        Args:
            context: Context dict with relevant data.

        Returns:
            List of matched rules.

        Raises:
            RuleViolationError: If a BLOCK rule matches.
        """
        with self._lock:
            matches: list[RuleMatch] = []

            # Sort rules by priority (highest first)
            sorted_rules = sorted(
                [r for r in self._rules.values() if r.enabled],
                key=lambda r: r.priority.value,
                reverse=True,
            )

            for rule in sorted_rules:
                try:
                    if rule.condition(context):
                        match = RuleMatch(
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            action=rule.action,
                            context=dict(context),
                            message=rule.message,
                        )
                        matches.append(match)
                        self._matches.append(match)

                        if self._on_match:
                            self._on_match(match)

                        if rule.action == RuleAction.BLOCK:
                            if self._on_block:
                                self._on_block(rule, context)
                            raise RuleViolationError(
                                rule.rule_id,
                                rule.name,
                                rule.message or rule.description,
                            )

                except RuleViolationError:
                    raise
                except Exception:
                    # Rule condition failed, skip this rule
                    pass

            return matches

    def check(self, context: dict[str, Any]) -> bool:
        """Check if context passes all rules (non-throwing).

        Returns:
            True if all rules pass, False if any BLOCK rule matches.
        """
        try:
            self.evaluate(context)
            return True
        except RuleViolationError:
            return False

    def get_matches(
        self,
        rule_id: str | None = None,
        action: RuleAction | None = None,
        limit: int | None = None,
    ) -> list[RuleMatch]:
        """Get rule match history."""
        with self._lock:
            matches = list(self._matches)

            if rule_id:
                matches = [m for m in matches if m.rule_id == rule_id]
            if action:
                matches = [m for m in matches if m.action == action]
            if limit:
                matches = matches[-limit:]

            return matches

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get a rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def get_all_rules(self) -> list[Rule]:
        """Get all rules."""
        with self._lock:
            return list(self._rules.values())

    def clear_matches(self) -> None:
        """Clear match history."""
        with self._lock:
            self._matches.clear()


# Predefined rule factories
def budget_rule(
    threshold: float,
    action: RuleAction = RuleAction.BLOCK,
    rule_id: str = "budget_threshold",
) -> Rule:
    """Create a budget threshold rule."""
    return Rule(
        rule_id=rule_id,
        name="Budget Threshold",
        description=f"Triggers when remaining budget < ${threshold}",
        condition=lambda ctx: ctx.get("remaining_budget", float("inf")) < threshold,
        action=action,
        message=f"Remaining budget below ${threshold}",
    )


def step_limit_rule(
    threshold: int,
    action: RuleAction = RuleAction.WARN,
    rule_id: str = "step_threshold",
) -> Rule:
    """Create a step threshold rule."""
    return Rule(
        rule_id=rule_id,
        name="Step Threshold",
        description=f"Triggers when steps > {threshold}",
        condition=lambda ctx: ctx.get("total_steps", 0) > threshold,
        action=action,
        message=f"Steps exceeded {threshold}",
    )


def agent_blocked_rule(
    agent_id: str,
    action: RuleAction = RuleAction.BLOCK,
) -> Rule:
    """Create a rule that blocks a specific agent."""
    return Rule(
        rule_id=f"block_agent_{agent_id}",
        name=f"Block Agent {agent_id}",
        description=f"Blocks agent {agent_id} from executing",
        condition=lambda ctx: ctx.get("agent_id") == agent_id,
        action=action,
        message=f"Agent {agent_id} is blocked",
    )


def tool_blocked_rule(
    tool_name: str,
    action: RuleAction = RuleAction.BLOCK,
) -> Rule:
    """Create a rule that blocks a specific tool."""
    return Rule(
        rule_id=f"block_tool_{tool_name}",
        name=f"Block Tool {tool_name}",
        description=f"Blocks tool {tool_name} from executing",
        condition=lambda ctx: ctx.get("tool_name") == tool_name,
        action=action,
        message=f"Tool {tool_name} is blocked",
    )


def pattern_rule(
    pattern: str,
    field: str,
    action: RuleAction = RuleAction.BLOCK,
    rule_id: str | None = None,
) -> Rule:
    """Create a rule that matches a regex pattern."""
    compiled = re.compile(pattern)
    return Rule(
        rule_id=rule_id or f"pattern_{field}_{pattern[:10]}",
        name=f"Pattern Match: {pattern}",
        description=f"Matches pattern '{pattern}' in field '{field}'",
        condition=lambda ctx: bool(compiled.search(str(ctx.get(field, "")))),
        action=action,
        message=f"Pattern '{pattern}' matched in {field}",
    )
