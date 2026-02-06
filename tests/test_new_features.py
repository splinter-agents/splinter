"""Tests for new Control and Coordination features."""

import asyncio
import pytest
import time

# Control features
from splinter.control.rate_limit import RateLimiter, RateLimitConfig, RateLimitExceededError
from splinter.control.retry import RetryStrategy, RetryConfig, RetryMode, RetryExhaustedError
from splinter.control.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitOpenError,
)
from splinter.control.decisions import DecisionEnforcer, DecisionType, DecisionLockError
from splinter.control.rules import (
    RulesEngine,
    Rule,
    RuleAction,
    RulePriority,
    RuleViolationError,
    budget_rule,
    agent_blocked_rule,
)

# Coordination features
from splinter.coordination.execution import (
    ChainContext,
    GoalTracker,
    Goal,
    ActionEligibility,
    CompletionTracker,
    WaitTracker,
    WaitReason,
    AgentNotEligibleError,
    CompletionNotDeclaredError,
)


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    def test_agent_rate_limit(self):
        """Test agent rate limiting."""
        limiter = RateLimiter()
        limiter.set_agent_limit("test_agent", calls=3, window_seconds=60)

        # First 3 calls should work
        for _ in range(3):
            limiter.check_agent("test_agent")
            limiter.record_agent_call("test_agent")

        # 4th call should fail
        with pytest.raises(RateLimitExceededError):
            limiter.check_agent("test_agent")

    def test_tool_rate_limit(self):
        """Test tool rate limiting."""
        limiter = RateLimiter()
        limiter.set_tool_limit("web_search", calls=2, window_seconds=60)

        limiter.check_tool("web_search")
        limiter.record_tool_call("web_search")
        limiter.check_tool("web_search")
        limiter.record_tool_call("web_search")

        with pytest.raises(RateLimitExceededError):
            limiter.check_tool("web_search")

    def test_no_limit_allows_all(self):
        """Test that no limit allows all calls."""
        limiter = RateLimiter()

        for _ in range(100):
            assert limiter.check_agent("any_agent")
            limiter.record_agent_call("any_agent")


# =============================================================================
# Retry Strategy Tests
# =============================================================================


class TestRetryStrategy:
    @pytest.mark.asyncio
    async def test_successful_first_try(self):
        """Test that successful call doesn't retry."""
        strategy = RetryStrategy(config=RetryConfig(max_attempts=3))

        async def success():
            return "ok"

        result = await strategy.execute(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failure."""
        strategy = RetryStrategy(config=RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
        ))

        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "ok"

        result = await strategy.execute(fail_then_succeed)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry exhaustion."""
        strategy = RetryStrategy(config=RetryConfig(
            max_attempts=2,
            initial_delay=0.01,
        ))

        async def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError):
            await strategy.execute(always_fail)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        """Test circuit starts closed."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED

    def test_trip_on_failures(self):
        """Test circuit trips after failures."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=3),
        )

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    def test_open_circuit_blocks(self):
        """Test open circuit blocks calls."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )

        breaker.record_failure()
        assert breaker.is_open

        with pytest.raises(CircuitOpenError):
            breaker.check()

    def test_success_keeps_closed(self):
        """Test successful calls keep circuit closed."""
        breaker = CircuitBreaker("test")

        for _ in range(10):
            breaker.check()
            breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_registry(self):
        """Test circuit breaker registry."""
        registry = CircuitBreakerRegistry()
        registry.register("api")
        registry.register("database")

        assert registry.get("api") is not None
        assert registry.get("database") is not None
        assert registry.get("nonexistent") is None


# =============================================================================
# Decision Enforcer Tests
# =============================================================================


class TestDecisionEnforcer:
    def test_record_decision(self):
        """Test recording a decision."""
        enforcer = DecisionEnforcer()

        decision = enforcer.record_decision(
            decision_id="approach",
            agent_id="planner",
            decision_type=DecisionType.STRATEGY,
            value="parallel",
        )

        assert decision.value == "parallel"
        assert enforcer.get_decision("approach") == "parallel"

    def test_locked_decision_cannot_change(self):
        """Test locked decision cannot be changed."""
        enforcer = DecisionEnforcer(auto_lock=True)

        enforcer.record_decision(
            decision_id="choice",
            agent_id="planner",
            decision_type=DecisionType.CHOICE,
            value="option_a",
        )

        with pytest.raises(DecisionLockError):
            enforcer.record_decision(
                decision_id="choice",
                agent_id="other_agent",
                decision_type=DecisionType.CHOICE,
                value="option_b",
            )

    def test_unlocked_decision_can_change(self):
        """Test unlocked decision can be changed."""
        enforcer = DecisionEnforcer(auto_lock=False)

        enforcer.record_decision(
            decision_id="choice",
            agent_id="planner",
            decision_type=DecisionType.CHOICE,
            value="option_a",
        )

        enforcer.update_decision(
            decision_id="choice",
            agent_id="other_agent",
            new_value="option_b",
        )

        assert enforcer.get_decision("choice") == "option_b"


# =============================================================================
# Rules Engine Tests
# =============================================================================


class TestRulesEngine:
    def test_block_rule(self):
        """Test blocking rule."""
        engine = RulesEngine()
        engine.add_rule(budget_rule(threshold=1.0))

        # Should pass with enough budget
        matches = engine.evaluate({"remaining_budget": 5.0})
        assert len(matches) == 0

        # Should fail with low budget
        with pytest.raises(RuleViolationError):
            engine.evaluate({"remaining_budget": 0.5})

    def test_warn_rule(self):
        """Test warning rule (doesn't block)."""
        engine = RulesEngine()
        engine.add_rule(Rule(
            rule_id="test_warn",
            name="Test Warning",
            description="Warns on condition",
            condition=lambda ctx: ctx.get("steps", 0) > 10,
            action=RuleAction.WARN,
        ))

        # Should return matches but not raise
        matches = engine.evaluate({"steps": 20})
        assert len(matches) == 1
        assert matches[0].action == RuleAction.WARN

    def test_agent_blocked_rule(self):
        """Test agent blocking rule."""
        engine = RulesEngine()
        engine.add_rule(agent_blocked_rule("bad_agent"))

        # Should block bad_agent
        with pytest.raises(RuleViolationError):
            engine.evaluate({"agent_id": "bad_agent"})

        # Should allow good_agent
        matches = engine.evaluate({"agent_id": "good_agent"})
        assert len(matches) == 0


# =============================================================================
# Chain Context Tests
# =============================================================================


class TestChainContext:
    def test_register_agents(self):
        """Test agent registration."""
        context = ChainContext(workflow_id="test")
        context.register_agent("researcher", role="Research topics")
        context.register_agent("writer", role="Write content")

        agents = context.get_all_agents()
        assert len(agents) == 2

    def test_execution_history(self):
        """Test execution history tracking."""
        context = ChainContext(workflow_id="test")
        context.register_agent("agent1")

        context.start_execution("agent1", step=0, input_summary="Start")
        context.complete_execution("agent1", step=0, output_summary="Done")

        history = context.get_execution_history()
        assert len(history) == 1
        assert history[0].output_summary == "Done"

    def test_context_for_agent(self):
        """Test getting context for an agent."""
        context = ChainContext(workflow_id="test")
        context.register_agent("agent1", role="Role 1")
        context.register_agent("agent2", role="Role 2")

        agent_context = context.get_context_for_agent("agent1")

        assert agent_context["current_agent"] == "agent1"
        assert len(agent_context["other_agents"]) == 1
        assert agent_context["other_agents"][0]["agent_id"] == "agent2"


# =============================================================================
# Goal Tracker Tests
# =============================================================================


class TestGoalTracker:
    def test_set_and_get_goal(self):
        """Test setting and getting goals."""
        tracker = GoalTracker()
        tracker.set_goal(Goal(
            goal_id="main",
            description="Complete the task",
            success_criteria=["Step 1", "Step 2"],
        ))

        goal = tracker.get_goal("main")
        assert goal is not None
        assert goal.description == "Complete the task"

    def test_progress_tracking(self):
        """Test progress tracking."""
        tracker = GoalTracker()
        tracker.set_goal(Goal(
            goal_id="main",
            description="Test",
            success_criteria=[],
        ))

        tracker.update_progress("main", 0.5)
        assert tracker.get_progress("main") == 0.5

    def test_criteria_achievement(self):
        """Test criteria achievement tracking."""
        achieved = []
        tracker = GoalTracker(on_goal_achieved=lambda g: achieved.append(g))

        tracker.set_goal(Goal(
            goal_id="main",
            description="Test",
            success_criteria=["A", "B"],
        ))

        tracker.mark_criterion_met("main", "A")
        assert not tracker.is_achieved("main")

        tracker.mark_criterion_met("main", "B")
        assert tracker.is_achieved("main")
        assert len(achieved) == 1


# =============================================================================
# Action Eligibility Tests
# =============================================================================


class TestActionEligibility:
    def test_no_rules_allows_all(self):
        """Test that no rules allows all agents."""
        eligibility = ActionEligibility()

        assert eligibility.check("any_agent")
        assert eligibility.is_eligible("any_agent")

    def test_set_eligible(self):
        """Test setting eligible agents."""
        eligibility = ActionEligibility()
        eligibility.set_eligible("agent1", "agent2")

        assert eligibility.check("agent1")
        assert eligibility.check("agent2")

        with pytest.raises(AgentNotEligibleError):
            eligibility.check("agent3")

    def test_transfer_eligibility(self):
        """Test transferring eligibility."""
        eligibility = ActionEligibility()
        eligibility.set_eligible("agent1")

        eligibility.transfer("agent1", "agent2")

        assert not eligibility.is_eligible("agent1")
        assert eligibility.is_eligible("agent2")


# =============================================================================
# Completion Tracker Tests
# =============================================================================


class TestCompletionTracker:
    def test_declare_complete(self):
        """Test declaring completion."""
        tracker = CompletionTracker()

        tracker.declare_complete(
            agent_id="agent1",
            step=0,
            success=True,
            output_summary="Done",
        )

        assert tracker.is_complete("agent1", 0)

    def test_require_completion(self):
        """Test requiring explicit completion."""
        tracker = CompletionTracker(require_explicit=True)

        # Should raise if not declared
        with pytest.raises(CompletionNotDeclaredError):
            tracker.require_completion("agent1", 0)

        # Should pass after declaration
        tracker.declare_complete("agent1", 0, True)
        tracker.require_completion("agent1", 0)


# =============================================================================
# Wait Tracker Tests
# =============================================================================


class TestWaitTracker:
    def test_start_waiting(self):
        """Test starting wait state."""
        tracker = WaitTracker()

        tracker.start_waiting(
            agent_id="agent1",
            reason=WaitReason.DEPENDENCY,
            waiting_for="agent2",
        )

        assert tracker.is_waiting("agent1")
        state = tracker.get_wait_state("agent1")
        assert state.waiting_for == "agent2"

    def test_stop_waiting(self):
        """Test stopping wait state."""
        tracker = WaitTracker()
        tracker.start_waiting("agent1", WaitReason.DEPENDENCY)

        tracker.stop_waiting("agent1")

        assert not tracker.is_waiting("agent1")

    def test_get_waiting_for(self):
        """Test getting agents waiting for a target."""
        tracker = WaitTracker()
        tracker.start_waiting("agent1", WaitReason.DEPENDENCY, "agent3")
        tracker.start_waiting("agent2", WaitReason.DEPENDENCY, "agent3")

        waiting = tracker.get_waiting_for("agent3")
        assert len(waiting) == 2
