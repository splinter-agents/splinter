"""Tests for Splinter Control Layer."""

import pytest
import time

from splinter.control.limits import LimitsEnforcer, PerAgentLimits
from splinter.control.loop_detection import LoopDetector
from splinter.control.tool_access import ToolAccessController
from splinter.control.memory import MemoryStore, AgentMemory
from splinter.types import ExecutionLimits, LoopDetectionConfig, MemoryConfig, EvictionPolicy
from splinter.exceptions import (
    BudgetExceededError,
    StepLimitExceededError,
    TimeLimitExceededError,
    LoopDetectedError,
    ToolAccessDeniedError,
    MemoryLimitError,
)


class TestLimitsEnforcer:
    """Tests for LimitsEnforcer."""

    def test_budget_limit(self):
        """Test budget limit enforcement."""
        enforcer = LimitsEnforcer(
            limits=ExecutionLimits(max_budget=1.0)
        )
        enforcer.start()

        # Should pass initially
        enforcer.check_limits()

        # Record cost up to limit
        enforcer.record_cost(0.5)
        enforcer.check_limits()

        enforcer.record_cost(0.5)
        with pytest.raises(BudgetExceededError):
            enforcer.check_limits()

    def test_step_limit(self):
        """Test step limit enforcement."""
        enforcer = LimitsEnforcer(
            limits=ExecutionLimits(max_steps=5)
        )
        enforcer.start()

        for _ in range(5):
            enforcer.check_limits()
            enforcer.increment_steps()

        with pytest.raises(StepLimitExceededError):
            enforcer.check_limits()

    def test_limit_update_mid_run(self):
        """Test that limit updates take effect on next check."""
        enforcer = LimitsEnforcer(
            limits=ExecutionLimits(max_steps=10)
        )
        enforcer.start()

        # Use 5 steps
        for _ in range(5):
            enforcer.increment_steps()

        # Update to lower limit
        enforcer.update_limits(ExecutionLimits(max_steps=4))

        with pytest.raises(StepLimitExceededError):
            enforcer.check_limits()

    def test_get_remaining(self):
        """Test remaining budget/steps calculation."""
        enforcer = LimitsEnforcer(
            limits=ExecutionLimits(max_budget=10.0, max_steps=100)
        )
        enforcer.start()

        enforcer.record_cost(3.0)
        enforcer.increment_steps(25)

        remaining = enforcer.get_remaining()
        assert remaining["budget"] == 7.0
        assert remaining["steps"] == 75


class TestLoopDetector:
    """Tests for LoopDetector."""

    def test_repeated_outputs(self):
        """Test detection of repeated identical outputs."""
        detector = LoopDetector(
            config=LoopDetectionConfig(max_repeated_outputs=3)
        )

        state = {"counter": 0}

        # Same output 3 times should trigger
        for i in range(2):
            detector.record_step(
                agent_id="test",
                input_data={"q": "test"},
                output_data="same output",
                state=state,
            )
            detector.check_for_loops()  # Should pass

        detector.record_step(
            agent_id="test",
            input_data={"q": "test"},
            output_data="same output",
            state=state,
        )

        with pytest.raises(LoopDetectedError):
            detector.check_for_loops()

    def test_no_state_change(self):
        """Test detection of no state change."""
        detector = LoopDetector(
            config=LoopDetectionConfig(max_no_state_change=3, max_repeated_outputs=10)
        )

        state = {"value": "constant"}

        for i in range(2):
            detector.record_step(
                agent_id="test",
                input_data={"q": f"query {i}"},
                output_data=f"output {i}",
                state=state,  # State never changes
            )
            detector.check_for_loops()

        detector.record_step(
            agent_id="test",
            input_data={"q": "query 3"},
            output_data="output 3",
            state=state,
        )

        with pytest.raises(LoopDetectedError):
            detector.check_for_loops()

    def test_reset(self):
        """Test that reset clears detection state."""
        detector = LoopDetector(
            config=LoopDetectionConfig(max_repeated_outputs=2)
        )

        state = {"v": 1}
        detector.record_step("a", {}, "out", state)
        detector.record_step("a", {}, "out", state)

        detector.reset()

        # Should not raise after reset
        detector.record_step("a", {}, "out", state)
        detector.check_for_loops()


class TestToolAccessController:
    """Tests for ToolAccessController."""

    def test_allowed_access(self):
        """Test that allowed agents can access tools."""
        controller = ToolAccessController()
        controller.register_tool("search", allowed_agents=["researcher"])

        # Should not raise
        controller.check_access("researcher", "search")

    def test_denied_access(self):
        """Test that unauthorized agents are blocked."""
        controller = ToolAccessController()
        controller.register_tool("search", allowed_agents=["researcher"])

        with pytest.raises(ToolAccessDeniedError):
            controller.check_access("other_agent", "search")

    def test_wildcard_access(self):
        """Test wildcard agent matching."""
        controller = ToolAccessController()
        controller.register_tool("read_file", allowed_agents=["*"])

        # All agents should have access
        controller.check_access("any_agent", "read_file")
        controller.check_access("another_agent", "read_file")

    def test_explicit_denial(self):
        """Test explicit denial overrides allow."""
        controller = ToolAccessController()
        controller.register_tool(
            "dangerous_tool",
            allowed_agents=["*"],
            denied_agents=["untrusted"]
        )

        controller.check_access("trusted", "dangerous_tool")

        with pytest.raises(ToolAccessDeniedError):
            controller.check_access("untrusted", "dangerous_tool")

    def test_get_allowed_tools(self):
        """Test getting list of allowed tools for an agent."""
        controller = ToolAccessController()
        controller.register_tool("tool_a", allowed_agents=["agent_1"])
        controller.register_tool("tool_b", allowed_agents=["agent_1", "agent_2"])
        controller.register_tool("tool_c", allowed_agents=["agent_2"])

        allowed = controller.get_allowed_tools("agent_1")
        assert "tool_a" in allowed
        assert "tool_b" in allowed
        assert "tool_c" not in allowed


class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_basic_operations(self):
        """Test basic get/set/delete."""
        store = MemoryStore()

        store.set("key1", {"data": "value1"})
        assert store.get("key1") == {"data": "value1"}

        store.delete("key1")
        assert store.get("key1") is None

    def test_size_limit(self):
        """Test size limit enforcement."""
        store = MemoryStore(
            config=MemoryConfig(max_size_bytes=100)
        )

        # Small values should work
        store.set("small", "x" * 10)
        assert store.exists("small")

        # Value exceeding total limit should raise
        with pytest.raises(MemoryLimitError):
            store.set("huge", "x" * 200)

    def test_entry_limit_eviction(self):
        """Test eviction when entry limit reached."""
        store = MemoryStore(
            config=MemoryConfig(max_entries=3, max_size_bytes=10000)
        )

        store.set("a", 1)
        store.set("b", 2)
        store.set("c", 3)

        assert store.entry_count == 3

        # Adding 4th should evict oldest
        store.set("d", 4)

        assert store.entry_count == 3
        assert not store.exists("a")  # First entry evicted
        assert store.exists("d")

    def test_agent_memory_namespace(self):
        """Test agent memory namespace isolation."""
        store = MemoryStore()

        agent1 = AgentMemory(store, "agent_1")
        agent2 = AgentMemory(store, "agent_2")

        agent1.set("key", "value1")
        agent2.set("key", "value2")

        assert agent1.get("key") == "value1"
        assert agent2.get("key") == "value2"

        # Clear only affects own namespace
        agent1.clear()
        assert agent1.get("key") is None
        assert agent2.get("key") == "value2"
