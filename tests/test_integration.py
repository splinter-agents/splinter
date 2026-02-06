"""Integration tests for Splinter using MockProvider.

These tests validate the full workflow without external API calls.
"""

import asyncio
import sys
import pytest

sys.path.insert(0, "/home/user/upgraded-enigma")

from splinter import (
    # Control Layer
    LimitsEnforcer,
    LoopDetector,
    ToolAccessController,
    MemoryStore,
    # Coordination Layer
    SharedState,
    StateOwnershipManager,
    HandoffManager,
    CheckpointManager,
    # Gateway
    Gateway,
    MockProvider,
    SmartMockProvider,
    # Workflow
    Workflow,
    Agent,
    AgentBuilder,
    # Types
    ExecutionLimits,
    LoopDetectionConfig,
    AgentConfig,
    LLMProvider,
    LLMMessage,
    AgentStatus,
    WorkflowStatus,
    # Exceptions
    BudgetExceededError,
    StepLimitExceededError,
)


class TestGatewayWithMock:
    """Test Gateway with MockProvider."""

    @pytest.mark.asyncio
    async def test_basic_completion(self):
        """Test basic completion with mock provider."""
        gateway = Gateway(
            limits=ExecutionLimits(max_budget=1.0, max_steps=10)
        )
        gateway.configure_provider("mock", default_response="Hello from mock!")

        response = await gateway.complete(
            agent_id="test",
            provider="mock",
            model="mock-model",
            messages=[LLMMessage(role="user", content="Hello")],
        )

        assert response.content == "Hello from mock!"
        assert response.cost > 0
        assert response.input_tokens > 0

    @pytest.mark.asyncio
    async def test_limit_enforcement(self):
        """Test that limits are enforced during gateway calls."""
        gateway = Gateway(
            limits=ExecutionLimits(max_steps=3)
        )
        gateway.configure_provider("mock")

        # Should succeed for first 3 calls
        for i in range(3):
            await gateway.complete(
                agent_id="test",
                provider="mock",
                model="mock-model",
                messages=[LLMMessage(role="user", content=f"Message {i}")],
            )

        # 4th call should fail
        with pytest.raises(StepLimitExceededError):
            await gateway.complete(
                agent_id="test",
                provider="mock",
                model="mock-model",
                messages=[LLMMessage(role="user", content="Message 4")],
            )

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        gateway = Gateway()
        gateway.configure_provider("mock", cost_per_call=0.01)

        for _ in range(5):
            await gateway.complete(
                agent_id="test",
                provider="mock",
                model="mock-model",
                messages=[LLMMessage(role="user", content="Test")],
            )

        metrics = gateway.get_metrics()
        assert metrics["total_steps"] == 5
        assert metrics["total_cost"] == pytest.approx(0.05, rel=0.01)

    @pytest.mark.asyncio
    async def test_call_history(self):
        """Test that call history is recorded."""
        gateway = Gateway()
        gateway.configure_provider("mock")

        await gateway.complete(
            agent_id="agent_1",
            provider="mock",
            model="mock-model",
            messages=[LLMMessage(role="user", content="Test 1")],
        )

        await gateway.complete(
            agent_id="agent_2",
            provider="mock",
            model="mock-model",
            messages=[LLMMessage(role="user", content="Test 2")],
        )

        history = gateway.get_call_history()
        assert len(history) == 2
        assert history[0].agent_id == "agent_1"
        assert history[1].agent_id == "agent_2"

        # Filter by agent
        agent1_history = gateway.get_call_history(agent_id="agent_1")
        assert len(agent1_history) == 1


class TestAgentWithMock:
    """Test Agent with MockProvider."""

    @pytest.mark.asyncio
    async def test_agent_run(self):
        """Test basic agent execution."""
        gateway = Gateway()
        gateway.configure_provider("mock", default_response='{"result": "success"}')

        agent = (
            AgentBuilder("test_agent")
            .with_system_prompt("You are a test agent.")
            .build(gateway)
        )

        # Manually set provider since AgentBuilder uses OPENAI by default
        agent._config.provider = LLMProvider.OPENAI
        agent._config.model = "mock-model"

        # We need to register mock under openai name for this test
        gateway._providers["openai"] = MockProvider(default_response='{"result": "success"}')

        result = await agent.run(task="Test task")

        assert agent.status == AgentStatus.COMPLETED
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_with_state(self):
        """Test agent with shared state."""
        gateway = Gateway()
        gateway._providers["openai"] = SmartMockProvider()

        agent = (
            AgentBuilder("researcher")
            .with_system_prompt("You are a research agent. Output JSON.")
            .build(gateway)
        )

        state = SharedState()
        state.set("topic", "Machine Learning")

        result = await agent.run(
            task="Research the topic",
            state=state,
        )

        assert agent.status == AgentStatus.COMPLETED


class TestWorkflowWithMock:
    """Test Workflow with MockProvider."""

    @pytest.mark.asyncio
    async def test_simple_workflow(self):
        """Test simple single-agent workflow."""
        # Create gateway with mock provider
        gateway = Gateway(
            limits=ExecutionLimits(max_budget=1.0, max_steps=20)
        )
        gateway._providers["openai"] = SmartMockProvider()

        workflow = Workflow(
            workflow_id="test-workflow",
            limits=ExecutionLimits(max_budget=1.0, max_steps=20),
        )
        workflow._gateway = gateway

        workflow.add_agent(AgentConfig(
            agent_id="analyzer",
            provider=LLMProvider.OPENAI,
            model="mock-model",
            system_prompt="You analyze topics. Output JSON with analysis.",
            state_ownership=["analyzer.*"],
        ))

        workflow.add_step("analyzer")

        result = await workflow.run(
            initial_state={"topic": "Test Topic"}
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.success
        assert "analyzer" in result.outputs

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self):
        """Test multi-agent workflow with handoffs."""
        gateway = Gateway(
            limits=ExecutionLimits(max_budget=1.0, max_steps=20)
        )
        gateway._providers["openai"] = SmartMockProvider()

        workflow = Workflow(
            workflow_id="multi-agent-test",
            limits=ExecutionLimits(max_budget=1.0, max_steps=20),
        )
        workflow._gateway = gateway

        # Add two agents
        workflow.add_agent(AgentConfig(
            agent_id="researcher",
            provider=LLMProvider.OPENAI,
            model="mock-model",
            system_prompt="You are a research agent. Output JSON with findings.",
            state_ownership=["researcher.*"],
        ))

        workflow.add_agent(AgentConfig(
            agent_id="summarizer",
            provider=LLMProvider.OPENAI,
            model="mock-model",
            system_prompt="You are a summary agent. Output JSON with summary.",
            state_ownership=["summarizer.*"],
        ))

        # Add steps with dependency
        workflow.add_step("researcher")
        workflow.add_step("summarizer", depends_on=["researcher"])

        result = await workflow.run(
            initial_state={"topic": "Integration Testing"}
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert "researcher" in result.outputs
        assert "summarizer" in result.outputs

    @pytest.mark.asyncio
    async def test_workflow_with_limits(self):
        """Test that workflow respects execution limits."""
        gateway = Gateway(
            limits=ExecutionLimits(max_steps=1)  # Very low limit
        )
        gateway._providers["openai"] = SmartMockProvider()

        workflow = Workflow(
            workflow_id="limit-test",
            limits=ExecutionLimits(max_steps=1),
        )
        workflow._gateway = gateway

        # Add multiple agents that would exceed limit
        for i in range(3):
            workflow.add_agent(AgentConfig(
                agent_id=f"agent_{i}",
                provider=LLMProvider.OPENAI,
                model="mock-model",
                system_prompt="Test agent",
            ))
            workflow.add_step(f"agent_{i}")

        # Should fail due to step limit
        with pytest.raises(Exception):  # WorkflowExecutionError wraps the limit error
            await workflow.run()

    @pytest.mark.asyncio
    async def test_workflow_state_management(self):
        """Test that workflow manages state correctly."""
        gateway = Gateway()
        gateway._providers["openai"] = SmartMockProvider()

        workflow = Workflow(workflow_id="state-test")
        workflow._gateway = gateway

        workflow.add_agent(AgentConfig(
            agent_id="writer",
            provider=LLMProvider.OPENAI,
            model="mock-model",
            system_prompt="Output JSON with data.",
            state_ownership=["writer.*"],
        ))

        workflow.add_step("writer")

        result = await workflow.run(
            initial_state={"input": "test data"}
        )

        # Check that initial state is preserved
        assert "input" in result.state
        assert result.state["input"] == "test data"

        # Check that agent wrote to state
        assert "writer" in result.state or any("writer" in k for k in result.state.keys())


class TestEndToEndIntegration:
    """Full end-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """Test a complete multi-agent pipeline."""
        # Set up gateway
        gateway = Gateway(
            limits=ExecutionLimits(max_budget=5.0, max_steps=50)
        )
        gateway._providers["openai"] = SmartMockProvider()

        # Set up workflow
        workflow = Workflow(
            workflow_id="complete-pipeline",
            limits=ExecutionLimits(max_budget=5.0, max_steps=50),
            checkpoint_enabled=True,
        )
        workflow._gateway = gateway

        # Add agents
        workflow.add_agent(AgentConfig(
            agent_id="research",
            provider=LLMProvider.OPENAI,
            model="mock-model",
            system_prompt="Research agent. Output JSON with findings and themes.",
            state_ownership=["research.*"],
        ))

        workflow.add_agent(AgentConfig(
            agent_id="analysis",
            provider=LLMProvider.OPENAI,
            model="mock-model",
            system_prompt="Analysis agent. Output JSON with analysis and points.",
            state_ownership=["analysis.*"],
        ))

        workflow.add_agent(AgentConfig(
            agent_id="summary",
            provider=LLMProvider.OPENAI,
            model="mock-model",
            system_prompt="Summary agent. Output JSON with executive_summary.",
            state_ownership=["summary.*"],
        ))

        # Define pipeline
        workflow.add_step("research")
        workflow.add_step("analysis", depends_on=["research"])
        workflow.add_step("summary", depends_on=["analysis"])

        # Add state ownership
        workflow.add_state_ownership("research", ["research.*"])
        workflow.add_state_ownership("analysis", ["analysis.*"])
        workflow.add_state_ownership("summary", ["summary.*"])

        # Run workflow
        result = await workflow.run(
            initial_state={
                "topic": "Artificial Intelligence",
                "depth": "comprehensive",
            }
        )

        # Verify results
        assert result.success
        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.outputs) == 3
        assert "research" in result.outputs
        assert "analysis" in result.outputs
        assert "summary" in result.outputs

        # Verify metrics
        assert result.metrics["total_steps"] == 3
        assert result.metrics["total_cost"] > 0

        print(f"\nPipeline completed successfully!")
        print(f"  Status: {result.status}")
        print(f"  Total Cost: ${result.metrics['total_cost']:.4f}")
        print(f"  Total Steps: {result.metrics['total_steps']}")
        print(f"  Agents executed: {list(result.outputs.keys())}")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("SPLINTER INTEGRATION TESTS (with MockProvider)")
    print("=" * 60)

    # Run pytest programmatically
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])

    return exit_code == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
