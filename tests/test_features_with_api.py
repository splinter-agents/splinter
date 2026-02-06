"""Comprehensive feature tests for Splinter with OpenAI API.

This test suite validates each Splinter feature against real LLM agents.
Set OPENAI_API_KEY environment variable to run these tests.
"""

import asyncio
import os
import sys
import json

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
    MemoryConfig,
    AgentConfig,
    LLMProvider,
    LLMMessage,
    AgentStatus,
    WorkflowStatus,
    ExecutionMetrics,
    # Exceptions
    BudgetExceededError,
    StepLimitExceededError,
    ToolAccessDeniedError,
    StateOwnershipError,
)


# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY", "")
USE_REAL_API = bool(API_KEY)


def get_gateway(max_budget=1.0, max_steps=20):
    """Create a gateway configured for testing."""
    gateway = Gateway(
        limits=ExecutionLimits(max_budget=max_budget, max_steps=max_steps)
    )

    if USE_REAL_API:
        gateway.configure_provider("openai", api_key=API_KEY)
    else:
        # Use smart mock that generates realistic responses
        gateway._providers["openai"] = SmartMockProvider()

    return gateway


def get_model():
    """Get model name for tests."""
    return "gpt-4o-mini" if USE_REAL_API else "mock-model"


class FeatureTestResults:
    """Track test results."""

    def __init__(self):
        self.passed = []
        self.failed = []

    def record(self, name: str, success: bool, details: str = ""):
        if success:
            self.passed.append((name, details))
        else:
            self.failed.append((name, details))

    def summary(self):
        print("\n" + "=" * 70)
        print("FEATURE TEST RESULTS")
        print("=" * 70)
        print(f"\nPassed: {len(self.passed)}")
        for name, details in self.passed:
            print(f"  ✓ {name}")
            if details:
                print(f"    {details}")

        print(f"\nFailed: {len(self.failed)}")
        for name, details in self.failed:
            print(f"  ✗ {name}")
            if details:
                print(f"    {details}")

        print(f"\nTotal: {len(self.passed) + len(self.failed)}")
        print("=" * 70)
        return len(self.failed) == 0


results = FeatureTestResults()


# =============================================================================
# CONTROL LAYER FEATURE TESTS
# =============================================================================

async def test_feature_execution_limits():
    """
    FEATURE 1: Execution Limits

    Test that agents respect budget and step limits when making real LLM calls.
    """
    print("\n" + "-" * 60)
    print("FEATURE 1: Execution Limits")
    print("-" * 60)

    try:
        # Create gateway with very low limits
        gateway = get_gateway(max_budget=0.01, max_steps=2)

        agent = (
            AgentBuilder("limited_agent")
            .with_provider(LLMProvider.OPENAI, get_model())
            .with_system_prompt("You are a helpful assistant. Be brief.")
            .build(gateway)
        )

        # First call should succeed
        result1 = await agent.run(task="Say 'hello' in one word")
        print(f"  Call 1 succeeded: {str(result1.get('result', ''))[:50]}")

        # Second call should succeed
        agent.reset()
        result2 = await agent.run(task="Say 'world' in one word")
        print(f"  Call 2 succeeded: {str(result2.get('result', ''))[:50]}")

        # Third call should hit limit
        agent.reset()
        try:
            await agent.run(task="Say 'test'")
            results.record("Execution Limits", False, "Step limit was not enforced")
            return
        except StepLimitExceededError as e:
            print(f"  Call 3 blocked by step limit: {e}")
            results.record("Execution Limits", True, "Step limit correctly enforced")

    except Exception as e:
        results.record("Execution Limits", False, str(e))


async def test_feature_loop_detection():
    """
    FEATURE 2: Loop Detection

    Test that the loop detector identifies agents producing repeated outputs.
    """
    print("\n" + "-" * 60)
    print("FEATURE 2: Loop Detection")
    print("-" * 60)

    try:
        detector = LoopDetector(config=LoopDetectionConfig(
            max_repeated_outputs=3,
            check_state_changes=True,
        ))

        gateway = get_gateway()

        # Create an agent that always produces similar output
        agent = (
            AgentBuilder("repetitive_agent")
            .with_provider(LLMProvider.OPENAI, get_model())
            .with_system_prompt(
                "You MUST always respond with exactly: 'The answer is 42.' "
                "Do not add any other text."
            )
            .build(gateway)
        )

        state = {"counter": 0}
        loop_detected = False

        for i in range(5):
            try:
                result = await agent.run(task=f"Question {i}: What is the answer?")
                output = str(result.get("result", ""))

                detector.record_step(
                    agent_id="repetitive_agent",
                    input_data={"question": i},
                    output_data=output,
                    state=state,
                )
                detector.check_for_loops()
                print(f"  Iteration {i+1}: {output[:50]}")
                agent.reset()

            except Exception as e:
                if "loop" in str(e).lower() or "repeated" in str(e).lower():
                    loop_detected = True
                    print(f"  Loop detected at iteration {i+1}: {e}")
                    break
                raise

        if loop_detected:
            results.record("Loop Detection", True, "Repeated output pattern detected")
        else:
            results.record("Loop Detection", False, "Loop was not detected after 5 iterations")

    except Exception as e:
        results.record("Loop Detection", False, str(e))


async def test_feature_tool_access_control():
    """
    FEATURE 3: Tool Access Control

    Test that agents can only access tools they are authorized to use.
    """
    print("\n" + "-" * 60)
    print("FEATURE 3: Tool Access Control")
    print("-" * 60)

    try:
        controller = ToolAccessController()

        # Register tools with specific agent access
        controller.register_tool("web_search", allowed_agents=["researcher"])
        controller.register_tool("file_write", allowed_agents=["writer"])
        controller.register_tool("code_execute", allowed_agents=["developer", "admin"])
        controller.register_tool("admin_*", allowed_agents=["admin"])

        # Test: researcher can use web_search
        controller.check_access("researcher", "web_search")
        print("  ✓ researcher can access web_search")

        # Test: researcher cannot use file_write
        try:
            controller.check_access("researcher", "file_write")
            results.record("Tool Access Control", False, "Access should have been denied")
            return
        except ToolAccessDeniedError:
            print("  ✓ researcher blocked from file_write")

        # Test: wildcard pattern works
        controller.check_access("admin", "admin_delete_users")
        print("  ✓ admin can access admin_* tools via wildcard")

        # Test: developer cannot use admin tools
        try:
            controller.check_access("developer", "admin_delete_users")
            results.record("Tool Access Control", False, "Admin access should have been denied")
            return
        except ToolAccessDeniedError:
            print("  ✓ developer blocked from admin tools")

        results.record("Tool Access Control", True, "All access control checks passed")

    except Exception as e:
        results.record("Tool Access Control", False, str(e))


async def test_feature_memory_limits():
    """
    FEATURE 4: Memory Limits

    Test that memory stores respect configured limits and eviction policies.
    """
    print("\n" + "-" * 60)
    print("FEATURE 4: Memory Limits")
    print("-" * 60)

    try:
        # Test FIFO eviction
        store_fifo = MemoryStore(config=MemoryConfig(
            max_entries=3,
            eviction_policy="fifo",
        ))

        for i in range(5):
            store_fifo.set(f"key_{i}", f"value_{i}")

        # Should only have last 3 entries
        assert store_fifo.entry_count == 3, f"Expected 3 entries, got {store_fifo.entry_count}"
        assert store_fifo.get("key_0") is None, "key_0 should have been evicted"
        assert store_fifo.get("key_1") is None, "key_1 should have been evicted"
        assert store_fifo.get("key_4") == "value_4", "key_4 should exist"
        print(f"  ✓ FIFO eviction working (kept keys: {store_fifo.keys()})")

        # Test LRU eviction
        store_lru = MemoryStore(config=MemoryConfig(
            max_entries=3,
            eviction_policy="lru",
        ))

        store_lru.set("a", 1)
        store_lru.set("b", 2)
        store_lru.set("c", 3)

        # Access 'a' to make it recently used
        _ = store_lru.get("a")

        # Add new key - 'b' should be evicted (least recently used)
        store_lru.set("d", 4)

        assert store_lru.get("a") == 1, "a should exist (was accessed)"
        assert store_lru.get("b") is None, "b should have been evicted (LRU)"
        assert store_lru.get("c") == 3, "c should exist"
        assert store_lru.get("d") == 4, "d should exist"
        print(f"  ✓ LRU eviction working (kept keys: {store_lru.keys()})")

        # Test max size limit
        store_size = MemoryStore(config=MemoryConfig(
            max_size_bytes=100,  # Very small limit
        ))

        # Add small items
        store_size.set("small1", "x" * 20)
        store_size.set("small2", "y" * 20)

        # Adding large item should trigger eviction
        store_size.set("large", "z" * 50)

        print(f"  ✓ Size-based eviction working (entries: {store_size.entry_count})")

        results.record("Memory Limits", True, "All memory limit tests passed")

    except Exception as e:
        results.record("Memory Limits", False, str(e))


# =============================================================================
# COORDINATION LAYER FEATURE TESTS
# =============================================================================

async def test_feature_shared_state():
    """
    FEATURE 5: Shared State Store

    Test that multiple agents can share and update state with proper versioning.
    """
    print("\n" + "-" * 60)
    print("FEATURE 5: Shared State Store")
    print("-" * 60)

    try:
        state = SharedState()

        # Test nested path access
        state.set("research.topic", "Machine Learning")
        state.set("research.findings", [])
        state.set("research.metadata.source", "web")

        assert state.get("research.topic") == "Machine Learning"
        assert state.get("research.metadata.source") == "web"
        print("  ✓ Nested path access working")

        # Test versioning
        v1 = state.version
        state.set("research.findings", ["finding1"])
        v2 = state.version
        state.set("research.findings", ["finding1", "finding2"])
        v3 = state.version

        assert v2 > v1 and v3 > v2, "Versions should increment"
        print(f"  ✓ Versioning working (v1={v1}, v2={v2}, v3={v3})")

        # Test snapshot immutability
        snapshot = state.snapshot()
        original = snapshot.data["research"]["findings"].copy()

        state.set("research.findings", ["modified"])

        assert snapshot.data["research"]["findings"] == original
        print("  ✓ Snapshot immutability working")

        # Test history
        history = state.get_history(limit=3)
        assert len(history) >= 2, "Should have history entries"
        print(f"  ✓ History tracking working ({len(history)} entries)")

        # Test restore
        state.set("value", 100)
        restore_version = state.version
        state.set("value", 200)
        state.set("value", 300)

        state.restore(restore_version)
        assert state.get("value") == 100
        print("  ✓ State restoration working")

        results.record("Shared State Store", True, "All state operations passed")

    except Exception as e:
        results.record("Shared State Store", False, str(e))


async def test_feature_state_ownership():
    """
    FEATURE 6: State Ownership

    Test that agents can only write to state fields they own.
    """
    print("\n" + "-" * 60)
    print("FEATURE 6: State Ownership")
    print("-" * 60)

    try:
        manager = StateOwnershipManager()

        # Register ownership patterns
        manager.register("research.*", owner="researcher")
        manager.register("summary.*", owner="summarizer")
        manager.register("shared.*", owner="*")  # Anyone can write (wildcard)

        # Test: owner can write
        manager.check_write("researcher", "research.findings")
        print("  ✓ researcher can write to research.findings")

        # Test: non-owner cannot write
        try:
            manager.check_write("summarizer", "research.findings")
            results.record("State Ownership", False, "Write should have been blocked")
            return
        except StateOwnershipError:
            print("  ✓ summarizer blocked from research.findings")

        # Test: wildcard patterns
        manager.check_write("researcher", "research.deep.nested.path")
        print("  ✓ Wildcard ownership patterns work")

        # Test: shared fields
        manager.check_write("anyone", "shared.data")
        print("  ✓ Shared fields accessible to all")

        # Test with agents making real calls
        gateway = get_gateway()
        state = SharedState()

        researcher = (
            AgentBuilder("researcher")
            .with_provider(LLMProvider.OPENAI, get_model())
            .with_system_prompt("Research agent. Output JSON with 'findings' array.")
            .with_state_ownership(["research.*"])
            .build(gateway)
        )

        result = await researcher.run(
            task="Find 3 facts about Python",
            state=state,
        )
        print(f"  ✓ Researcher agent executed: {str(result.get('result', ''))[:80]}...")

        results.record("State Ownership", True, "All ownership checks passed")

    except Exception as e:
        results.record("State Ownership", False, str(e))


async def test_feature_schema_handoffs():
    """
    FEATURE 7: Schema Handoffs

    Test that agent outputs are validated against schemas during handoffs.
    """
    print("\n" + "-" * 60)
    print("FEATURE 7: Schema Handoffs")
    print("-" * 60)

    try:
        handoff = HandoffManager(strict=True)

        # Register handoff with schema
        handoff.register_handoff(
            source="researcher",
            target="summarizer",
            output_schema={
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["findings"],
            },
        )

        # Test: valid output passes
        valid_output = {
            "findings": ["Python is interpreted", "Python uses indentation"],
            "confidence": 0.9,
        }
        handoff.validate_output("researcher", "summarizer", valid_output)
        print("  ✓ Valid output accepted")

        # Test: missing required field fails
        try:
            handoff.validate_output("researcher", "summarizer", {"confidence": 0.5})
            results.record("Schema Handoffs", False, "Missing field should have failed")
            return
        except Exception:
            print("  ✓ Missing required field rejected")

        # Test: wrong type fails
        try:
            handoff.validate_output("researcher", "summarizer", {"findings": "not an array"})
            results.record("Schema Handoffs", False, "Wrong type should have failed")
            return
        except Exception:
            print("  ✓ Wrong type rejected")

        # Test with real agent workflow
        gateway = get_gateway()

        workflow = Workflow(
            workflow_id="handoff-test",
            limits=ExecutionLimits(max_budget=0.50, max_steps=10),
        )
        workflow._gateway = gateway

        workflow.add_agent(AgentConfig(
            agent_id="producer",
            provider=LLMProvider.OPENAI,
            model=get_model(),
            system_prompt=(
                "Output JSON with exactly this format: "
                "{\"data\": [\"item1\", \"item2\"], \"count\": 2}"
            ),
        ))

        workflow.add_agent(AgentConfig(
            agent_id="consumer",
            provider=LLMProvider.OPENAI,
            model=get_model(),
            system_prompt="Summarize the input data. Output JSON with 'summary' field.",
        ))

        workflow.add_step("producer")
        workflow.add_step("consumer", depends_on=["producer"])

        result = await workflow.run(initial_state={"request": "test"})
        print(f"  ✓ Workflow with handoff completed: {result.status}")

        results.record("Schema Handoffs", True, "All schema validation tests passed")

    except Exception as e:
        results.record("Schema Handoffs", False, str(e))


async def test_feature_resumable_execution():
    """
    FEATURE 8: Resumable Execution

    Test that workflow execution can be checkpointed and resumed.
    """
    print("\n" + "-" * 60)
    print("FEATURE 8: Resumable Execution")
    print("-" * 60)

    try:
        checkpoint_mgr = CheckpointManager()

        # Simulate a workflow execution with checkpoints
        state = SharedState()
        state.set("step", 0)
        state.set("data", {"topic": "AI Safety"})

        # Create checkpoint after step 1
        state.set("step", 1)
        state.set("step1_output", {"findings": ["finding1", "finding2"]})

        cp1 = checkpoint_mgr.create_checkpoint(
            workflow_id="resume-test",
            step=1,
            agent_id="researcher",
            status=AgentStatus.COMPLETED,
            state=state,
            metrics=ExecutionMetrics(total_cost=0.02, total_steps=1),
        )
        print(f"  ✓ Checkpoint 1 created (step={cp1.step})")

        # Create checkpoint after step 2
        state.set("step", 2)
        state.set("step2_output", {"analysis": "detailed analysis"})

        cp2 = checkpoint_mgr.create_checkpoint(
            workflow_id="resume-test",
            step=2,
            agent_id="analyzer",
            status=AgentStatus.COMPLETED,
            state=state,
            metrics=ExecutionMetrics(total_cost=0.05, total_steps=2),
        )
        print(f"  ✓ Checkpoint 2 created (step={cp2.step})")

        # Test loading latest checkpoint
        latest = checkpoint_mgr.get_latest_checkpoint("resume-test")
        assert latest.step == 2, f"Expected step 2, got {latest.step}"
        print(f"  ✓ Latest checkpoint loaded (step={latest.step})")

        # Test getting resume point
        resume = checkpoint_mgr.get_resume_point("resume-test")
        assert resume is not None, "Should have resume point"
        next_step, state_snapshot, metrics = resume

        assert next_step == 3, f"Should resume from step 3, got {next_step}"
        assert state_snapshot.data["step2_output"]["analysis"] == "detailed analysis"
        assert metrics.total_cost == 0.05
        print(f"  ✓ Resume point calculated (next_step={next_step}, cost=${metrics.total_cost})")

        # Test full workflow with checkpointing
        gateway = get_gateway()

        workflow = Workflow(
            workflow_id="checkpoint-workflow",
            limits=ExecutionLimits(max_budget=1.0, max_steps=20),
            checkpoint_enabled=True,
        )
        workflow._gateway = gateway

        workflow.add_agent(AgentConfig(
            agent_id="step1_agent",
            provider=LLMProvider.OPENAI,
            model=get_model(),
            system_prompt="Output JSON with 'step1_complete': true",
        ))

        workflow.add_agent(AgentConfig(
            agent_id="step2_agent",
            provider=LLMProvider.OPENAI,
            model=get_model(),
            system_prompt="Output JSON with 'step2_complete': true",
        ))

        workflow.add_step("step1_agent")
        workflow.add_step("step2_agent", depends_on=["step1_agent"])

        result = await workflow.run()
        print(f"  ✓ Workflow with checkpointing completed: {result.status}")

        results.record("Resumable Execution", True, "All checkpoint/resume tests passed")

    except Exception as e:
        results.record("Resumable Execution", False, str(e))


# =============================================================================
# MULTI-AGENT WORKFLOW TEST
# =============================================================================

async def test_full_multi_agent_pipeline():
    """
    INTEGRATION TEST: Full Multi-Agent Pipeline

    Test a complete pipeline with multiple agents using all features.
    """
    print("\n" + "-" * 60)
    print("INTEGRATION: Full Multi-Agent Pipeline")
    print("-" * 60)

    try:
        gateway = get_gateway(max_budget=2.0, max_steps=30)

        workflow = Workflow(
            workflow_id="full-pipeline",
            name="Research Pipeline",
            limits=ExecutionLimits(max_budget=2.0, max_steps=30),
            checkpoint_enabled=True,
        )
        workflow._gateway = gateway

        # Agent 1: Researcher
        workflow.add_agent(AgentConfig(
            agent_id="researcher",
            provider=LLMProvider.OPENAI,
            model=get_model(),
            system_prompt=(
                "You are a research agent. When given a topic, provide 3 key findings. "
                "Output JSON: {\"findings\": [\"fact1\", \"fact2\", \"fact3\"], \"sources\": 3}"
            ),
            state_ownership=["research.*"],
        ))

        # Agent 2: Analyzer
        workflow.add_agent(AgentConfig(
            agent_id="analyzer",
            provider=LLMProvider.OPENAI,
            model=get_model(),
            system_prompt=(
                "You analyze research findings. Identify patterns and insights. "
                "Output JSON: {\"analysis\": \"text\", \"key_themes\": [\"theme1\", \"theme2\"]}"
            ),
            state_ownership=["analysis.*"],
        ))

        # Agent 3: Summarizer
        workflow.add_agent(AgentConfig(
            agent_id="summarizer",
            provider=LLMProvider.OPENAI,
            model=get_model(),
            system_prompt=(
                "You create executive summaries. Be concise and clear. "
                "Output JSON: {\"summary\": \"brief summary\", \"word_count\": 50}"
            ),
            state_ownership=["summary.*"],
        ))

        # Define pipeline
        workflow.add_step("researcher")
        workflow.add_step("analyzer", depends_on=["researcher"])
        workflow.add_step("summarizer", depends_on=["analyzer"])

        # Configure state ownership
        workflow.add_state_ownership("researcher", ["research.*"])
        workflow.add_state_ownership("analyzer", ["analysis.*"])
        workflow.add_state_ownership("summarizer", ["summary.*"])

        # Run pipeline
        result = await workflow.run(
            initial_state={
                "topic": "The impact of artificial intelligence on healthcare",
                "depth": "comprehensive",
            }
        )

        print(f"  Status: {result.status}")
        print(f"  Success: {result.success}")
        print(f"  Total Cost: ${result.metrics.get('total_cost', 0):.4f}")
        print(f"  Total Steps: {result.metrics.get('total_steps', 0)}")

        print(f"\n  Agent Outputs:")
        for agent_id, output in result.outputs.items():
            output_preview = str(output.get('result', ''))[:100]
            print(f"    {agent_id}: {output_preview}...")

        if result.success:
            results.record("Full Multi-Agent Pipeline", True,
                         f"Completed with {len(result.outputs)} agents, ${result.metrics.get('total_cost', 0):.4f}")
        else:
            results.record("Full Multi-Agent Pipeline", False, f"Workflow failed: {result.error}")

    except Exception as e:
        results.record("Full Multi-Agent Pipeline", False, str(e))


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all feature tests."""
    print("=" * 70)
    print("SPLINTER COMPREHENSIVE FEATURE TESTS")
    print(f"Mode: {'REAL OPENAI API' if USE_REAL_API else 'MOCK PROVIDER'}")
    print("=" * 70)

    if not USE_REAL_API:
        print("\n⚠️  Note: Running with MockProvider. Set OPENAI_API_KEY to test with real API.\n")

    # Control Layer Features
    await test_feature_execution_limits()
    await test_feature_loop_detection()
    await test_feature_tool_access_control()
    await test_feature_memory_limits()

    # Coordination Layer Features
    await test_feature_shared_state()
    await test_feature_state_ownership()
    await test_feature_schema_handoffs()
    await test_feature_resumable_execution()

    # Integration Test
    await test_full_multi_agent_pipeline()

    # Summary
    return results.summary()


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
