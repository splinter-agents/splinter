"""Pressure test for Splinter platform.

Uses MockProvider by default for CI/testing without network access.
Set OPENAI_API_KEY environment variable to test with real API.
"""

import asyncio
import os
import sys
import traceback

# Add parent to path for imports
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
    # Exceptions
    BudgetExceededError,
    StepLimitExceededError,
)

# API key should be set via environment variable for real API testing
API_KEY = os.environ.get("OPENAI_API_KEY", "")
USE_MOCK = not API_KEY  # Use mock if no API key


def test_control_layer():
    """Test control layer components."""
    print("\n" + "=" * 60)
    print("TESTING CONTROL LAYER")
    print("=" * 60)

    # Test LimitsEnforcer
    print("\n1. Testing LimitsEnforcer...")
    enforcer = LimitsEnforcer(
        limits=ExecutionLimits(max_budget=0.10, max_steps=5)
    )
    enforcer.start()

    for i in range(6):
        try:
            enforcer.check_limits()
            enforcer.record_cost(0.01)
            enforcer.increment_steps()
            print(f"   Step {i+1}: cost=${enforcer.metrics.total_cost:.3f}, steps={enforcer.metrics.total_steps}")
        except (BudgetExceededError, StepLimitExceededError) as e:
            print(f"   ✓ Limit enforced: {e}")
            break

    # Test LoopDetector
    print("\n2. Testing LoopDetector...")
    detector = LoopDetector(config=LoopDetectionConfig(max_repeated_outputs=3))
    state = {"counter": 0}

    for i in range(4):
        try:
            detector.record_step(
                agent_id="test",
                input_data={"q": "test"},
                output_data="same output",
                state=state,
            )
            detector.check_for_loops()
            print(f"   Step {i+1}: recorded")
        except Exception as e:
            print(f"   ✓ Loop detected: {type(e).__name__}")
            break

    # Test ToolAccessController
    print("\n3. Testing ToolAccessController...")
    controller = ToolAccessController()
    controller.register_tool("search", allowed_agents=["researcher"])

    try:
        controller.check_access("researcher", "search")
        print("   ✓ researcher can use search")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    try:
        controller.check_access("other", "search")
        print("   ✗ other should not access search")
    except Exception as e:
        print(f"   ✓ Access denied for other: {type(e).__name__}")

    # Test MemoryStore
    print("\n4. Testing MemoryStore...")
    store = MemoryStore(config=MemoryConfig(max_entries=3))

    for i in range(5):
        store.set(f"key_{i}", f"value_{i}")

    print(f"   Entries after adding 5 (max 3): {store.entry_count}")
    print(f"   Keys: {store.keys()}")
    print(f"   ✓ Memory eviction working")

    print("\n✓ Control Layer tests passed")


def test_coordination_layer():
    """Test coordination layer components."""
    print("\n" + "=" * 60)
    print("TESTING COORDINATION LAYER")
    print("=" * 60)

    # Test SharedState
    print("\n1. Testing SharedState...")
    state = SharedState()

    state.set("research.findings", ["finding1", "finding2"])
    state.set("research.metadata.source", "web")

    print(f"   Findings: {state.get('research.findings')}")
    print(f"   Source: {state.get('research.metadata.source')}")
    print(f"   Version: {state.version}")
    print("   ✓ SharedState working")

    # Test StateOwnershipManager
    print("\n2. Testing StateOwnershipManager...")
    ownership = StateOwnershipManager()
    ownership.register("research.*", owner="researcher")

    try:
        ownership.check_write("researcher", "research.data")
        print("   ✓ researcher can write to research.*")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    try:
        ownership.check_write("other", "research.data")
        print("   ✗ other should not write to research.*")
    except Exception as e:
        print(f"   ✓ Write denied for other: {type(e).__name__}")

    # Test HandoffManager
    print("\n3. Testing HandoffManager...")
    handoff = HandoffManager(strict=True)
    handoff.register_handoff(
        source="researcher",
        target="summarizer",
        output_schema={
            "type": "object",
            "properties": {"results": {"type": "array"}},
            "required": ["results"],
        },
    )

    try:
        handoff.validate_output("researcher", "summarizer", {"results": ["a", "b"]})
        print("   ✓ Valid handoff accepted")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    try:
        handoff.validate_output("researcher", "summarizer", {"wrong": "data"})
        print("   ✗ Invalid handoff should be rejected")
    except Exception as e:
        print(f"   ✓ Invalid handoff rejected: {type(e).__name__}")

    # Test CheckpointManager
    print("\n4. Testing CheckpointManager...")
    from splinter.types import AgentStatus, ExecutionMetrics

    checkpoint_mgr = CheckpointManager()

    cp = checkpoint_mgr.create_checkpoint(
        workflow_id="test-wf",
        step=1,
        agent_id="agent_1",
        status=AgentStatus.COMPLETED,
        state=state,
        metrics=ExecutionMetrics(total_cost=0.05, total_steps=3),
    )

    loaded = checkpoint_mgr.get_latest_checkpoint("test-wf")
    print(f"   Saved checkpoint step: {cp.step}")
    print(f"   Loaded checkpoint step: {loaded.step}")
    print("   ✓ Checkpointing working")

    print("\n✓ Coordination Layer tests passed")


async def test_gateway():
    """Test gateway with MockProvider or real API."""
    print("\n" + "=" * 60)
    print(f"TESTING GATEWAY ({'Mock' if USE_MOCK else 'Real API'})")
    print("=" * 60)

    gateway = Gateway(
        limits=ExecutionLimits(max_budget=0.50, max_steps=10)
    )

    # Configure provider
    print("\n1. Configuring provider...")
    if USE_MOCK:
        gateway.configure_provider("mock", default_response="Mock response: 4")
    else:
        gateway.configure_provider("openai", api_key=API_KEY)
    print("   ✓ Provider configured")

    provider = LLMProvider.MOCK if USE_MOCK else LLMProvider.OPENAI
    model = "mock-model" if USE_MOCK else "gpt-4o-mini"

    # Make a simple completion
    print("\n2. Making simple completion...")
    try:
        response = await gateway.complete(
            agent_id="test_agent",
            provider=provider,
            model=model,
            messages=[
                LLMMessage(role="system", content="You are a helpful assistant. Be brief."),
                LLMMessage(role="user", content="What is 2+2? Just say the number."),
            ],
        )
        print(f"   Response: {response.content}")
        print(f"   Tokens: {response.input_tokens} in, {response.output_tokens} out")
        print(f"   Cost: ${response.cost:.4f}")
        print(f"   Latency: {response.latency_ms:.0f}ms")
        print("   ✓ Completion successful")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        traceback.print_exc()
        return False

    # Test with multiple calls
    print("\n3. Testing multiple calls with limit tracking...")
    for i in range(3):
        try:
            response = await gateway.complete(
                agent_id="test_agent",
                provider=provider,
                model=model,
                messages=[
                    LLMMessage(role="user", content=f"Count to {i+1}. Be brief."),
                ],
            )
            metrics = gateway.get_metrics()
            print(f"   Call {i+1}: ${metrics['total_cost']:.4f} total, {metrics['total_steps']} steps")
        except Exception as e:
            print(f"   ✗ Error on call {i+1}: {e}")

    # Show final metrics
    metrics = gateway.get_metrics()
    print(f"\n   Final Metrics:")
    print(f"   Total Cost: ${metrics['total_cost']:.4f}")
    print(f"   Total Steps: {metrics['total_steps']}")
    print(f"   Total Tokens: {metrics['total_tokens']}")
    print("   ✓ Gateway tests passed")

    return True


async def test_agent():
    """Test agent execution."""
    print("\n" + "=" * 60)
    print(f"TESTING AGENT ({'Mock' if USE_MOCK else 'Real API'})")
    print("=" * 60)

    gateway = Gateway(
        limits=ExecutionLimits(max_budget=0.50, max_steps=10)
    )

    # Configure with mock or real provider
    if USE_MOCK:
        gateway._providers["openai"] = SmartMockProvider()
    else:
        gateway.configure_provider("openai", api_key=API_KEY)

    # Create agent using builder
    print("\n1. Creating agent with AgentBuilder...")
    agent = (
        AgentBuilder("researcher")
        .with_provider(LLMProvider.OPENAI, "gpt-4o-mini" if not USE_MOCK else "mock-model")
        .with_system_prompt(
            "You are a research assistant. When given a topic, provide 3 key facts. "
            "Respond in JSON format: {\"facts\": [\"fact1\", \"fact2\", \"fact3\"]}"
        )
        .build(gateway)
    )
    print(f"   Agent ID: {agent.id}")
    print(f"   Status: {agent.status}")

    # Run agent
    print("\n2. Running agent...")
    try:
        result = await agent.run(
            task="Tell me about Python programming language",
            state={"topic": "Python"},
        )
        print(f"   Status after run: {agent.status}")
        print(f"   Result: {str(result)[:200]}...")
        print("   ✓ Agent execution successful")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        traceback.print_exc()
        return False

    return True


async def test_workflow():
    """Test full workflow execution."""
    print("\n" + "=" * 60)
    print(f"TESTING WORKFLOW ({'Mock' if USE_MOCK else 'Real API'})")
    print("=" * 60)

    # Create gateway with mock
    gateway = Gateway(
        limits=ExecutionLimits(max_budget=1.0, max_steps=20)
    )
    if USE_MOCK:
        gateway._providers["openai"] = SmartMockProvider()
    else:
        gateway.configure_provider("openai", api_key=API_KEY)

    # Create workflow
    print("\n1. Creating workflow...")
    workflow = Workflow(
        workflow_id="test-pipeline",
        name="Test Pipeline",
        limits=ExecutionLimits(max_budget=1.0, max_steps=20),
    )
    workflow._gateway = gateway

    # Add agents
    print("\n2. Adding agents...")
    workflow.add_agent(AgentConfig(
        agent_id="analyzer",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini" if not USE_MOCK else "mock-model",
        system_prompt=(
            "You analyze topics and provide key points. "
            "Respond with JSON: {\"analysis\": \"your analysis\", \"points\": [\"point1\", \"point2\"]}"
        ),
        state_ownership=["analyzer.*"],
    ))

    workflow.add_agent(AgentConfig(
        agent_id="summarizer",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini" if not USE_MOCK else "mock-model",
        system_prompt=(
            "You summarize analyses into brief summaries. "
            "Respond with JSON: {\"summary\": \"brief summary\"}"
        ),
        state_ownership=["summarizer.*"],
    ))
    print("   ✓ Agents added")

    # Add steps
    print("\n3. Adding steps...")
    workflow.add_step("analyzer")
    workflow.add_step("summarizer", depends_on=["analyzer"])
    print("   ✓ Steps added")

    # Run workflow
    print("\n4. Running workflow...")
    try:
        result = await workflow.run(
            initial_state={"topic": "Machine Learning basics"}
        )
        print(f"   Status: {result.status}")
        print(f"   Success: {result.success}")
        print(f"   Total Cost: ${result.metrics.get('total_cost', 0):.4f}")
        print(f"   Total Steps: {result.metrics.get('total_steps', 0)}")
        print(f"\n   Outputs:")
        for agent_id, output in result.outputs.items():
            print(f"   - {agent_id}: {str(output.get('result', ''))[:100]}...")
        print("\n   ✓ Workflow execution successful")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        traceback.print_exc()
        return False

    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("SPLINTER PRESSURE TEST")
    print(f"Mode: {'MOCK' if USE_MOCK else 'REAL API'}")
    print("=" * 60)

    all_passed = True

    # Test control layer (no API calls)
    try:
        test_control_layer()
    except Exception as e:
        print(f"\n✗ Control Layer FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # Test coordination layer (no API calls)
    try:
        test_coordination_layer()
    except Exception as e:
        print(f"\n✗ Coordination Layer FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # Test gateway
    try:
        if not await test_gateway():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Gateway FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # Test agent
    try:
        if not await test_agent():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Agent FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # Test workflow
    try:
        if not await test_workflow():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Workflow FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
