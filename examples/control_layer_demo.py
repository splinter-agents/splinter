"""Example: Control Layer Features Demo.

This example demonstrates Splinter's control layer features
without requiring actual LLM API calls.
"""

import time
from datetime import datetime

from splinter import (
    # Control Layer
    LimitsEnforcer,
    LoopDetector,
    ToolAccessController,
    MemoryStore,
    AgentMemory,
    # Types
    ExecutionLimits,
    LoopDetectionConfig,
    MemoryConfig,
    EvictionPolicy,
    # Exceptions
    BudgetExceededError,
    StepLimitExceededError,
    LoopDetectedError,
    ToolAccessDeniedError,
)


def demo_execution_limits():
    """Demonstrate execution limits enforcement."""
    print("\n" + "=" * 50)
    print("EXECUTION LIMITS DEMO")
    print("=" * 50)

    # Create enforcer with limits
    limits = ExecutionLimits(
        max_budget=1.0,    # $1 max
        max_steps=10,      # 10 steps max
        max_time_seconds=5 # 5 seconds max
    )

    enforcer = LimitsEnforcer(
        limits=limits,
        on_limit_warning=lambda t, c, l: print(f"  ⚠️  Warning: {t} at {c:.2f}/{l:.2f}"),
        warning_threshold=0.8,
    )

    enforcer.start()

    # Simulate some steps
    print("\nSimulating execution steps...")

    for i in range(15):
        try:
            # Check limits before each step
            enforcer.check_limits()

            # Simulate work
            enforcer.record_cost(0.08, input_tokens=100, output_tokens=50)
            enforcer.increment_steps()

            remaining = enforcer.get_remaining()
            print(f"  Step {i+1}: ${enforcer.metrics.total_cost:.2f} spent, "
                  f"{enforcer.metrics.total_steps} steps")

        except BudgetExceededError as e:
            print(f"\n  🛑 Budget limit hit: {e}")
            break
        except StepLimitExceededError as e:
            print(f"\n  🛑 Step limit hit: {e}")
            break

    print(f"\nFinal metrics:")
    print(f"  Total cost: ${enforcer.metrics.total_cost:.2f}")
    print(f"  Total steps: {enforcer.metrics.total_steps}")
    print(f"  Total tokens: {enforcer.metrics.total_tokens}")


def demo_loop_detection():
    """Demonstrate loop detection."""
    print("\n" + "=" * 50)
    print("LOOP DETECTION DEMO")
    print("=" * 50)

    config = LoopDetectionConfig(
        max_repeated_outputs=3,
        max_no_state_change=4,
        max_ping_pong=3,
    )

    detector = LoopDetector(
        config=config,
        on_loop_warning=lambda p, c: print(f"  ⚠️  Loop warning: {p}"),
    )

    # Simulate repeated outputs
    print("\nSimulating repeated outputs from same agent...")
    state = {"counter": 0}

    for i in range(5):
        try:
            # Agent produces same output repeatedly
            detector.record_step(
                agent_id="stuck_agent",
                input_data={"query": "what is 2+2?"},
                output_data="The answer is 4",  # Same output every time
                state=state,
            )
            detector.check_for_loops()
            print(f"  Step {i+1}: Agent produced output")

        except LoopDetectedError as e:
            print(f"\n  🛑 Loop detected: {e}")
            break

    # Reset and demo ping-pong
    detector.reset()
    print("\nSimulating ping-pong between agents...")
    state = {"value": "A"}

    for i in range(10):
        try:
            agent = "agent_A" if i % 2 == 0 else "agent_B"
            state["value"] = "A" if state["value"] == "B" else "B"

            detector.record_step(
                agent_id=agent,
                input_data={"from": "other"},
                output_data=f"Handing off to other agent",
                state=state,
            )
            detector.check_for_loops()
            print(f"  Step {i+1}: {agent} -> state={state['value']}")

        except LoopDetectedError as e:
            print(f"\n  🛑 Loop detected: {e}")
            break

    # Show analysis
    print("\nPattern analysis:")
    analysis = detector.get_pattern_analysis()
    print(f"  Total steps analyzed: {analysis['total_steps']}")


def demo_tool_access():
    """Demonstrate tool access control."""
    print("\n" + "=" * 50)
    print("TOOL ACCESS CONTROL DEMO")
    print("=" * 50)

    controller = ToolAccessController(
        default_policy="deny",
        on_access_denied=lambda a, t: print(f"  🚫 Denied: {a} -> {t}"),
    )

    # Register tools with permissions
    controller.register_tool("web_search", allowed_agents=["researcher"])
    controller.register_tool("database_write", allowed_agents=["executor"])
    controller.register_tool("read_file", allowed_agents=["*"])  # All agents

    print("\nTesting tool access...")

    tests = [
        ("researcher", "web_search"),
        ("researcher", "database_write"),
        ("executor", "database_write"),
        ("executor", "web_search"),
        ("any_agent", "read_file"),
    ]

    for agent_id, tool_name in tests:
        try:
            controller.check_access(agent_id, tool_name)
            print(f"  ✅ {agent_id} can use {tool_name}")
        except ToolAccessDeniedError:
            print(f"  ❌ {agent_id} cannot use {tool_name}")

    # Show which tools each agent can use
    print("\nAgent permissions:")
    for agent in ["researcher", "executor", "other"]:
        tools = controller.get_allowed_tools(agent)
        print(f"  {agent}: {tools}")


def demo_memory_limits():
    """Demonstrate memory limits and eviction."""
    print("\n" + "=" * 50)
    print("MEMORY LIMITS DEMO")
    print("=" * 50)

    config = MemoryConfig(
        max_size_bytes=1000,  # 1KB limit for demo
        ttl_seconds=2,        # 2 second TTL
        eviction_policy=EvictionPolicy.FIFO,
        max_entries=5,
    )

    store = MemoryStore(
        config=config,
        on_eviction=lambda k, v: print(f"  🗑️  Evicted: {k}"),
    )

    # Store some values
    print("\nStoring values...")
    for i in range(7):
        key = f"item_{i}"
        value = {"data": f"Value {i}", "timestamp": datetime.now().isoformat()}
        store.set(key, value, agent_id="agent_1")
        print(f"  Stored {key} (size: {store.current_size} bytes, "
              f"entries: {store.entry_count})")

    print(f"\nCurrent keys: {store.keys()}")

    # Demonstrate TTL expiration
    print("\nWaiting for TTL expiration...")
    time.sleep(2.5)

    # Access triggers cleanup
    store.get("item_6")  # This will clean expired entries
    print(f"Keys after TTL: {store.keys()}")

    # Show stats
    stats = store.get_stats()
    print(f"\nMemory stats:")
    print(f"  Usage: {stats['usage_percent']:.1f}%")
    print(f"  Entries: {stats['total_entries']}/{stats['max_entries']}")

    # Agent-specific memory
    print("\nAgent-specific memory:")
    agent_memory = AgentMemory(store, "my_agent")
    agent_memory.set("private_key", {"secret": "data"})
    print(f"  Agent keys: {agent_memory.keys()}")


def main():
    """Run all demos."""
    print("=" * 50)
    print("SPLINTER CONTROL LAYER DEMOS")
    print("=" * 50)

    demo_execution_limits()
    demo_loop_detection()
    demo_tool_access()
    demo_memory_limits()

    print("\n" + "=" * 50)
    print("ALL DEMOS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()