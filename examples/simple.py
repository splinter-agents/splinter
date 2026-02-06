"""Splinter examples with OpenAI.

Run: OPENAI_API_KEY=sk-... python examples/simple.py
"""

import asyncio
import os

from splinter import (
    Gateway,
    Workflow,
    AgentConfig,
    AgentBuilder,
    LLMProvider,
    ExecutionLimits,
)

API_KEY = os.environ.get("OPENAI_API_KEY", "")


async def single_agent():
    """Example 1: Single agent."""
    print("\n" + "=" * 50)
    print("EXAMPLE 1: SINGLE AGENT")
    print("=" * 50)

    gateway = Gateway(limits=ExecutionLimits(max_budget=0.50))
    gateway.configure_provider("openai", api_key=API_KEY)

    agent = (
        AgentBuilder("analyst")
        .with_provider(LLMProvider.OPENAI, "gpt-4o-mini")
        .with_system_prompt("Analyze topics. Output JSON: {analysis: string, points: [string]}")
        .build(gateway)
    )

    result = await agent.run(task="Analyze the benefits of exercise")

    print(f"Result: {result['result']}")
    print(f"Cost: ${gateway.get_metrics()['total_cost']:.4f}")


async def multi_agent_pipeline():
    """Example 2: Multi-agent pipeline."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: MULTI-AGENT PIPELINE")
    print("=" * 50)

    # Gateway with limits
    gateway = Gateway(limits=ExecutionLimits(max_budget=1.0, max_steps=10))
    gateway.configure_provider("openai", api_key=API_KEY)

    # Workflow
    workflow = Workflow(workflow_id="research-pipeline")
    workflow._gateway = gateway

    # Agent 1: Researcher
    workflow.add_agent(AgentConfig(
        agent_id="researcher",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        system_prompt="Research topics. Output JSON: {findings: [string]}",
    ))

    # Agent 2: Writer
    workflow.add_agent(AgentConfig(
        agent_id="writer",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        system_prompt="Write based on research. Output JSON: {article: string}",
    ))

    # Flow: researcher -> writer
    workflow.add_step("researcher")
    workflow.add_step("writer", depends_on=["researcher"])

    # Run
    result = await workflow.run(initial_state={"topic": "AI in healthcare"})

    print(f"Status: {result.status}")
    print(f"Cost: ${result.metrics['total_cost']:.4f}")
    for agent_id, output in result.outputs.items():
        print(f"\n{agent_id}: {output.get('result', output)}")


async def main():
    if not API_KEY:
        print("Usage: OPENAI_API_KEY=sk-... python examples/simple.py")
        return

    await single_agent()
    await multi_agent_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
