"""Example: Research Pipeline with Splinter.

This example demonstrates a multi-agent research pipeline that:
1. Researches a topic using one agent
2. Summarizes the research using another agent
3. Creates an action plan using a third agent

All with execution limits, state management, and handoff validation.
"""

import asyncio
import os

from splinter import (
    AgentConfig,
    ExecutionLimits,
    Gateway,
    LLMProvider,
    Workflow,
    configure_logging,
)


async def main():
    # Configure logging
    configure_logging(level="INFO")

    # Create workflow with execution limits
    workflow = Workflow(
        workflow_id="research-pipeline",
        name="Research Pipeline",
        limits=ExecutionLimits(
            max_budget=5.0,      # Max $5 spend
            max_steps=50,        # Max 50 LLM calls
            max_time_seconds=300 # Max 5 minutes
        ),
    )

    # Configure LLM providers (use environment variables for API keys)
    workflow.configure_providers(
        openai={"api_key": os.environ.get("OPENAI_API_KEY")},
        anthropic={"api_key": os.environ.get("ANTHROPIC_API_KEY")},
    )

    # Add research agent (uses OpenAI GPT-4)
    workflow.add_agent(AgentConfig(
        agent_id="researcher",
        name="Research Agent",
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        system_prompt="""You are a research assistant. Your job is to:
1. Analyze the given topic
2. Identify key concepts and themes
3. Provide factual, well-organized research findings

Output your research as a JSON object with:
- "key_findings": list of main discoveries
- "themes": list of major themes
- "sources": suggested areas for further research
""",
        state_ownership=["researcher.*"],  # Owns all researcher.* fields
    ))

    # Add summarizer agent (uses Anthropic Claude)
    workflow.add_agent(AgentConfig(
        agent_id="summarizer",
        name="Summarization Agent",
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        system_prompt="""You are a summarization expert. Your job is to:
1. Read the research findings provided
2. Create a concise, clear summary
3. Highlight the most important points

Output your summary as a JSON object with:
- "executive_summary": 2-3 sentence overview
- "key_points": list of 3-5 main takeaways
- "recommendations": list of suggested next steps
""",
        state_ownership=["summarizer.*"],
    ))

    # Add planner agent (uses OpenAI)
    workflow.add_agent(AgentConfig(
        agent_id="planner",
        name="Planning Agent",
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        system_prompt="""You are a strategic planner. Your job is to:
1. Review the research summary
2. Create an actionable plan
3. Define clear next steps

Output your plan as a JSON object with:
- "objectives": list of goals to achieve
- "action_items": list of specific tasks with priorities
- "timeline": suggested timeline for execution
""",
        state_ownership=["planner.*"],
    ))

    # Define handoff schemas for validation
    workflow.add_handoff_schema(
        source="researcher",
        target="summarizer",
        schema={
            "type": "object",
            "properties": {
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "themes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key_findings"],
        },
    )

    workflow.add_handoff_schema(
        source="summarizer",
        target="planner",
        schema={
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["executive_summary"],
        },
    )

    # Define execution steps
    workflow.add_step("researcher")
    workflow.add_step("summarizer", depends_on=["researcher"])
    workflow.add_step("planner", depends_on=["summarizer"])

    # Run the workflow
    print("Starting research pipeline...")
    print("=" * 50)

    result = await workflow.run(
        initial_state={
            "topic": "The impact of large language models on software development",
            "depth": "comprehensive",
        }
    )

    # Display results
    print("\n" + "=" * 50)
    print("WORKFLOW COMPLETED")
    print("=" * 50)
    print(f"Status: {result.status.value}")
    print(f"Total Cost: ${result.metrics.get('total_cost', 0):.4f}")
    print(f"Total Steps: {result.metrics.get('total_steps', 0)}")
    print(f"Elapsed Time: {result.metrics.get('elapsed_seconds', 0):.2f}s")

    print("\n--- Research Findings ---")
    print(result.outputs.get("researcher", {}).get("result", "No output"))

    print("\n--- Summary ---")
    print(result.outputs.get("summarizer", {}).get("result", "No output"))

    print("\n--- Action Plan ---")
    print(result.outputs.get("planner", {}).get("result", "No output"))

    return result


if __name__ == "__main__":
    asyncio.run(main())
