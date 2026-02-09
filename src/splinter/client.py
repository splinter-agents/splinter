"""Splinter - Simple API.

Usage:
    from splinter import Splinter

    # Local (free)
    s = Splinter(openai_key="sk-...")
    result = await s.run("researcher", "Find info about AI")

    # With cloud features (paid)
    s = Splinter(openai_key="sk-...", api_key="sk-splinter-...")
    # Now get live dashboard, remote control, etc.
"""

import asyncio
import os
from typing import Any, Optional

from .gateway import Gateway
from .workflow import Agent, AgentBuilder, Workflow, WorkflowResult
from .schemas import (
    AgentConfig,
    AgentStatus,
    ExecutionLimits,
    LLMProvider,
    LoopDetectionConfig,
)
from .cloud import CloudClient


class Splinter:
    """Dead simple multi-agent control.

    Usage:
        from splinter import Splinter

        # Setup (one line)
        s = Splinter(openai_key="sk-...")

        # Single agent (one line)
        result = await s.run("helper", "What is 2+2?")

        # Or with custom prompt
        result = await s.run(
            "analyst",
            "Analyze this data",
            instructions="You analyze data. Output JSON."
        )

        # Multi-agent pipeline
        s.add_agent("researcher", "Research topics. Output JSON: {findings: []}")
        s.add_agent("writer", "Write articles. Output JSON: {article: string}")

        result = await s.pipeline(
            ["researcher", "writer"],
            input={"topic": "AI trends"}
        )
    """

    def __init__(
        self,
        openai_key: str | None = None,
        anthropic_key: str | None = None,
        gemini_key: str | None = None,
        grok_key: str | None = None,
        max_budget: float = 10.0,
        max_steps: int = 100,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Create Splinter instance.

        Args:
            openai_key: OpenAI API key (or set OPENAI_API_KEY env var)
            anthropic_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            gemini_key: Gemini API key (or set GEMINI_API_KEY env var)
            grok_key: Grok/xAI API key (or set XAI_API_KEY env var)
            max_budget: Max spend in dollars (default $10)
            max_steps: Max LLM calls (default 100)
            model: Default model (auto-detected based on provider)
            api_key: Splinter Cloud API key for paid features (optional)
        """
        self._gateway = Gateway(
            limits=ExecutionLimits(max_budget=max_budget, max_steps=max_steps)
        )
        self._default_provider: LLMProvider | None = None
        self._cloud: Optional[CloudClient] = None

        # Connect to Splinter Cloud if API key provided
        api_key = api_key or os.environ.get("SPLINTER_API_KEY")
        if api_key:
            self._cloud = CloudClient(api_key=api_key)
            self._cloud.register("gateway", self._gateway)

        # Configure providers (in priority order)
        openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self._gateway.configure_provider("openai", api_key=openai_key)
            self._default_provider = LLMProvider.OPENAI
            self._model = model or "gpt-4o-mini"

        anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self._gateway.configure_provider("anthropic", api_key=anthropic_key)
            if self._default_provider is None:
                self._default_provider = LLMProvider.ANTHROPIC
                self._model = model or "claude-sonnet-4-20250514"

        gemini_key = gemini_key or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            self._gateway.configure_provider("gemini", api_key=gemini_key)
            if self._default_provider is None:
                self._default_provider = LLMProvider.GEMINI
                self._model = model or "gemini-1.5-flash"

        grok_key = grok_key or os.environ.get("XAI_API_KEY")
        if grok_key:
            self._gateway.configure_provider("grok", api_key=grok_key)
            if self._default_provider is None:
                self._default_provider = LLMProvider.GROK
                self._model = model or "grok-3-mini-fast"

        # Fallback if no provider configured
        if self._default_provider is None:
            self._default_provider = LLMProvider.OPENAI
            self._model = model or "gpt-4o-mini"

        self._agents: dict[str, AgentConfig] = {}

    # =========================================================================
    # SIMPLE API
    # =========================================================================

    async def run(
        self,
        agent_name: str,
        task: str,
        instructions: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run a single agent.

        Args:
            agent_name: Name for the agent
            task: What to do
            instructions: System prompt (optional)
            model: Model to use (optional)

        Returns:
            Agent output as dict

        Example:
            result = await s.run("helper", "What is 2+2?")
            result = await s.run(
                "analyst",
                "Analyze sales",
                instructions="You analyze data. Output JSON."
            )
        """
        agent = (
            AgentBuilder(agent_name)
            .with_provider(self._default_provider, model or self._model)
            .with_system_prompt(instructions or f"You are a helpful {agent_name}. Output JSON when possible.")
            .build(self._gateway)
        )

        return await agent.run(task=task)

    def add_agent(
        self,
        name: str,
        instructions: str,
        model: str | None = None,
        tools: list[str] | None = None,
        owns: list[str] | None = None,
    ) -> "Splinter":
        """Add an agent for pipelines.

        Args:
            name: Agent name (used as ID)
            instructions: System prompt
            model: Model to use (optional)
            tools: Tools the agent can use (optional)
            owns: State fields this agent owns (optional)

        Returns:
            Self for chaining

        Example:
            s.add_agent("researcher", "Research topics. Output JSON.")
            s.add_agent("writer", "Write articles.", owns=["content.*"])
        """
        self._agents[name] = AgentConfig(
            agent_id=name,
            provider=self._default_provider,
            model=model or self._model,
            system_prompt=instructions,
            tools=tools or [],
            state_ownership=owns or [],
        )
        return self

    async def pipeline(
        self,
        agents: list[str],
        input: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Run a multi-agent pipeline.

        Agents run in order. Each waits for the previous.

        Args:
            agents: List of agent names (in order)
            input: Initial state/input

        Returns:
            WorkflowResult with outputs from all agents

        Example:
            s.add_agent("researcher", "Research. Output JSON: {findings: []}")
            s.add_agent("writer", "Write. Output JSON: {article: string}")

            result = await s.pipeline(
                ["researcher", "writer"],
                input={"topic": "AI"}
            )
        """
        workflow = Workflow(workflow_id="pipeline")
        workflow._gateway = self._gateway

        # Add agents
        for name in agents:
            if name not in self._agents:
                raise ValueError(f"Agent '{name}' not found. Call add_agent() first.")
            workflow.add_agent(self._agents[name])

        # Add steps with dependencies
        prev = None
        for name in agents:
            if prev:
                workflow.add_step(name, depends_on=[prev])
            else:
                workflow.add_step(name)
            prev = name

        return await workflow.run(initial_state=input or {})

    # =========================================================================
    # ADVANCED API
    # =========================================================================

    def agent(
        self,
        name: str,
        instructions: str | None = None,
        model: str | None = None,
    ) -> Agent:
        """Create a reusable agent.

        Args:
            name: Agent name
            instructions: System prompt
            model: Model to use

        Returns:
            Agent instance

        Example:
            researcher = s.agent("researcher", "Research topics.")
            result = await researcher.run(task="Find info about AI")
        """
        return (
            AgentBuilder(name)
            .with_provider(self._default_provider, model or self._model)
            .with_system_prompt(instructions or f"You are a helpful {name}.")
            .build(self._gateway)
        )

    @property
    def gateway(self) -> Gateway:
        """Access the underlying gateway."""
        return self._gateway

    @property
    def cost(self) -> float:
        """Total cost so far."""
        return self._gateway.get_metrics()["total_cost"]

    @property
    def steps(self) -> int:
        """Total steps so far."""
        return self._gateway.get_metrics()["total_steps"]

    def reset(self) -> None:
        """Reset metrics and state."""
        self._gateway.reset()

    # =========================================================================
    # CLOUD (PAID FEATURES)
    # =========================================================================

    @property
    def cloud(self) -> Optional[CloudClient]:
        """Access cloud client (None if not connected)."""
        return self._cloud

    @property
    def is_cloud_connected(self) -> bool:
        """Check if connected to Splinter Cloud."""
        return self._cloud is not None and self._cloud.is_connected

    async def connect_cloud(self, api_key: str | None = None) -> None:
        """Connect to Splinter Cloud for paid features.

        Args:
            api_key: Splinter Cloud API key

        Paid features enabled:
            - Live agent dashboard
            - Pause/resume/stop agents remotely
            - Live rule and limit changes
            - Rollback to checkpoint
            - Deadlock detection
            - Bottleneck analysis
        """
        api_key = api_key or os.environ.get("SPLINTER_API_KEY")
        if not api_key:
            raise ValueError("API key required. Pass api_key or set SPLINTER_API_KEY env var.")

        self._cloud = CloudClient(api_key=api_key)
        self._cloud.register("gateway", self._gateway)
        await self._cloud.connect_async()

    async def disconnect_cloud(self) -> None:
        """Disconnect from Splinter Cloud."""
        if self._cloud:
            await self._cloud.disconnect()
            self._cloud = None


# Convenience function for sync usage
def run_sync(coro):
    """Run async code synchronously.

    Example:
        from splinter import Splinter, run_sync

        s = Splinter(openai_key="sk-...")
        result = run_sync(s.run("helper", "What is 2+2?"))
    """
    return asyncio.run(coro)
