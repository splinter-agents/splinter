"""Agent definition and execution for Splinter workflows."""

import json
from typing import Any, Callable

from ..control.memory import AgentMemory
from ..coordination.state import SharedState
from ..gateway.gateway import Gateway
from ..types import (
    AgentConfig,
    AgentStatus,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)


class Agent:
    """An agent in a Splinter workflow.

    An agent represents a single AI entity with:
    - A specific role/purpose
    - Access to certain tools
    - Ownership of certain state fields
    - Configuration for LLM calls

    Agents don't track history - Splinter tracks history.
    Agents only see current state + task.

    Example:
        config = AgentConfig(
            agent_id="researcher",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            system_prompt="You are a research assistant...",
            tools=["web_search", "read_file"],
            state_ownership=["research.*"],
        )

        agent = Agent(config, gateway=gateway)
        result = await agent.run(
            task="Research the topic of...",
            state=shared_state,
        )
    """

    def __init__(
        self,
        config: AgentConfig,
        gateway: Gateway,
        memory: AgentMemory | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        """Initialize an agent.

        Args:
            config: Agent configuration.
            gateway: Gateway for LLM calls.
            memory: Optional agent-specific memory.
            tool_executor: Function to execute tools.
        """
        self._config = config
        self._gateway = gateway
        self._memory = memory
        self._tool_executor = tool_executor
        self._status = AgentStatus.PENDING
        self._last_response: LLMResponse | None = None

    @property
    def id(self) -> str:
        """Agent ID."""
        return self._config.agent_id

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    @property
    def status(self) -> AgentStatus:
        """Current agent status."""
        return self._status

    @property
    def last_response(self) -> LLMResponse | None:
        """Last LLM response from this agent."""
        return self._last_response

    def _build_messages(
        self,
        task: str,
        state: SharedState | dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[LLMMessage]:
        """Build messages for the LLM call.

        Args:
            task: The task to perform.
            state: Current shared state.
            context: Additional context.

        Returns:
            List of messages for the LLM.
        """
        messages: list[LLMMessage] = []

        # System prompt
        if self._config.system_prompt:
            system_content = self._config.system_prompt

            # Inject state information if available
            if state:
                state_data = state.to_dict() if hasattr(state, "to_dict") else state
                system_content += f"\n\nCurrent state:\n```json\n{json.dumps(state_data, indent=2, default=str)}\n```"

            messages.append(LLMMessage(role="system", content=system_content))

        # User message with task
        user_content = task

        # Add context if provided
        if context:
            user_content += f"\n\nAdditional context:\n```json\n{json.dumps(context, indent=2, default=str)}\n```"

        messages.append(LLMMessage(role="user", content=user_content))

        return messages

    def _get_tools(self) -> list[dict[str, Any]] | None:
        """Get tool definitions for this agent."""
        if not self._config.tools:
            return None

        # This should be populated from a tool registry
        # For now, return a placeholder structure
        tools = []
        for tool_name in self._config.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Tool: {tool_name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        return tools if tools else None

    async def run(
        self,
        task: str,
        state: SharedState | dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_tool_iterations: int = 10,
    ) -> dict[str, Any]:
        """Run the agent on a task.

        Args:
            task: The task to perform.
            state: Current shared state.
            context: Additional context.
            max_tool_iterations: Max tool call iterations.

        Returns:
            Agent output as a dict.
        """
        self._status = AgentStatus.RUNNING

        try:
            messages = self._build_messages(task, state, context)
            tools = self._get_tools()

            state_dict = state.to_dict() if hasattr(state, "to_dict") else (state or {})

            if tools and self._tool_executor:
                # Run with tool execution loop
                response = await self._gateway.complete_with_tools(
                    agent_id=self.id,
                    provider=self._config.provider,
                    model=self._config.model,
                    messages=messages,
                    tools=tools,
                    tool_executor=self._tool_executor,
                    max_iterations=max_tool_iterations,
                    state=state_dict,
                )
            else:
                # Simple completion
                response = await self._gateway.complete(
                    agent_id=self.id,
                    provider=self._config.provider,
                    model=self._config.model,
                    messages=messages,
                    tools=tools,
                    state=state_dict,
                )

            self._last_response = response
            self._status = AgentStatus.COMPLETED

            # Parse response
            return self._parse_response(response)

        except Exception as e:
            self._status = AgentStatus.FAILED
            raise

    def _parse_response(self, response: LLMResponse) -> dict[str, Any]:
        """Parse LLM response into structured output.

        Args:
            response: The LLM response.

        Returns:
            Parsed output dict.
        """
        content = response.content or ""

        # Try to parse as JSON if it looks like JSON
        if content.strip().startswith("{") or content.strip().startswith("["):
            try:
                return {"result": json.loads(content), "raw": content}
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            try:
                start = content.index("```json") + 7
                end = content.index("```", start)
                json_str = content[start:end].strip()
                return {"result": json.loads(json_str), "raw": content}
            except (ValueError, json.JSONDecodeError):
                pass

        # Return as plain text
        return {
            "result": content,
            "raw": content,
            "tool_calls": response.tool_calls,
        }

    def reset(self) -> None:
        """Reset agent state."""
        self._status = AgentStatus.PENDING
        self._last_response = None


class AgentBuilder:
    """Builder for creating agents with fluent API.

    Example:
        agent = (
            AgentBuilder("researcher")
            .with_provider(LLMProvider.ANTHROPIC, "claude-3-sonnet")
            .with_system_prompt("You are a research assistant...")
            .with_tools(["web_search", "read_file"])
            .with_state_ownership(["research.*"])
            .build(gateway)
        )
    """

    def __init__(self, agent_id: str):
        """Initialize the builder.

        Args:
            agent_id: Unique agent identifier.
        """
        self._config = AgentConfig(agent_id=agent_id)

    def with_name(self, name: str) -> "AgentBuilder":
        """Set agent display name."""
        self._config.name = name
        return self

    def with_description(self, description: str) -> "AgentBuilder":
        """Set agent description."""
        self._config.description = description
        return self

    def with_provider(
        self, provider: LLMProvider, model: str | None = None
    ) -> "AgentBuilder":
        """Set LLM provider and model."""
        self._config.provider = provider
        if model:
            self._config.model = model
        return self

    def with_model(self, model: str) -> "AgentBuilder":
        """Set model name."""
        self._config.model = model
        return self

    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        """Set system prompt."""
        self._config.system_prompt = prompt
        return self

    def with_tools(self, tools: list[str]) -> "AgentBuilder":
        """Set available tools."""
        self._config.tools = tools
        return self

    def with_allowed_tools(self, tools: list[str]) -> "AgentBuilder":
        """Set allowed tools (for access control)."""
        self._config.allowed_tools = tools
        return self

    def with_state_ownership(self, fields: list[str]) -> "AgentBuilder":
        """Set state fields this agent owns."""
        self._config.state_ownership = fields
        return self

    def with_output_schema(self, schema: dict[str, Any]) -> "AgentBuilder":
        """Set expected output schema."""
        self._config.output_schema = schema
        return self

    def with_limits(
        self, max_steps: int | None = None, max_budget: float | None = None
    ) -> "AgentBuilder":
        """Set per-agent limits."""
        self._config.max_steps = max_steps
        self._config.max_budget = max_budget
        return self

    def build(
        self,
        gateway: Gateway,
        memory: AgentMemory | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> Agent:
        """Build the agent.

        Args:
            gateway: Gateway for LLM calls.
            memory: Optional agent memory.
            tool_executor: Optional tool executor.

        Returns:
            Configured Agent instance.
        """
        return Agent(
            config=self._config,
            gateway=gateway,
            memory=memory,
            tool_executor=tool_executor,
        )

    @property
    def config(self) -> AgentConfig:
        """Get the current configuration."""
        return self._config
