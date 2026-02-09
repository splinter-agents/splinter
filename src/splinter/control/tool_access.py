"""Tool access control for Splinter.

This module provides a permission system defining:
- Which agent can call which tool
- Hard enforcement at runtime

Without this, every agent does everything, costs explode, and
behavior becomes nondeterministic. This is how roles become real.
"""

import fnmatch
import threading
from datetime import datetime
from typing import Any, Callable

from ..exceptions import ToolAccessDeniedError
from ..schemas import ToolCall, ToolPermission


class ToolAccessController:
    """Controls which agents can access which tools.

    This class enforces tool permissions at runtime, ensuring that:
    - Only authorized agents can call specific tools
    - Unauthorized calls are blocked, logged, and counted as failures
    - Tool usage can be audited

    Example:
        controller = ToolAccessController()

        # Define permissions
        controller.register_tool("web_search", allowed_agents=["researcher"])
        controller.register_tool("database_write", allowed_agents=["executor"])
        controller.register_tool("read_file", allowed_agents=["*"])  # All agents

        # Check before tool execution
        controller.check_access("researcher", "web_search")  # OK
        controller.check_access("executor", "web_search")  # Raises ToolAccessDeniedError
    """

    def __init__(
        self,
        default_policy: str = "allow",  # "allow" or "deny"
        on_access_denied: Callable[[str, str], None] | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
    ):
        """Initialize the tool access controller.

        Args:
            default_policy: Default policy when no permission defined ("allow" or "deny").
            on_access_denied: Callback when access is denied (agent_id, tool_name).
            on_tool_call: Callback after each tool call (for logging/auditing).
        """
        self._default_policy = default_policy
        self._on_access_denied = on_access_denied
        self._on_tool_call = on_tool_call
        self._permissions: dict[str, ToolPermission] = {}
        self._tool_history: list[ToolCall] = []
        self._denied_count: dict[str, int] = {}  # Per-agent denied call count
        self._lock = threading.RLock()

    def register_tool(
        self,
        tool_name: str,
        allowed_agents: list[str] | None = None,
        denied_agents: list[str] | None = None,
        require_approval: bool = False,
    ) -> None:
        """Register a tool with its access permissions.

        Args:
            tool_name: Name of the tool (supports wildcards like "db_*").
            allowed_agents: List of agent IDs allowed to use this tool.
                           Use ["*"] to allow all agents.
            denied_agents: List of agent IDs explicitly denied.
            require_approval: If True, tool calls need human approval.
        """
        with self._lock:
            self._permissions[tool_name] = ToolPermission(
                tool_name=tool_name,
                allowed_agents=allowed_agents or [],
                denied_agents=denied_agents or [],
                require_approval=require_approval,
            )

    def register_tools(self, permissions: list[ToolPermission]) -> None:
        """Register multiple tools at once.

        Args:
            permissions: List of tool permissions.
        """
        with self._lock:
            for perm in permissions:
                self._permissions[perm.tool_name] = perm

    def set_agent_tools(self, agent_id: str, allowed_tools: list[str]) -> None:
        """Set the tools an agent is allowed to use.

        This is an alternative way to configure permissions - from the
        agent's perspective rather than the tool's perspective.

        Args:
            agent_id: The agent ID.
            allowed_tools: List of tool names this agent can use.
        """
        with self._lock:
            # Update each tool's permission
            for tool_name in allowed_tools:
                if tool_name not in self._permissions:
                    self._permissions[tool_name] = ToolPermission(
                        tool_name=tool_name,
                        allowed_agents=[agent_id],
                    )
                elif agent_id not in self._permissions[tool_name].allowed_agents:
                    self._permissions[tool_name].allowed_agents.append(agent_id)

    def check_access(self, agent_id: str, tool_name: str) -> bool:
        """Check if an agent can access a tool.

        Args:
            agent_id: The agent attempting to use the tool.
            tool_name: The tool being accessed.

        Returns:
            True if access is allowed.

        Raises:
            ToolAccessDeniedError: If access is denied.
        """
        with self._lock:
            permission = self._find_permission(tool_name)

            if permission is None:
                # No permission defined - use default policy
                if self._default_policy == "deny":
                    self._record_denial(agent_id, tool_name)
                    raise ToolAccessDeniedError(agent_id, tool_name)
                return True

            # Check explicit denials first
            if self._matches_agent_list(agent_id, permission.denied_agents):
                self._record_denial(agent_id, tool_name)
                raise ToolAccessDeniedError(agent_id, tool_name)

            # Check allowed list
            if permission.allowed_agents:
                if not self._matches_agent_list(agent_id, permission.allowed_agents):
                    self._record_denial(agent_id, tool_name)
                    raise ToolAccessDeniedError(agent_id, tool_name)

            return True

    def _find_permission(self, tool_name: str) -> ToolPermission | None:
        """Find permission for a tool, supporting wildcards."""
        # Exact match first
        if tool_name in self._permissions:
            return self._permissions[tool_name]

        # Wildcard match
        for pattern, perm in self._permissions.items():
            if fnmatch.fnmatch(tool_name, pattern):
                return perm

        return None

    def _matches_agent_list(self, agent_id: str, agent_list: list[str]) -> bool:
        """Check if an agent matches any pattern in the list."""
        for pattern in agent_list:
            if pattern == "*" or fnmatch.fnmatch(agent_id, pattern):
                return True
        return False

    def _record_denial(self, agent_id: str, tool_name: str) -> None:
        """Record a denied access attempt."""
        key = f"{agent_id}:{tool_name}"
        self._denied_count[key] = self._denied_count.get(key, 0) + 1

        if self._on_access_denied:
            self._on_access_denied(agent_id, tool_name)

    def record_tool_call(
        self,
        tool_name: str,
        agent_id: str,
        arguments: dict[str, Any],
        result: Any = None,
        success: bool = True,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> ToolCall:
        """Record a tool call for auditing.

        Args:
            tool_name: Name of the tool.
            agent_id: Agent that made the call.
            arguments: Arguments passed to the tool.
            result: Result from the tool.
            success: Whether the call succeeded.
            error: Error message if failed.
            duration_ms: Call duration in milliseconds.

        Returns:
            The recorded ToolCall.
        """
        with self._lock:
            call = ToolCall(
                tool_name=tool_name,
                agent_id=agent_id,
                arguments=arguments,
                result=result,
                success=success,
                error=error,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
            )

            self._tool_history.append(call)

            if self._on_tool_call:
                self._on_tool_call(call)

            return call

    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires human approval.

        Args:
            tool_name: The tool to check.

        Returns:
            True if approval is required.
        """
        with self._lock:
            permission = self._find_permission(tool_name)
            return permission.require_approval if permission else False

    def get_allowed_tools(self, agent_id: str) -> list[str]:
        """Get list of tools an agent is allowed to use.

        Args:
            agent_id: The agent ID.

        Returns:
            List of tool names.
        """
        with self._lock:
            allowed = []
            for tool_name, perm in self._permissions.items():
                # Skip if explicitly denied
                if self._matches_agent_list(agent_id, perm.denied_agents):
                    continue

                # Add if allowed
                if not perm.allowed_agents or self._matches_agent_list(
                    agent_id, perm.allowed_agents
                ):
                    allowed.append(tool_name)

            return allowed

    def get_tool_history(
        self,
        agent_id: str | None = None,
        tool_name: str | None = None,
        limit: int | None = None,
    ) -> list[ToolCall]:
        """Get tool call history with optional filters.

        Args:
            agent_id: Filter by agent.
            tool_name: Filter by tool.
            limit: Maximum number of results.

        Returns:
            List of tool calls matching filters.
        """
        with self._lock:
            history = self._tool_history

            if agent_id:
                history = [c for c in history if c.agent_id == agent_id]

            if tool_name:
                history = [c for c in history if c.tool_name == tool_name]

            if limit:
                history = history[-limit:]

            return list(history)

    def get_denial_stats(self) -> dict[str, int]:
        """Get statistics on denied access attempts.

        Returns:
            Dict mapping "agent:tool" to denial count.
        """
        with self._lock:
            return dict(self._denied_count)

    def reset(self) -> None:
        """Reset tool history and denial counts."""
        with self._lock:
            self._tool_history.clear()
            self._denied_count.clear()


class ToolRegistry:
    """Registry for tool definitions with schema validation.

    This complements ToolAccessController by storing tool metadata
    and providing tool discovery.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        handler: Callable[..., Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Register a tool definition.

        Args:
            name: Tool name.
            description: Tool description.
            parameters: JSON Schema for parameters.
            handler: Optional callable to execute the tool.
            tags: Optional tags for categorization.
        """
        with self._lock:
            self._tools[name] = {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}},
                "handler": handler,
                "tags": tags or [],
            }

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a tool definition by name."""
        with self._lock:
            return self._tools.get(name)

    def get_all(self) -> list[dict[str, Any]]:
        """Get all registered tools."""
        with self._lock:
            return [
                {k: v for k, v in tool.items() if k != "handler"}
                for tool in self._tools.values()
            ]

    def get_by_tags(self, tags: list[str]) -> list[dict[str, Any]]:
        """Get tools matching any of the given tags."""
        with self._lock:
            results = []
            for tool in self._tools.values():
                if any(tag in tool.get("tags", []) for tag in tags):
                    results.append({k: v for k, v in tool.items() if k != "handler"})
            return results

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name.

        Args:
            name: Tool name.
            **kwargs: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            KeyError: If tool not found.
            ValueError: If tool has no handler.
        """
        with self._lock:
            tool = self._tools.get(name)
            if not tool:
                raise KeyError(f"Tool '{name}' not found")
            if not tool.get("handler"):
                raise ValueError(f"Tool '{name}' has no handler")
            return tool["handler"](**kwargs)

    def to_openai_format(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Convert tools to OpenAI function calling format.

        Args:
            tool_names: Optional filter for specific tools.

        Returns:
            List of tool definitions in OpenAI format.
        """
        with self._lock:
            tools = []
            for name, tool in self._tools.items():
                if tool_names and name not in tool_names:
                    continue
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                })
            return tools

    def to_anthropic_format(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Convert tools to Anthropic tool use format.

        Args:
            tool_names: Optional filter for specific tools.

        Returns:
            List of tool definitions in Anthropic format.
        """
        with self._lock:
            tools = []
            for name, tool in self._tools.items():
                if tool_names and name not in tool_names:
                    continue
                tools.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                })
            return tools
