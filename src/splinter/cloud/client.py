"""
CloudClient - Connect to Splinter Cloud for paid features.
"""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

from .sync import StateSync, SyncEvent
from .commands import CommandHandler, Command, CommandType


class ConnectionState(Enum):
    """Cloud connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class CloudConfig:
    """Configuration for Splinter Cloud connection."""
    api_key: str
    endpoint: str = "https://api.splinter.dev"
    sync_interval: float = 1.0  # seconds
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    enable_telemetry: bool = True
    project_id: Optional[str] = None


@dataclass
class CloudClient:
    """
    Client for Splinter Cloud.

    Enables paid features:
    - Live agent dashboard
    - Remote pause/resume/stop
    - Live rule changes
    - Rollback & recovery
    - Deadlock detection
    - Bottleneck analysis

    Usage:
        client = CloudClient(api_key="sk-...")
        client.connect()

        # Now all agents are visible in the cloud dashboard
        # and can be controlled remotely
    """

    config: CloudConfig
    _state: ConnectionState = field(default=ConnectionState.DISCONNECTED, init=False)
    _sync: Optional[StateSync] = field(default=None, init=False)
    _command_handler: Optional[CommandHandler] = field(default=None, init=False)
    _sync_task: Optional[asyncio.Task] = field(default=None, init=False)
    _command_task: Optional[asyncio.Task] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _callbacks: dict[str, list[Callable]] = field(default_factory=dict, init=False)

    def __init__(self, api_key: str, **kwargs):
        """Create cloud client with API key."""
        self.config = CloudConfig(api_key=api_key, **kwargs)
        self._state = ConnectionState.DISCONNECTED
        self._sync = None
        self._command_handler = None
        self._sync_task = None
        self._command_task = None
        self._lock = threading.Lock()
        self._callbacks = {}
        self._local_refs = {}  # References to local control/coordination objects

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to cloud."""
        return self._state == ConnectionState.CONNECTED

    def connect(self) -> None:
        """Connect to Splinter Cloud (sync version)."""
        asyncio.get_event_loop().run_until_complete(self.connect_async())

    async def connect_async(self) -> None:
        """Connect to Splinter Cloud."""
        with self._lock:
            if self._state == ConnectionState.CONNECTED:
                return
            self._state = ConnectionState.CONNECTING

        try:
            # Initialize sync and command handler
            self._sync = StateSync(self.config)
            self._command_handler = CommandHandler(self.config, self._handle_command)

            # Validate API key with cloud
            await self._validate_api_key()

            # Start background tasks
            self._sync_task = asyncio.create_task(self._sync.start())
            self._command_task = asyncio.create_task(self._command_handler.start())

            self._state = ConnectionState.CONNECTED
            self._emit("connected", {})

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._emit("error", {"error": str(e)})
            raise

    async def disconnect(self) -> None:
        """Disconnect from Splinter Cloud."""
        with self._lock:
            if self._state == ConnectionState.DISCONNECTED:
                return

        # Stop background tasks
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self._command_task:
            self._command_task.cancel()
            try:
                await self._command_task
            except asyncio.CancelledError:
                pass

        self._state = ConnectionState.DISCONNECTED
        self._emit("disconnected", {})

    async def _validate_api_key(self) -> None:
        """Validate API key with cloud server."""
        # In production, this would make an HTTP request
        # For now, just validate format
        if not self.config.api_key.startswith("sk-"):
            raise ValueError("Invalid API key format. Expected 'sk-...'")

    def register(self, name: str, obj: Any) -> None:
        """Register a local object for cloud sync."""
        self._local_refs[name] = obj
        if self._sync:
            self._sync.register(name, obj)

    def sync_state(self, event: SyncEvent) -> None:
        """Send state update to cloud."""
        if self._sync and self.is_connected:
            self._sync.queue_event(event)

    async def _handle_command(self, command: Command) -> dict:
        """Handle command from cloud."""
        result = {"success": False, "error": "Unknown command"}

        # Route command to appropriate handler
        if command.type == CommandType.PAUSE_AGENT:
            result = await self._pause_agent(command.payload)
        elif command.type == CommandType.RESUME_AGENT:
            result = await self._resume_agent(command.payload)
        elif command.type == CommandType.STOP_AGENT:
            result = await self._stop_agent(command.payload)
        elif command.type == CommandType.GLOBAL_STOP:
            result = await self._global_stop(command.payload)
        elif command.type == CommandType.UPDATE_RULES:
            result = await self._update_rules(command.payload)
        elif command.type == CommandType.UPDATE_LIMITS:
            result = await self._update_limits(command.payload)
        elif command.type == CommandType.ROLLBACK:
            result = await self._rollback(command.payload)
        elif command.type == CommandType.BREAK_LOOP:
            result = await self._break_loop(command.payload)
        elif command.type == CommandType.UPDATE_TOOL_ACCESS:
            result = await self._update_tool_access(command.payload)

        self._emit("command_executed", {"command": command.type.value, "result": result})
        return result

    # Control commands (paid features)

    async def _pause_agent(self, payload: dict) -> dict:
        """Pause a running agent."""
        agent_id = payload.get("agent_id")
        if not agent_id:
            return {"success": False, "error": "agent_id required"}

        # Find agent in local refs and pause
        workflow = self._local_refs.get("workflow")
        if workflow and hasattr(workflow, "pause_agent"):
            await workflow.pause_agent(agent_id)
            return {"success": True, "agent_id": agent_id, "status": "paused"}

        return {"success": False, "error": "No workflow registered"}

    async def _resume_agent(self, payload: dict) -> dict:
        """Resume a paused agent."""
        agent_id = payload.get("agent_id")
        if not agent_id:
            return {"success": False, "error": "agent_id required"}

        workflow = self._local_refs.get("workflow")
        if workflow and hasattr(workflow, "resume_agent"):
            await workflow.resume_agent(agent_id)
            return {"success": True, "agent_id": agent_id, "status": "resumed"}

        return {"success": False, "error": "No workflow registered"}

    async def _stop_agent(self, payload: dict) -> dict:
        """Stop an agent immediately."""
        agent_id = payload.get("agent_id")
        if not agent_id:
            return {"success": False, "error": "agent_id required"}

        workflow = self._local_refs.get("workflow")
        if workflow and hasattr(workflow, "stop_agent"):
            await workflow.stop_agent(agent_id)
            return {"success": True, "agent_id": agent_id, "status": "stopped"}

        return {"success": False, "error": "No workflow registered"}

    async def _global_stop(self, payload: dict) -> dict:
        """Trigger global stop - stop all agents immediately."""
        registry = self._local_refs.get("circuit_breaker_registry")
        if registry and hasattr(registry, "global_stop"):
            registry.global_stop()
            return {"success": True, "status": "all_stopped"}

        # Fallback: stop via workflow
        workflow = self._local_refs.get("workflow")
        if workflow and hasattr(workflow, "stop_all"):
            await workflow.stop_all()
            return {"success": True, "status": "all_stopped"}

        return {"success": False, "error": "No circuit breaker registry or workflow registered"}

    async def _update_rules(self, payload: dict) -> dict:
        """Update rules live."""
        rules = payload.get("rules", [])

        rules_engine = self._local_refs.get("rules_engine")
        if rules_engine and hasattr(rules_engine, "update_rules"):
            rules_engine.update_rules(rules)
            return {"success": True, "rules_count": len(rules)}

        return {"success": False, "error": "No rules engine registered"}

    async def _update_limits(self, payload: dict) -> dict:
        """Update budget/rate limits live."""
        limits = self._local_refs.get("execution_limits")
        if limits:
            if "max_budget" in payload:
                limits.max_budget = payload["max_budget"]
            if "max_steps" in payload:
                limits.max_steps = payload["max_steps"]
            if "max_time" in payload:
                limits.max_time = payload["max_time"]
            return {"success": True, "limits": payload}

        return {"success": False, "error": "No execution limits registered"}

    async def _rollback(self, payload: dict) -> dict:
        """Rollback to checkpoint or last action."""
        checkpoint_id = payload.get("checkpoint_id")

        checkpoint_manager = self._local_refs.get("checkpoint_manager")
        if checkpoint_manager and hasattr(checkpoint_manager, "restore"):
            state = checkpoint_manager.restore(checkpoint_id)
            if state:
                return {"success": True, "checkpoint_id": checkpoint_id}
            return {"success": False, "error": "Checkpoint not found"}

        return {"success": False, "error": "No checkpoint manager registered"}

    async def _break_loop(self, payload: dict) -> dict:
        """Force break a detected loop."""
        agent_id = payload.get("agent_id")

        loop_detector = self._local_refs.get("loop_detector")
        if loop_detector and hasattr(loop_detector, "force_break"):
            loop_detector.force_break(agent_id)
            return {"success": True, "agent_id": agent_id, "status": "loop_broken"}

        return {"success": False, "error": "No loop detector registered"}

    async def _update_tool_access(self, payload: dict) -> dict:
        """Update tool access permissions live."""
        agent_id = payload.get("agent_id")
        tools = payload.get("tools", [])
        action = payload.get("action", "set")  # set, add, remove

        tool_controller = self._local_refs.get("tool_access_controller")
        if tool_controller:
            if action == "set":
                tool_controller.set_allowed_tools(agent_id, tools)
            elif action == "add":
                for tool in tools:
                    tool_controller.allow_tool(agent_id, tool)
            elif action == "remove":
                for tool in tools:
                    tool_controller.deny_tool(agent_id, tool)
            return {"success": True, "agent_id": agent_id, "tools": tools, "action": action}

        return {"success": False, "error": "No tool access controller registered"}

    # Event handling

    def on(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Remove event callback."""
        if event in self._callbacks:
            self._callbacks[event] = [cb for cb in self._callbacks[event] if cb != callback]

    def _emit(self, event: str, data: dict) -> None:
        """Emit event to callbacks."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception:
                    pass  # Don't let callback errors break the client
