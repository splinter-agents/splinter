"""
CommandHandler - Receive and execute commands from Splinter Cloud.

Enables paid features:
- Pause / resume agents
- Stop agents immediately
- Global stop (emergency)
- Live rule changes
- Live limit changes
- Rollback to checkpoint
- Break loops
- Update tool access
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import time


class CommandType(Enum):
    """Types of commands from cloud."""
    # Agent control
    PAUSE_AGENT = "pause_agent"
    RESUME_AGENT = "resume_agent"
    STOP_AGENT = "stop_agent"
    GLOBAL_STOP = "global_stop"

    # Configuration changes
    UPDATE_RULES = "update_rules"
    UPDATE_LIMITS = "update_limits"
    UPDATE_TOOL_ACCESS = "update_tool_access"
    UPDATE_RETRY = "update_retry"

    # Recovery
    ROLLBACK = "rollback"
    RESUME_CHECKPOINT = "resume_checkpoint"
    BREAK_LOOP = "break_loop"

    # Queries (cloud asking for info)
    GET_STATUS = "get_status"
    GET_STATE = "get_state"


@dataclass
class Command:
    """Command from cloud to execute locally."""
    id: str
    type: CommandType
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0  # seconds

    @classmethod
    def from_dict(cls, data: dict) -> "Command":
        """Create command from dictionary."""
        return cls(
            id=data["id"],
            type=CommandType(data["type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            timeout=data.get("timeout", 30.0),
        )


@dataclass
class CommandResult:
    """Result of executing a command."""
    command_id: str
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "command_id": self.command_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
        }


class CommandHandler:
    """
    Handles commands received from Splinter Cloud.

    Uses WebSocket or long-polling to receive commands in real-time.
    Executes commands locally and sends results back to cloud.
    """

    def __init__(self, config: "CloudConfig", handler: Callable[[Command], Any]):
        self.config = config
        self._handler = handler  # Function to execute commands
        self._running = False
        self._pending_commands: asyncio.Queue[Command] = asyncio.Queue()
        self._results: dict[str, CommandResult] = {}

    async def start(self) -> None:
        """Start command listener."""
        self._running = True
        await asyncio.gather(
            self._poll_commands(),
            self._process_commands(),
        )

    async def stop(self) -> None:
        """Stop command listener."""
        self._running = False

    async def _poll_commands(self) -> None:
        """Poll cloud for pending commands."""
        while self._running:
            try:
                # In production, this would be a WebSocket or long-poll
                # GET {self.config.endpoint}/v1/commands?api_key={key}
                commands = await self._fetch_commands()
                for cmd in commands:
                    await self._pending_commands.put(cmd)
                await asyncio.sleep(0.5)  # Poll interval
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)  # Back off on error

    async def _fetch_commands(self) -> list[Command]:
        """Fetch commands from cloud API."""
        # In production, this would make an HTTP request
        # For now, return empty list (no commands)
        return []

    async def _process_commands(self) -> None:
        """Process pending commands."""
        while self._running:
            try:
                # Wait for command with timeout
                try:
                    command = await asyncio.wait_for(
                        self._pending_commands.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Execute command
                result = await self._execute_command(command)

                # Store result
                self._results[command.id] = result

                # Send result back to cloud
                await self._send_result(result)

            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Log and continue

    async def _execute_command(self, command: Command) -> CommandResult:
        """Execute a single command."""
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._handler(command),
                timeout=command.timeout,
            )

            return CommandResult(
                command_id=command.id,
                success=result.get("success", False),
                result=result,
                execution_time=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            return CommandResult(
                command_id=command.id,
                success=False,
                error="Command timed out",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return CommandResult(
                command_id=command.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _send_result(self, result: CommandResult) -> None:
        """Send command result back to cloud."""
        # In production, this would make an HTTP POST request
        # POST {self.config.endpoint}/v1/commands/{result.command_id}/result
        _ = result  # Stub for now

    def get_result(self, command_id: str) -> Optional[CommandResult]:
        """Get result of a command by ID."""
        return self._results.get(command_id)
