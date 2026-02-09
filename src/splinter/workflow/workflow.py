"""Workflow - orchestrates multiple agents with dependencies."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable

from ..control.limits import LimitsEnforcer, PerAgentLimits
from ..control.loop_detection import LoopDetector
from ..control.memory import MemoryStore
from ..control.tool_access import ToolAccessController, ToolRegistry
from ..coordination.checkpoint import CheckpointManager, InMemoryCheckpointStorage
from ..coordination.ownership import StateOwnershipManager
from ..coordination.schema import HandoffManager
from ..coordination.state import SharedState
from ..exceptions import (
    AgentNotFoundError,
    SchemaValidationError,
    WorkflowExecutionError,
)
from ..gateway.gateway import Gateway
from ..schemas import (
    AgentConfig,
    AgentStatus,
    Checkpoint,
    ExecutionContext,
    ExecutionLimits,
    FieldOwnership,
    HandoffSchema,
    LoopDetectionConfig,
    MemoryConfig,
    WorkflowConfig,
    WorkflowStatus,
    WorkflowStep,
)
from .agent import Agent, AgentBuilder

logger = logging.getLogger(__name__)


class WorkflowResult:
    """Result of a workflow execution."""

    def __init__(
        self,
        workflow_id: str,
        status: WorkflowStatus,
        state: dict[str, Any],
        outputs: dict[str, Any],
        metrics: dict[str, Any],
        error: str | None = None,
    ):
        self.workflow_id = workflow_id
        self.status = status
        self.state = state
        self.outputs = outputs
        self.metrics = metrics
        self.error = error
        self.completed_at = datetime.now()

    @property
    def success(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED


class Workflow:
    """Orchestrates agents with dependencies, state sharing, and checkpointing."""

    def __init__(
        self,
        workflow_id: str | None = None,
        name: str | None = None,
        limits: ExecutionLimits | None = None,
        loop_detection: LoopDetectionConfig | None = None,
        memory_config: MemoryConfig | None = None,
        checkpoint_enabled: bool = True,
        on_step_complete: Callable[[int, str, dict[str, Any]], None] | None = None,
        on_workflow_complete: Callable[[WorkflowResult], None] | None = None,
    ):
        """Initialize a workflow.

        Args:
            workflow_id: Unique workflow identifier. Auto-generated if not provided.
            name: Human-readable workflow name.
            limits: Execution limits.
            loop_detection: Loop detection configuration.
            memory_config: Memory configuration.
            checkpoint_enabled: Whether to enable checkpointing.
            on_step_complete: Callback after each step (step, agent_id, output).
            on_workflow_complete: Callback when workflow completes.
        """
        self._workflow_id = workflow_id or f"wf-{uuid.uuid4().hex[:8]}"
        self._name = name
        self._limits = limits or ExecutionLimits()
        self._loop_detection_config = loop_detection or LoopDetectionConfig()
        self._memory_config = memory_config or MemoryConfig()
        self._checkpoint_enabled = checkpoint_enabled
        self._on_step_complete = on_step_complete
        self._on_workflow_complete = on_workflow_complete

        # Components
        self._gateway: Gateway | None = None
        self._state: SharedState | None = None
        self._memory: MemoryStore | None = None
        self._checkpoint_manager: CheckpointManager | None = None
        self._ownership_manager: StateOwnershipManager | None = None
        self._handoff_manager: HandoffManager | None = None
        self._tool_registry: ToolRegistry | None = None
        self._tool_access: ToolAccessController | None = None

        # Agents and steps
        self._agent_configs: dict[str, AgentConfig] = {}
        self._agents: dict[str, Agent] = {}
        self._steps: list[WorkflowStep] = []

        # Execution state
        self._status = WorkflowStatus.PENDING
        self._current_step = 0
        self._outputs: dict[str, Any] = {}
        self._run_id: str | None = None

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    @property
    def status(self) -> WorkflowStatus:
        return self._status

    @property
    def current_step(self) -> int:
        return self._current_step

    def configure_gateway(self, gateway: Gateway) -> "Workflow":
        """Set the gateway for LLM calls.

        Args:
            gateway: Configured Gateway instance.

        Returns:
            Self for chaining.
        """
        self._gateway = gateway
        return self

    def configure_providers(self, **provider_configs: dict[str, Any]) -> "Workflow":
        """Configure LLM providers.

        Args:
            **provider_configs: Provider configurations.
                e.g., openai={"api_key": "sk-..."}, anthropic={"api_key": "sk-ant-..."}

        Returns:
            Self for chaining.
        """
        if self._gateway is None:
            self._gateway = Gateway(limits=self._limits)

        for provider, config in provider_configs.items():
            self._gateway.configure_provider(provider, **config)

        return self

    def add_agent(self, config: AgentConfig) -> "Workflow":
        """Add an agent to the workflow.

        Args:
            config: Agent configuration.

        Returns:
            Self for chaining.
        """
        self._agent_configs[config.agent_id] = config
        return self

    def add_step(
        self,
        agent_id: str,
        depends_on: list[str] | None = None,
        input_mapping: dict[str, str] | None = None,
        condition: str | None = None,
    ) -> "Workflow":
        """Add an execution step to the workflow.

        Args:
            agent_id: Agent to execute in this step.
            depends_on: List of agent IDs that must complete first.
            input_mapping: Map state fields to agent input.
            condition: Optional condition expression for execution.

        Returns:
            Self for chaining.
        """
        step = WorkflowStep(
            agent_id=agent_id,
            depends_on=depends_on or [],
            input_mapping=input_mapping or {},
            condition=condition,
        )
        self._steps.append(step)
        return self

    def add_handoff_schema(
        self,
        source: str,
        target: str,
        schema: dict[str, Any],
        required_fields: list[str] | None = None,
    ) -> "Workflow":
        """Add a schema validation for handoffs between agents.

        Args:
            source: Source agent ID.
            target: Target agent ID.
            schema: JSON Schema for validation.
            required_fields: Additional required fields.

        Returns:
            Self for chaining.
        """
        if self._handoff_manager is None:
            self._handoff_manager = HandoffManager()

        self._handoff_manager.register_handoff(
            source=source,
            target=target,
            output_schema=schema,
            required_fields=required_fields,
        )
        return self

    def add_state_ownership(self, agent_id: str, fields: list[str]) -> "Workflow":
        """Define state field ownership for an agent.

        Args:
            agent_id: Agent ID.
            fields: Field patterns this agent owns.

        Returns:
            Self for chaining.
        """
        if self._ownership_manager is None:
            self._ownership_manager = StateOwnershipManager()

        self._ownership_manager.set_agent_ownership(agent_id, fields)
        return self

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        handler: Callable[..., Any] | None = None,
        allowed_agents: list[str] | None = None,
    ) -> "Workflow":
        """Register a tool for agents to use.

        Args:
            name: Tool name.
            description: Tool description.
            parameters: JSON Schema for parameters.
            handler: Function to execute the tool.
            allowed_agents: Agents allowed to use this tool.

        Returns:
            Self for chaining.
        """
        if self._tool_registry is None:
            self._tool_registry = ToolRegistry()
        if self._tool_access is None:
            self._tool_access = ToolAccessController()

        self._tool_registry.register(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

        if allowed_agents:
            self._tool_access.register_tool(name, allowed_agents=allowed_agents)

        return self

    def _initialize_components(self) -> None:
        """Initialize all workflow components."""
        # Gateway
        if self._gateway is None:
            self._gateway = Gateway(
                limits=self._limits,
                loop_detection=self._loop_detection_config,
                tool_access=self._tool_access,
            )

        # State
        self._state = SharedState()

        # Memory
        self._memory = MemoryStore(config=self._memory_config)

        # Checkpoint manager
        if self._checkpoint_enabled:
            self._checkpoint_manager = CheckpointManager(
                storage=InMemoryCheckpointStorage(),
            )

        # Ownership manager
        if self._ownership_manager is None:
            self._ownership_manager = StateOwnershipManager()

        # Set up state ownership from agent configs
        for config in self._agent_configs.values():
            if config.state_ownership:
                self._ownership_manager.set_agent_ownership(
                    config.agent_id, config.state_ownership
                )

        # Handoff manager
        if self._handoff_manager is None:
            self._handoff_manager = HandoffManager()

        # Set up output schemas from agent configs
        for config in self._agent_configs.values():
            if config.output_schema:
                self._handoff_manager.register_output_schema(
                    config.agent_id, config.output_schema
                )

        # Tool executor
        def tool_executor(name: str, args: dict[str, Any]) -> Any:
            if self._tool_registry:
                return self._tool_registry.execute(name, **args)
            raise ValueError(f"Tool '{name}' not found")

        # Create agent instances
        for agent_id, config in self._agent_configs.items():
            self._agents[agent_id] = Agent(
                config=config,
                gateway=self._gateway,
                tool_executor=tool_executor if self._tool_registry else None,
            )

    async def run(
        self,
        initial_state: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> WorkflowResult:
        """Run the workflow.

        Args:
            initial_state: Initial state data.
            resume: If True, attempt to resume from checkpoint.

        Returns:
            WorkflowResult with final state and outputs.
        """
        self._run_id = f"run-{uuid.uuid4().hex[:8]}"
        self._initialize_components()

        # Initialize state
        if initial_state:
            self._state.merge(initial_state)

        # Check for resume
        start_step = 0
        if resume and self._checkpoint_manager:
            resume_point = self._checkpoint_manager.get_resume_point(self._workflow_id)
            if resume_point:
                start_step, state_snapshot, metrics = resume_point
                self._state = SharedState(initial_data=state_snapshot.data)
                self._current_step = start_step
                logger.info(f"Resuming workflow from step {start_step}")

        self._status = WorkflowStatus.RUNNING
        self._gateway.start()

        try:
            # Execute each step
            for step_idx, step in enumerate(self._steps):
                if step_idx < start_step:
                    continue

                self._current_step = step_idx

                # Check dependencies
                for dep in step.depends_on:
                    if dep not in self._outputs:
                        raise WorkflowExecutionError(
                            f"Dependency '{dep}' not satisfied",
                            step=step_idx,
                            agent_id=step.agent_id,
                        )

                # Check condition
                if step.condition and not self._evaluate_condition(step.condition):
                    logger.info(f"Skipping step {step_idx} ({step.agent_id}): condition not met")
                    continue

                # Get agent
                agent = self._agents.get(step.agent_id)
                if agent is None:
                    raise AgentNotFoundError(step.agent_id)

                # Build task from input mapping
                task = self._build_task(step)

                # Execute agent
                logger.info(f"Executing step {step_idx}: agent '{step.agent_id}'")
                output = await agent.run(
                    task=task,
                    state=self._state,
                )

                # Validate output schema
                if self._handoff_manager:
                    downstream = self._handoff_manager.get_downstream_agents(step.agent_id)
                    for target in downstream:
                        self._handoff_manager.validate_output(
                            source=step.agent_id,
                            target=target,
                            data=output.get("result"),
                        )

                # Update state with output
                self._update_state(step.agent_id, output)

                # Record output
                self._outputs[step.agent_id] = output

                # Checkpoint
                if self._checkpoint_enabled and self._checkpoint_manager:
                    self._checkpoint_manager.create_checkpoint(
                        workflow_id=self._workflow_id,
                        step=step_idx,
                        agent_id=step.agent_id,
                        status=agent.status,
                        state=self._state,
                        metrics=self._gateway.metrics,
                    )

                # Callback
                if self._on_step_complete:
                    self._on_step_complete(step_idx, step.agent_id, output)

            self._status = WorkflowStatus.COMPLETED

        except Exception as e:
            self._status = WorkflowStatus.FAILED
            logger.error(f"Workflow failed: {e}")

            result = WorkflowResult(
                workflow_id=self._workflow_id,
                status=self._status,
                state=self._state.to_dict(),
                outputs=self._outputs,
                metrics=self._gateway.get_metrics(),
                error=str(e),
            )

            if self._on_workflow_complete:
                self._on_workflow_complete(result)

            raise WorkflowExecutionError(str(e))

        result = WorkflowResult(
            workflow_id=self._workflow_id,
            status=self._status,
            state=self._state.to_dict(),
            outputs=self._outputs,
            metrics=self._gateway.get_metrics(),
        )

        if self._on_workflow_complete:
            self._on_workflow_complete(result)

        return result

    def _build_task(self, step: WorkflowStep) -> str:
        """Build task string from step configuration."""
        if not step.input_mapping:
            return f"Execute your task as agent '{step.agent_id}'."

        # Build task from mapped inputs
        parts = [f"Execute your task as agent '{step.agent_id}'."]
        parts.append("\nInputs:")

        for key, state_path in step.input_mapping.items():
            value = self._state.get(state_path)
            parts.append(f"- {key}: {value}")

        return "\n".join(parts)

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a step condition expression."""
        # Simple condition evaluation
        # In production, use a proper expression evaluator
        try:
            state_dict = self._state.to_dict()
            return eval(condition, {"state": state_dict, "outputs": self._outputs})
        except Exception:
            return False

    def _update_state(self, agent_id: str, output: dict[str, Any]) -> None:
        """Update shared state with agent output."""
        result = output.get("result")
        if result is None:
            return

        # Store under agent's namespace
        if isinstance(result, dict):
            for key, value in result.items():
                field = f"{agent_id}.{key}"
                # Check ownership before writing
                try:
                    self._ownership_manager.check_write(agent_id, field)
                    self._state.set(field, value, agent_id=agent_id)
                except Exception as e:
                    logger.warning(f"Ownership violation: {e}")
        else:
            field = f"{agent_id}.output"
            self._ownership_manager.check_write(agent_id, field)
            self._state.set(field, result, agent_id=agent_id)

    @classmethod
    def from_config(cls, config: WorkflowConfig, gateway: Gateway | None = None) -> "Workflow":
        """Create a workflow from a configuration object.

        Args:
            config: WorkflowConfig with full workflow definition.
            gateway: Optional pre-configured gateway.

        Returns:
            Configured Workflow instance.
        """
        workflow = cls(
            workflow_id=config.workflow_id,
            name=config.name,
            limits=config.limits,
            loop_detection=config.loop_detection,
            memory_config=config.memory_config,
            checkpoint_enabled=config.checkpoint_enabled,
        )

        if gateway:
            workflow.configure_gateway(gateway)

        # Add agents
        for agent_config in config.agents:
            workflow.add_agent(agent_config)

        # Add steps
        for step in config.steps:
            workflow.add_step(
                agent_id=step.agent_id,
                depends_on=step.depends_on,
                input_mapping=step.input_mapping,
                condition=step.condition,
            )

        # Add handoff schemas
        for handoff in config.handoff_schemas:
            workflow.add_handoff_schema(
                source=handoff.source_agent_id,
                target=handoff.target_agent_id,
                schema=handoff.output_schema,
                required_fields=handoff.required_fields,
            )

        # Add state ownership
        for ownership in config.field_ownership:
            workflow.add_state_ownership(
                agent_id=ownership.owner_agent_id,
                fields=[ownership.field_pattern],
            )

        return workflow
