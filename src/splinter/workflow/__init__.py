"""Splinter Workflow Layer.

The Workflow Layer provides the main orchestration for multi-agent systems.

Components:
- agent: Agent definition and execution
- workflow: Workflow orchestration and execution
"""

from .agent import Agent, AgentBuilder
from .workflow import Workflow, WorkflowResult

__all__ = [
    "Agent",
    "AgentBuilder",
    "Workflow",
    "WorkflowResult",
]
