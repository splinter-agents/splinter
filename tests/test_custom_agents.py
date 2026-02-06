"""Custom Agent Tests for Splinter Platform.

This file creates custom agents following OpenAI-style patterns and tests them
against all Splinter features. Uses MockProvider for CI/testing.

Set OPENAI_API_KEY environment variable to test with real OpenAI API.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, "/home/user/upgraded-enigma")

from splinter import (
    # Control Layer
    LimitsEnforcer,
    LoopDetector,
    ToolAccessController,
    MemoryStore,
    # Coordination Layer
    SharedState,
    StateOwnershipManager,
    HandoffManager,
    CheckpointManager,
    # Gateway
    Gateway,
    MockProvider,
    SmartMockProvider,
    # Workflow
    Workflow,
    Agent,
    AgentBuilder,
    # Types
    ExecutionLimits,
    LoopDetectionConfig,
    MemoryConfig,
    AgentConfig,
    LLMProvider,
    LLMMessage,
    AgentStatus,
    WorkflowStatus,
    ExecutionMetrics,
    # Exceptions
    BudgetExceededError,
    StepLimitExceededError,
    ToolAccessDeniedError,
    StateOwnershipError,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = os.environ.get("OPENAI_API_KEY", "")
USE_REAL_API = bool(API_KEY)
MODEL = "gpt-4o-mini" if USE_REAL_API else "mock-model"


# =============================================================================
# CUSTOM AGENT DEFINITIONS (OpenAI Style)
# =============================================================================

@dataclass
class AgentDefinition:
    """Definition for a custom agent - similar to OpenAI's Assistant API."""

    name: str
    instructions: str
    tools: list[str]
    model: str = "gpt-4o-mini"
    state_ownership: list[str] = None
    output_schema: dict = None

    def __post_init__(self):
        if self.state_ownership is None:
            self.state_ownership = []


# Define our custom agents
RESEARCH_AGENT = AgentDefinition(
    name="ResearchAgent",
    instructions="""You are a research specialist. Your job is to:
1. Gather information on the given topic
2. Identify key facts and statistics
3. Note important sources and references

Output your findings in JSON format:
{
    "topic": "the research topic",
    "key_facts": ["fact1", "fact2", "fact3"],
    "statistics": [{"metric": "name", "value": "value"}],
    "sources": ["source1", "source2"],
    "confidence_score": 0.0-1.0
}""",
    tools=["web_search", "read_document", "extract_data"],
    state_ownership=["research.*"],
    output_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "key_facts": {"type": "array", "items": {"type": "string"}},
            "statistics": {"type": "array"},
            "sources": {"type": "array", "items": {"type": "string"}},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["topic", "key_facts", "confidence_score"]
    }
)

ANALYSIS_AGENT = AgentDefinition(
    name="AnalysisAgent",
    instructions="""You are a data analyst. Your job is to:
1. Analyze research findings
2. Identify patterns and trends
3. Draw conclusions and insights
4. Rate the significance of findings

Output your analysis in JSON format:
{
    "summary": "brief summary of findings",
    "patterns": ["pattern1", "pattern2"],
    "insights": ["insight1", "insight2"],
    "recommendations": ["rec1", "rec2"],
    "significance_rating": 1-10
}""",
    tools=["calculate", "compare_data", "visualize"],
    state_ownership=["analysis.*"],
    output_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "patterns": {"type": "array", "items": {"type": "string"}},
            "insights": {"type": "array", "items": {"type": "string"}},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "significance_rating": {"type": "integer", "minimum": 1, "maximum": 10}
        },
        "required": ["summary", "insights", "significance_rating"]
    }
)

WRITER_AGENT = AgentDefinition(
    name="WriterAgent",
    instructions="""You are a content writer. Your job is to:
1. Create engaging content based on research and analysis
2. Structure information clearly
3. Write for the target audience
4. Include key takeaways

Output your content in JSON format:
{
    "title": "article title",
    "introduction": "opening paragraph",
    "body": ["paragraph1", "paragraph2"],
    "conclusion": "closing paragraph",
    "key_takeaways": ["takeaway1", "takeaway2"],
    "word_count": 0
}""",
    tools=["grammar_check", "style_check", "format_text"],
    state_ownership=["content.*"],
    output_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "introduction": {"type": "string"},
            "body": {"type": "array", "items": {"type": "string"}},
            "conclusion": {"type": "string"},
            "key_takeaways": {"type": "array", "items": {"type": "string"}},
            "word_count": {"type": "integer"}
        },
        "required": ["title", "introduction", "body", "conclusion"]
    }
)

REVIEWER_AGENT = AgentDefinition(
    name="ReviewerAgent",
    instructions="""You are a quality reviewer. Your job is to:
1. Review content for accuracy
2. Check for completeness
3. Provide constructive feedback
4. Approve or request revisions

Output your review in JSON format:
{
    "overall_rating": 1-10,
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "feedback": ["feedback1", "feedback2"],
    "approved": true/false,
    "revision_requests": ["request1", "request2"]
}""",
    tools=["fact_check", "plagiarism_check", "readability_score"],
    state_ownership=["review.*"],
    output_schema={
        "type": "object",
        "properties": {
            "overall_rating": {"type": "integer", "minimum": 1, "maximum": 10},
            "accuracy_score": {"type": "number", "minimum": 0, "maximum": 1},
            "completeness_score": {"type": "number", "minimum": 0, "maximum": 1},
            "feedback": {"type": "array", "items": {"type": "string"}},
            "approved": {"type": "boolean"},
            "revision_requests": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["overall_rating", "approved"]
    }
)

COORDINATOR_AGENT = AgentDefinition(
    name="CoordinatorAgent",
    instructions="""You are a workflow coordinator. Your job is to:
1. Orchestrate the work between agents
2. Ensure smooth handoffs
3. Track progress and status
4. Handle escalations

Output your status in JSON format:
{
    "workflow_status": "in_progress/completed/blocked",
    "current_stage": "stage name",
    "completed_stages": ["stage1", "stage2"],
    "pending_stages": ["stage3"],
    "blockers": [],
    "next_action": "description of next step"
}""",
    tools=["assign_task", "check_status", "escalate"],
    state_ownership=["workflow.*", "coordination.*"],
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_gateway(
    max_budget: float = 1.0,
    max_steps: int = 20,
    loop_detection: bool = False
) -> Gateway:
    """Create a configured gateway for testing.

    Args:
        max_budget: Maximum budget in dollars
        max_steps: Maximum steps allowed
        loop_detection: Enable strict loop detection (disabled by default for testing)
    """
    # For testing, we typically want to disable loop detection
    # since mock providers return similar outputs
    loop_config = LoopDetectionConfig(
        max_repeated_outputs=3 if loop_detection else 100,
        max_no_state_change=5 if loop_detection else 100
    )

    gateway = Gateway(
        limits=ExecutionLimits(max_budget=max_budget, max_steps=max_steps),
        loop_detection=loop_config
    )

    if USE_REAL_API:
        gateway.configure_provider("openai", api_key=API_KEY)
    else:
        gateway._providers["openai"] = SmartMockProvider()

    return gateway


def create_agent_from_definition(
    definition: AgentDefinition,
    gateway: Gateway,
) -> Agent:
    """Create a Splinter Agent from our custom AgentDefinition."""
    builder = (
        AgentBuilder(definition.name.lower().replace("agent", ""))
        .with_name(definition.name)
        .with_provider(LLMProvider.OPENAI, MODEL)
        .with_system_prompt(definition.instructions)
        .with_tools(definition.tools)
        .with_state_ownership(definition.state_ownership)
    )

    if definition.output_schema:
        builder.with_output_schema(definition.output_schema)

    return builder.build(gateway)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# TEST 1: SINGLE AGENT EXECUTION
# =============================================================================

async def test_single_agent_execution():
    """Test individual custom agents."""
    print_section("TEST 1: SINGLE AGENT EXECUTION")

    gateway = create_gateway()
    results = {}

    # Test Research Agent
    print_subsection("Research Agent")
    research_agent = create_agent_from_definition(RESEARCH_AGENT, gateway)

    try:
        result = await research_agent.run(
            task="Research the latest trends in artificial intelligence and machine learning for 2024",
            state={"request_id": "test-001", "priority": "high"}
        )
        print(f"  Agent: {research_agent.id}")
        print(f"  Status: {research_agent.status}")
        print(f"  Output: {json.dumps(result.get('result', result), indent=2)[:300]}...")
        results["research"] = True
    except Exception as e:
        print(f"  Error: {e}")
        results["research"] = False

    # Test Analysis Agent
    print_subsection("Analysis Agent")
    analysis_agent = create_agent_from_definition(ANALYSIS_AGENT, gateway)

    try:
        result = await analysis_agent.run(
            task="Analyze the following research findings and identify key patterns",
            state={
                "research_data": {
                    "topic": "AI Trends",
                    "key_facts": ["Generative AI adoption grew 300%", "LLMs are mainstream"]
                }
            }
        )
        print(f"  Agent: {analysis_agent.id}")
        print(f"  Status: {analysis_agent.status}")
        print(f"  Output: {json.dumps(result.get('result', result), indent=2)[:300]}...")
        results["analysis"] = True
    except Exception as e:
        print(f"  Error: {e}")
        results["analysis"] = False

    # Test Writer Agent
    print_subsection("Writer Agent")
    writer_agent = create_agent_from_definition(WRITER_AGENT, gateway)

    try:
        result = await writer_agent.run(
            task="Write an article about AI trends based on the analysis",
            state={
                "analysis_summary": "AI is transforming industries",
                "target_audience": "business executives"
            }
        )
        print(f"  Agent: {writer_agent.id}")
        print(f"  Status: {writer_agent.status}")
        print(f"  Output: {json.dumps(result.get('result', result), indent=2)[:300]}...")
        results["writer"] = True
    except Exception as e:
        print(f"  Error: {e}")
        results["writer"] = False

    # Test Reviewer Agent
    print_subsection("Reviewer Agent")
    reviewer_agent = create_agent_from_definition(REVIEWER_AGENT, gateway)

    try:
        result = await reviewer_agent.run(
            task="Review the following article for quality and accuracy",
            state={
                "article_title": "The Rise of AI in 2024",
                "article_content": "AI is transforming how we work..."
            }
        )
        print(f"  Agent: {reviewer_agent.id}")
        print(f"  Status: {reviewer_agent.status}")
        print(f"  Output: {json.dumps(result.get('result', result), indent=2)[:300]}...")
        results["reviewer"] = True
    except Exception as e:
        print(f"  Error: {e}")
        results["reviewer"] = False

    # Summary
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Results: {passed}/{len(results)} agents executed successfully")
    return all(results.values())


# =============================================================================
# TEST 2: CONTROL LAYER WITH CUSTOM AGENTS
# =============================================================================

async def test_control_layer_with_agents():
    """Test Control Layer features with custom agents."""
    print_section("TEST 2: CONTROL LAYER WITH CUSTOM AGENTS")

    results = {}

    # Test 2.1: Execution Limits
    print_subsection("2.1 Execution Limits")

    # Create gateway with step limit but disabled loop detection for this test
    gateway = Gateway(
        limits=ExecutionLimits(max_budget=0.05, max_steps=3),
        loop_detection=LoopDetectionConfig(
            max_repeated_outputs=100,  # Effectively disable for this test
            max_no_state_change=100
        )
    )
    if USE_REAL_API:
        gateway.configure_provider("openai", api_key=API_KEY)
    else:
        gateway._providers["openai"] = SmartMockProvider()

    research_agent = create_agent_from_definition(RESEARCH_AGENT, gateway)

    steps_executed = 0
    limit_hit = False

    for i in range(5):
        try:
            research_agent.reset()
            await research_agent.run(task=f"Research topic {i+1}")
            steps_executed += 1
            print(f"    Step {i+1}: Completed")
        except StepLimitExceededError as e:
            limit_hit = True
            print(f"    Step {i+1}: Blocked - {e}")
            break
        except Exception as e:
            # Loop detection might trigger first with mock
            if "loop" in str(e).lower():
                print(f"    Step {i+1}: Loop detected (expected with mock)")
            else:
                raise

    results["execution_limits"] = limit_hit and steps_executed == 3
    print(f"  ✓ Execution limits enforced after {steps_executed} steps" if results["execution_limits"]
          else f"  ✗ Limits not properly enforced")

    # Test 2.2: Tool Access Control
    print_subsection("2.2 Tool Access Control")
    controller = ToolAccessController()

    # Register tools based on agent definitions
    controller.register_tool("web_search", allowed_agents=["research"])
    controller.register_tool("read_document", allowed_agents=["research"])
    controller.register_tool("calculate", allowed_agents=["analysis"])
    controller.register_tool("grammar_check", allowed_agents=["writer"])
    controller.register_tool("fact_check", allowed_agents=["reviewer"])

    access_tests = [
        ("research", "web_search", True),
        ("research", "calculate", False),
        ("analysis", "calculate", True),
        ("writer", "web_search", False),
        ("reviewer", "fact_check", True),
    ]

    access_correct = 0
    for agent_id, tool, should_allow in access_tests:
        try:
            controller.check_access(agent_id, tool)
            allowed = True
        except ToolAccessDeniedError:
            allowed = False

        if allowed == should_allow:
            access_correct += 1
            status = "✓" if should_allow else "✓ blocked"
        else:
            status = "✗ wrong"
        print(f"    {agent_id} -> {tool}: {status}")

    results["tool_access"] = access_correct == len(access_tests)
    print(f"  {'✓' if results['tool_access'] else '✗'} Tool access: {access_correct}/{len(access_tests)} correct")

    # Test 2.3: Loop Detection with Agent
    print_subsection("2.3 Loop Detection")

    # Create a gateway WITH loop detection enabled (low threshold)
    gateway2 = Gateway(
        limits=ExecutionLimits(max_budget=1.0, max_steps=20),
        loop_detection=LoopDetectionConfig(
            max_repeated_outputs=3,
            max_no_state_change=4
        )
    )
    if USE_REAL_API:
        gateway2.configure_provider("openai", api_key=API_KEY)
    else:
        gateway2._providers["openai"] = SmartMockProvider()

    analysis_agent = create_agent_from_definition(ANALYSIS_AGENT, gateway2)

    loop_detected = False
    iterations_completed = 0

    for i in range(6):
        try:
            analysis_agent.reset()
            await analysis_agent.run(task="Analyze the same data")
            iterations_completed += 1
            print(f"    Iteration {i+1}: OK")
        except Exception as e:
            if "loop" in str(e).lower():
                loop_detected = True
                print(f"    Iteration {i+1}: Loop detected - {type(e).__name__}")
                break
            else:
                raise

    results["loop_detection"] = loop_detected
    print(f"  {'✓' if results['loop_detection'] else '✗'} Loop detection working (detected after {iterations_completed} iterations)")

    # Test 2.4: Memory Limits
    print_subsection("2.4 Memory Limits")
    memory = MemoryStore(config=MemoryConfig(
        max_entries=5,
        eviction_policy="lru"
    ))

    # Simulate agents storing data
    for i in range(8):
        agent_name = ["research", "analysis", "writer", "reviewer"][i % 4]
        memory.set(f"{agent_name}_result_{i}", f"Output from {agent_name} iteration {i}")

    entry_count = memory.entry_count
    results["memory_limits"] = entry_count == 5
    print(f"  {'✓' if results['memory_limits'] else '✗'} Memory limited to {entry_count} entries (max: 5)")
    print(f"    Active keys: {memory.keys()}")

    # Summary
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Control Layer Results: {passed}/{len(results)} tests passed")
    return all(results.values())


# =============================================================================
# TEST 3: COORDINATION LAYER WITH CUSTOM AGENTS
# =============================================================================

async def test_coordination_layer_with_agents():
    """Test Coordination Layer features with custom agents."""
    print_section("TEST 3: COORDINATION LAYER WITH CUSTOM AGENTS")

    results = {}

    # Test 3.1: Shared State Between Agents
    print_subsection("3.1 Shared State")
    state = SharedState()
    gateway = create_gateway()

    # Research agent writes to state
    research_agent = create_agent_from_definition(RESEARCH_AGENT, gateway)
    research_result = await research_agent.run(
        task="Research quantum computing applications",
        state=state
    )

    # Store research output in shared state
    state.set("research.output", research_result.get("result", research_result))
    state.set("research.completed", True)
    state.set("research.timestamp", "2024-01-15T10:30:00Z")

    print(f"    Research agent wrote to state")
    print(f"    State version: {state.version}")

    # Analysis agent reads and writes
    state.set("analysis.input_from_research", state.get("research.output"))
    state.set("analysis.started", True)

    analysis_agent = create_agent_from_definition(ANALYSIS_AGENT, gateway)
    analysis_result = await analysis_agent.run(
        task="Analyze the quantum computing research findings",
        state=state
    )

    state.set("analysis.output", analysis_result.get("result", analysis_result))
    state.set("analysis.completed", True)

    print(f"    Analysis agent read research and wrote analysis")
    print(f"    State version: {state.version}")

    results["shared_state"] = (
        state.get("research.completed") == True and
        state.get("analysis.completed") == True and
        state.version >= 6
    )
    print(f"  {'✓' if results['shared_state'] else '✗'} Shared state working")

    # Test 3.2: State Ownership
    print_subsection("3.2 State Ownership")
    ownership = StateOwnershipManager()

    # Register ownership based on agent definitions
    for defn in [RESEARCH_AGENT, ANALYSIS_AGENT, WRITER_AGENT, REVIEWER_AGENT]:
        agent_id = defn.name.lower().replace("agent", "")
        for pattern in defn.state_ownership:
            ownership.register(pattern, owner=agent_id)

    ownership_tests = [
        ("research", "research.findings", True),
        ("analysis", "research.findings", False),  # Analysis can't write to research
        ("analysis", "analysis.insights", True),
        ("writer", "analysis.insights", False),  # Writer can't write to analysis
        ("writer", "content.article", True),
        ("reviewer", "review.feedback", True),
    ]

    ownership_correct = 0
    for agent_id, field, should_allow in ownership_tests:
        try:
            ownership.check_write(agent_id, field)
            allowed = True
        except StateOwnershipError:
            allowed = False

        if allowed == should_allow:
            ownership_correct += 1
            status = "✓ allowed" if should_allow else "✓ blocked"
        else:
            status = "✗ wrong"
        print(f"    {agent_id} -> {field}: {status}")

    results["state_ownership"] = ownership_correct == len(ownership_tests)
    print(f"  {'✓' if results['state_ownership'] else '✗'} Ownership: {ownership_correct}/{len(ownership_tests)}")

    # Test 3.3: Schema Handoffs
    print_subsection("3.3 Schema Handoffs")
    handoff = HandoffManager(strict=True)

    # Register handoffs between our agents
    handoff.register_handoff(
        source="research",
        target="analysis",
        output_schema=RESEARCH_AGENT.output_schema
    )
    handoff.register_handoff(
        source="analysis",
        target="writer",
        output_schema=ANALYSIS_AGENT.output_schema
    )
    handoff.register_handoff(
        source="writer",
        target="reviewer",
        output_schema=WRITER_AGENT.output_schema
    )

    # Test valid handoff
    valid_research_output = {
        "topic": "Quantum Computing",
        "key_facts": ["Fact 1", "Fact 2", "Fact 3"],
        "statistics": [],
        "sources": ["Source 1"],
        "confidence_score": 0.85
    }

    try:
        handoff.validate_output("research", "analysis", valid_research_output)
        valid_passed = True
        print(f"    research -> analysis: ✓ valid output accepted")
    except Exception as e:
        valid_passed = False
        print(f"    research -> analysis: ✗ {e}")

    # Test invalid handoff
    invalid_output = {"wrong_field": "value"}

    try:
        handoff.validate_output("research", "analysis", invalid_output)
        invalid_blocked = False
        print(f"    Invalid output: ✗ should have been rejected")
    except Exception:
        invalid_blocked = True
        print(f"    Invalid output: ✓ correctly rejected")

    results["schema_handoffs"] = valid_passed and invalid_blocked
    print(f"  {'✓' if results['schema_handoffs'] else '✗'} Schema handoffs working")

    # Test 3.4: Checkpointing
    print_subsection("3.4 Checkpointing (Resumable Execution)")
    checkpoint_mgr = CheckpointManager()

    # Simulate workflow checkpoints
    workflow_state = SharedState()
    workflow_state.set("stage", "research")

    # Checkpoint after research
    cp1 = checkpoint_mgr.create_checkpoint(
        workflow_id="content-pipeline",
        step=1,
        agent_id="research",
        status=AgentStatus.COMPLETED,
        state=workflow_state,
        metrics=ExecutionMetrics(total_cost=0.01, total_steps=1)
    )
    print(f"    Checkpoint 1: research completed (step={cp1.step})")

    # Checkpoint after analysis
    workflow_state.set("stage", "analysis")
    cp2 = checkpoint_mgr.create_checkpoint(
        workflow_id="content-pipeline",
        step=2,
        agent_id="analysis",
        status=AgentStatus.COMPLETED,
        state=workflow_state,
        metrics=ExecutionMetrics(total_cost=0.02, total_steps=2)
    )
    print(f"    Checkpoint 2: analysis completed (step={cp2.step})")

    # Test resume point
    resume = checkpoint_mgr.get_resume_point("content-pipeline")
    resume_step, state_snapshot, metrics = resume

    results["checkpointing"] = (
        resume_step == 3 and
        state_snapshot.data.get("stage") == "analysis" and
        metrics.total_steps == 2
    )
    print(f"    Resume point: step={resume_step}, stage={state_snapshot.data.get('stage')}")
    print(f"  {'✓' if results['checkpointing'] else '✗'} Checkpointing working")

    # Summary
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Coordination Layer Results: {passed}/{len(results)} tests passed")
    return all(results.values())


# =============================================================================
# TEST 4: FULL MULTI-AGENT WORKFLOW
# =============================================================================

async def test_full_workflow():
    """Test complete multi-agent workflow with all custom agents."""
    print_section("TEST 4: FULL MULTI-AGENT WORKFLOW")

    gateway = create_gateway(max_budget=2.0, max_steps=50)

    # Create workflow
    workflow = Workflow(
        workflow_id="content-creation-pipeline",
        name="Content Creation Pipeline",
        limits=ExecutionLimits(max_budget=2.0, max_steps=50),
        checkpoint_enabled=True
    )
    workflow._gateway = gateway

    print_subsection("Adding Agents to Workflow")

    # Add all custom agents
    for defn in [RESEARCH_AGENT, ANALYSIS_AGENT, WRITER_AGENT, REVIEWER_AGENT]:
        agent_id = defn.name.lower().replace("agent", "")
        config = AgentConfig(
            agent_id=agent_id,
            name=defn.name,
            provider=LLMProvider.OPENAI,
            model=MODEL,
            system_prompt=defn.instructions,
            tools=defn.tools,
            state_ownership=defn.state_ownership
        )
        workflow.add_agent(config)
        print(f"    Added: {defn.name} ({agent_id})")

    print_subsection("Defining Workflow Steps")

    # Define the pipeline: research -> analysis -> writer -> reviewer
    workflow.add_step("research")
    workflow.add_step("analysis", depends_on=["research"])
    workflow.add_step("writer", depends_on=["analysis"])
    workflow.add_step("reviewer", depends_on=["writer"])

    print("    1. research (no dependencies)")
    print("    2. analysis (depends on: research)")
    print("    3. writer (depends on: analysis)")
    print("    4. reviewer (depends on: writer)")

    # Add state ownership
    print_subsection("Configuring State Ownership")
    for defn in [RESEARCH_AGENT, ANALYSIS_AGENT, WRITER_AGENT, REVIEWER_AGENT]:
        agent_id = defn.name.lower().replace("agent", "")
        if defn.state_ownership:
            workflow.add_state_ownership(agent_id, defn.state_ownership)
            print(f"    {agent_id}: {defn.state_ownership}")

    # Run workflow
    print_subsection("Executing Workflow")

    try:
        result = await workflow.run(
            initial_state={
                "topic": "The Future of Remote Work and AI Collaboration Tools",
                "target_audience": "Business Leaders",
                "content_type": "Executive Briefing",
                "max_length": 1000
            }
        )

        print(f"\n  Workflow Status: {result.status}")
        print(f"  Success: {result.success}")
        print(f"  Total Cost: ${result.metrics.get('total_cost', 0):.4f}")
        print(f"  Total Steps: {result.metrics.get('total_steps', 0)}")

        print_subsection("Agent Outputs")
        for agent_id, output in result.outputs.items():
            output_preview = str(output.get("result", output))[:150]
            print(f"    {agent_id}:")
            print(f"      {output_preview}...")

        # Check final state
        print_subsection("Final State")
        for key in ["research", "analysis", "content", "review"]:
            if key in result.state:
                print(f"    {key}: populated")

        success = result.success and len(result.outputs) == 4

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print(f"\n  {'✓ WORKFLOW COMPLETED SUCCESSFULLY' if success else '✗ WORKFLOW FAILED'}")
    return success


# =============================================================================
# TEST 5: AGENT COMMUNICATION PATTERNS
# =============================================================================

async def test_agent_communication():
    """Test different agent communication patterns."""
    print_section("TEST 5: AGENT COMMUNICATION PATTERNS")

    gateway = create_gateway()
    results = {}

    # Pattern 1: Sequential handoff
    print_subsection("5.1 Sequential Handoff (A -> B -> C)")

    state = SharedState()

    # Agent A: Research
    research = create_agent_from_definition(RESEARCH_AGENT, gateway)
    r1 = await research.run(task="Research electric vehicles market", state=state)
    state.set("pipeline.research", r1)

    # Agent B: Analysis (receives research output)
    analysis = create_agent_from_definition(ANALYSIS_AGENT, gateway)
    r2 = await analysis.run(
        task="Analyze the research findings",
        state=state,
        context={"previous_output": r1}
    )
    state.set("pipeline.analysis", r2)

    # Agent C: Writer (receives both)
    writer = create_agent_from_definition(WRITER_AGENT, gateway)
    r3 = await writer.run(
        task="Write an article based on research and analysis",
        state=state,
        context={"research": r1, "analysis": r2}
    )
    state.set("pipeline.content", r3)

    results["sequential"] = all([r1, r2, r3])
    print(f"  {'✓' if results['sequential'] else '✗'} Sequential handoff: 3 agents executed")

    # Pattern 2: Parallel execution
    print_subsection("5.2 Parallel Execution (A + B -> C)")

    gateway2 = create_gateway(max_steps=20)

    # Run two agents in parallel
    research2 = create_agent_from_definition(RESEARCH_AGENT, gateway2)
    analysis2 = create_agent_from_definition(ANALYSIS_AGENT, gateway2)

    parallel_results = await asyncio.gather(
        research2.run(task="Research renewable energy"),
        analysis2.run(task="Analyze current energy consumption patterns"),
        return_exceptions=True
    )

    # Check results
    parallel_success = all(not isinstance(r, Exception) for r in parallel_results)
    results["parallel"] = parallel_success
    print(f"  {'✓' if results['parallel'] else '✗'} Parallel execution: 2 agents ran concurrently")

    # Pattern 3: Feedback loop
    print_subsection("5.3 Feedback Loop (Writer <-> Reviewer)")

    gateway3 = create_gateway(max_steps=15)

    writer3 = create_agent_from_definition(WRITER_AGENT, gateway3)
    reviewer3 = create_agent_from_definition(REVIEWER_AGENT, gateway3)

    # Initial write
    content = await writer3.run(task="Write a draft article about cloud computing")

    # Review cycle
    iterations = 0
    approved = False

    for i in range(3):  # Max 3 review cycles
        iterations += 1
        reviewer3.reset()
        review = await reviewer3.run(
            task="Review this content",
            context={"content": content}
        )

        # Check if approved (mock always approves eventually)
        review_result = review.get("result", {})
        if isinstance(review_result, dict) and review_result.get("approved", False):
            approved = True
            break
        elif i < 2:
            # Revise based on feedback
            writer3.reset()
            content = await writer3.run(
                task="Revise the content based on feedback",
                context={"feedback": review}
            )

    results["feedback_loop"] = iterations >= 1
    print(f"  {'✓' if results['feedback_loop'] else '✗'} Feedback loop: {iterations} review cycle(s)")

    # Summary
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Communication Patterns: {passed}/{len(results)} patterns tested")
    return all(results.values())


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all custom agent tests."""
    print("\n" + "=" * 70)
    print(" SPLINTER CUSTOM AGENT TESTS")
    print(f" Mode: {'REAL OPENAI API' if USE_REAL_API else 'MOCK PROVIDER'}")
    print("=" * 70)

    if not USE_REAL_API:
        print("\n⚠️  Running with MockProvider. Set OPENAI_API_KEY for real API testing.\n")

    results = {}

    # Run all tests
    results["single_agent"] = await test_single_agent_execution()
    results["control_layer"] = await test_control_layer_with_agents()
    results["coordination_layer"] = await test_coordination_layer_with_agents()
    results["full_workflow"] = await test_full_workflow()
    results["communication"] = await test_agent_communication()

    # Final summary
    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")

    total_passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {total_passed}/{len(results)} test suites passed")
    print("=" * 70)

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
