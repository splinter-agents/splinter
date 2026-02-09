# Splinter

**Control and coordination infrastructure for multi-agent AI systems.**

```bash
pip install splinter
```

```python
from splinter import Splinter

s = Splinter(openai_key="sk-...", max_budget=5.0)
result = await s.run("agent", "Do the task")
# Automatically stops at $5 - no runaway costs
```

No Docker. No config files. Just pip install and go.

---

## 📋 Quick Reference

> All objects, organized by layer. Local = free, Cloud = paid.

### Core

| Object | What it does |
|--------|--------------|
| `Splinter` | Simple API - create and run agents in one line |
| `Workflow` | Run multiple agents with dependencies |
| `Agent` | Single AI entity with config |
| `AgentConfig` | Configuration for an agent |

### 🛡️ Control Layer (Local)

| Object | What it does |
|--------|--------------|
| `ExecutionLimits` | Budget ($), step count, time limits |
| `LoopDetectionConfig` | Catch infinite loops |
| `ToolAccessController` | Which agent can use which tools |
| `RateLimiter` | Max calls per minute per agent/tool |
| `CircuitBreaker` | Stop after N failures |
| `CircuitBreakerRegistry` | Manage all breakers, emergency stop |
| `DecisionEnforcer` | Lock decisions so agents can't flip-flop |
| `RetryStrategy` | Retry failed calls with backoff |
| `RulesEngine` | Custom BLOCK/WARN/LOG rules |
| `MemoryStore` | Capped memory with auto-eviction |

### 🤝 Coordination Layer (Local)

| Object | What it does |
|--------|--------------|
| `SharedState` | Single source of truth for all agents |
| `StateOwnership` | Who can write to which fields |
| `CheckpointManager` | Save progress, resume after crash |
| `SchemaValidator` | Validate agent outputs |
| `HandoffManager` | Validate data between agents |
| `ChainContext` | Agents see what happened before them |
| `GoalTracker` | Track progress toward goals |
| `ActionEligibility` | Which agent can act right now |
| `CompletionTracker` | Agents must say "I'm done" |
| `WaitTracker` | Track why agents are idle |

### ☁️ Splinter Cloud (Paid)

| Feature | What it does |
|---------|--------------|
| **Live Control** | Pause, resume, stop agents remotely |
| **Global Stop** | Emergency stop all agents instantly |
| **Live Rules** | Change rules without redeploying |
| **Live Limits** | Modify budgets/rate limits on the fly |
| **Tool Access** | Update permissions in real-time |
| **Break Loops** | Force-break detected loops |
| **Rollback** | Resume from safe checkpoint |
| **Dashboard** | Live view of all agents, state, handoffs |
| **Status View** | See active, waiting, blocked, eligible agents |
| **Deadlock Detection** | Automatically surface coordination stalls |
| **Bottleneck Analysis** | Find why agents are waiting |

---

## What is Splinter?

```
┌─────────────────────────────────────────┐
│            YOUR AGENTS                  │
│   Agent 1    Agent 2    Agent 3  ...    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         SPLINTER (Local - Free)         │
│                                         │
│  🛡️ CONTROL         🤝 COORDINATION     │
│  ───────────       ───────────────      │
│  Budget limits     Shared state         │
│  Rate limiting     Checkpointing        │
│  Circuit breakers  Schema validation    │
│  Decision locks    Goal tracking        │
│  Loop detection    Action eligibility   │
│                                         │
└─────────────────────┬───────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   OpenAI / Claude         ☁️ Splinter Cloud
   Gemini / Grok              (Paid API Key)
                              │
                              ├── Live Dashboard
                              ├── Remote Control
                              ├── Deadlock Detection
                              └── Bottleneck Analysis
```

**Local (Free)** = Control + Coordination. Runs entirely on your machine.

**Cloud (Paid)** = Live observability + remote control. Add API key to enable.

---

## 📦 Installation

```bash
pip install splinter
```

Then install your LLM provider:

```bash
pip install openai              # OpenAI
pip install anthropic           # Claude
pip install google-generativeai # Gemini
pip install openai              # Grok (uses OpenAI SDK)
```

---

## 🚀 Quick Start

**Step 1** — Create with limits

```python
from splinter import Splinter

s = Splinter(
    openai_key="sk-...",
    max_budget=5.0,   # Stop at $5
    max_steps=50,     # Stop after 50 calls
)
```

**Step 2** — Run

```python
result = await s.run("researcher", "Find the top 3 AI trends")
```

**Step 3** — Check spend

```python
print(f"Cost: ${s.cost:.4f} | Steps: {s.steps}")
```

Done. Your agent is protected.

---

## ☁️ Splinter Cloud (Paid)

Add an API key to unlock live control and observability.

**Connect:**

```python
from splinter import Splinter

# Option 1: Pass API key directly
s = Splinter(
    openai_key="sk-...",
    api_key="sk-splinter-...",  # Cloud API key
)

# Option 2: Connect later
s = Splinter(openai_key="sk-...")
await s.connect_cloud(api_key="sk-splinter-...")

# Option 3: Environment variable
# export SPLINTER_API_KEY="sk-splinter-..."
s = Splinter(openai_key="sk-...")  # Auto-connects
```

**What you get:**

| Feature | Description |
|---------|-------------|
| **Live Dashboard** | See all agents, shared state, ownership, checkpoints, handoffs in real-time |
| **Pause/Resume** | Pause any agent, resume when ready |
| **Stop Immediately** | Stop agents without redeploying |
| **Global Stop** | Emergency stop all agents at once |
| **Change Rules Live** | Update rules without restart |
| **Change Limits Live** | Modify budgets, rate limits on the fly |
| **Update Tool Access** | Change permissions in real-time |
| **Break Loops** | Force-break detected loops |
| **Rollback** | Resume from any checkpoint |
| **Status View** | See which agents are active, waiting, blocked, or eligible |
| **Deadlock Detection** | Automatically surface coordination stalls |
| **Bottleneck Analysis** | Understand why agents are waiting |

**Check connection:**

```python
if s.is_cloud_connected:
    print("Connected to Splinter Cloud")
```

---

## 🧱 Core Objects

<details>
<summary><b>Splinter</b> — Simple API</summary>

The easiest way. Creates everything internally.

```python
from splinter import Splinter

s = Splinter(openai_key="sk-...", max_budget=10.0)
result = await s.run("researcher", "Find AI trends")
print(f"Cost: ${s.cost:.4f}")
```

</details>

<details>
<summary><b>Workflow</b> — Multi-agent orchestration</summary>

Run multiple agents with dependencies and shared limits.

```python
from splinter.workflow import Workflow
from splinter.types import AgentConfig, ExecutionLimits, LLMProvider

workflow = Workflow(
    workflow_id="pipeline",
    limits=ExecutionLimits(max_budget=20.0),
    checkpoint_enabled=True,
)

workflow.add_agent(AgentConfig(
    agent_id="researcher",
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    system_prompt="Research topics. Output JSON.",
))

workflow.add_agent(AgentConfig(
    agent_id="writer",
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    system_prompt="Write articles. Output JSON.",
))

workflow.add_step("researcher")
workflow.add_step("writer", depends_on=["researcher"])

result = await workflow.run(initial_state={"topic": "AI"})
```

</details>

<details>
<summary><b>Agent</b> — Single AI entity</summary>

Build agents with the fluent API.

```python
from splinter.workflow import AgentBuilder
from splinter.types import LLMProvider

agent = (
    AgentBuilder("researcher")
    .with_provider(LLMProvider.OPENAI, "gpt-4o")
    .with_system_prompt("Research topics. Output JSON.")
    .with_tools(["web_search"])
    .with_state_ownership(["research.*"])
    .build(gateway)
)

result = await agent.run(task="Research AI trends")
```

</details>

---

## 🛡️ Control Objects

> Import from `splinter.control` or `splinter.types`

<details>
<summary><b>ExecutionLimits</b> — Budget, steps, time</summary>

```python
from splinter.types import ExecutionLimits

limits = ExecutionLimits(
    max_budget=10.0,       # Stop at $10
    max_steps=100,         # Stop after 100 calls
    max_time_seconds=600,  # Stop after 10 min
)
```

Use with `Workflow`:
```python
workflow = Workflow(workflow_id="x", limits=limits)
```

</details>

<details>
<summary><b>LoopDetectionConfig</b> — Break infinite loops</summary>

```python
from splinter.types import LoopDetectionConfig

config = LoopDetectionConfig(
    max_repeated_outputs=3,  # Same output 3x = loop
    max_no_state_change=5,   # No change 5x = loop
)

workflow = Workflow(workflow_id="x", loop_detection=config)
# Raises LoopDetectedError when stuck
```

</details>

<details>
<summary><b>ToolAccessController</b> — Per-agent permissions</summary>

```python
from splinter.control import ToolAccessController

ctrl = ToolAccessController()
ctrl.set_allowed_tools("researcher", ["web_search", "read_file"])
ctrl.set_allowed_tools("writer", ["write_file"])

ctrl.check_access("researcher", "web_search")   # ✓
ctrl.check_access("researcher", "delete_file")  # ✗ ToolAccessDeniedError
```

</details>

<details>
<summary><b>RateLimiter</b> — Calls per minute</summary>

```python
from splinter.control import RateLimiter

limiter = RateLimiter()
limiter.set_agent_limit("researcher", calls=10, window_seconds=60)
limiter.set_tool_limit("web_search", calls=20, window_seconds=60)

limiter.check_agent("researcher")        # Raises if over limit
limiter.record_agent_call("researcher")  # Record the call
```

</details>

<details>
<summary><b>CircuitBreaker</b> — Stop on failures</summary>

```python
from splinter.control import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(
    breaker_id="openai",
    config=CircuitBreakerConfig(
        failure_threshold=5,  # Open after 5 fails
        timeout_seconds=60,   # Retry after 60s
    )
)

breaker.check()  # Raises CircuitOpenError if open

try:
    result = await call()
    breaker.record_success()
except:
    breaker.record_failure()
```

**Global emergency stop:**
```python
from splinter.control import CircuitBreakerRegistry

registry = CircuitBreakerRegistry()
registry.register("openai", config)
registry.register("anthropic", config)
registry.trip_all("Emergency!")  # Stop everything
```

</details>

<details>
<summary><b>DecisionEnforcer</b> — Lock decisions</summary>

```python
from splinter.control import DecisionEnforcer, DecisionType

enforcer = DecisionEnforcer(auto_lock=True)

enforcer.record_decision(
    decision_id="strategy",
    agent_id="planner",
    decision_type=DecisionType.STRATEGY,
    value="parallel",
)

# Can't change now!
enforcer.record_decision(
    decision_id="strategy",
    agent_id="planner",
    decision_type=DecisionType.STRATEGY,
    value="sequential",  # Raises DecisionLockError
)
```

</details>

<details>
<summary><b>RetryStrategy</b> — Retry with backoff</summary>

```python
from splinter.control import RetryStrategy, RetryConfig, RetryMode

strategy = RetryStrategy(RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
    mode=RetryMode.BASIC,  # or FAIL_CLOSED
))

result = await strategy.execute(unreliable_fn, arg1, arg2)
```

</details>

<details>
<summary><b>RulesEngine</b> — Custom rules</summary>

```python
from splinter.control import RulesEngine, Rule, RuleAction

engine = RulesEngine()

engine.add_rule(Rule(
    rule_id="block_expensive",
    condition=lambda ctx: ctx.get("cost", 0) > 10,
    action=RuleAction.BLOCK,
    message="Cost > $10",
))

engine.add_rule(Rule(
    rule_id="warn_delete",
    condition=lambda ctx: ctx.get("tool") == "delete_file",
    action=RuleAction.WARN,
    message="Deleting files",
))

engine.evaluate({"cost": 15})  # Raises RuleViolationError
```

</details>

<details>
<summary><b>MemoryStore</b> — Capped storage</summary>

```python
from splinter.control import MemoryStore

store = MemoryStore(
    max_size_bytes=10*1024*1024,  # 10 MB
    max_entries=1000,
)

store.set("researcher", "key", "value")
value = store.get("researcher", "key")
# Auto-evicts old entries when full
```

</details>

---

## 🤝 Coordination Objects

> Import from `splinter.coordination`

<details>
<summary><b>SharedState</b> — Single source of truth</summary>

```python
from splinter.coordination import SharedState

state = SharedState(initial_data={"topic": "AI"})

state.set("research.findings", ["a", "b", "c"])
findings = state.get("research.findings")

print(state.version)  # Increments on change

# Snapshot & restore
snapshot = state.snapshot()
state.set("oops", "mistake")
state.restore(snapshot)  # Rolled back
```

</details>

<details>
<summary><b>StateOwnership</b> — Who writes what</summary>

```python
from splinter.coordination import StateOwnership

ownership = StateOwnership()
ownership.register_ownership("researcher", ["research.*"])
ownership.register_ownership("writer", ["content.*"])

ownership.check_write_permission("researcher", "research.x")  # ✓
ownership.check_write_permission("researcher", "content.x")   # ✗
```

</details>

<details>
<summary><b>CheckpointManager</b> — Save & resume</summary>

```python
from splinter.coordination import CheckpointManager, FileCheckpointStorage

mgr = CheckpointManager(storage=FileCheckpointStorage("./checkpoints"))

# Save
mgr.create_checkpoint(
    workflow_id="wf-1",
    step=3,
    agent_id="researcher",
    status=AgentStatus.COMPLETED,
    state=state,
    metrics=metrics,
)

# Resume after crash
cp = mgr.get_latest_checkpoint("wf-1")
resume_from = cp.state
```

</details>

<details>
<summary><b>SchemaValidator</b> — Validate outputs</summary>

```python
from splinter.coordination import SchemaValidator

validator = SchemaValidator()

schema = {
    "type": "object",
    "properties": {"findings": {"type": "array"}},
    "required": ["findings"],
}

validator.validate(output, schema)  # Raises SchemaValidationError
```

</details>

<details>
<summary><b>HandoffManager</b> — Agent-to-agent validation</summary>

```python
from splinter.coordination import HandoffManager

handoff = HandoffManager(mode="strict")
handoff.register_schema("researcher", "writer", schema)
handoff.validate_handoff("researcher", "writer", data)
```

</details>

<details>
<summary><b>ChainContext</b> — Execution history</summary>

```python
from splinter.coordination import ChainContext

ctx = ChainContext()
ctx.register_agent("researcher", "Researches topics")
ctx.register_agent("writer", "Writes articles")

ctx.record_execution("researcher", {"in": "..."}, {"out": "..."})

# Writer sees what researcher did
writer_ctx = ctx.get_context_for_agent("writer")
```

</details>

<details>
<summary><b>GoalTracker</b> — Track progress</summary>

```python
from splinter.coordination import GoalTracker, Goal

tracker = GoalTracker()

tracker.set_goal(Goal(
    goal_id="article",
    description="Write blog post",
    success_criteria=["Research", "Draft", "Review"],
))

tracker.mark_criterion_met("article", "Research")
tracker.update_progress("article", 0.33)

if tracker.is_achieved("article"):
    print("Done!")
```

</details>

<details>
<summary><b>ActionEligibility</b> — Who acts when</summary>

```python
from splinter.coordination import ActionEligibility

elig = ActionEligibility()
elig.set_eligible("researcher")

elig.can_act("researcher")  # True
elig.can_act("writer")      # False

elig.transfer("researcher", "writer")
elig.can_act("writer")      # True
```

</details>

<details>
<summary><b>CompletionTracker</b> — "I'm done" signals</summary>

```python
from splinter.coordination import CompletionTracker

tracker = CompletionTracker()
tracker.require_completion("researcher")
tracker.require_completion("writer")

tracker.declare_complete("researcher", output={...})
tracker.declare_complete("writer", output={...})

if tracker.all_complete():
    print("Workflow done!")
```

</details>

<details>
<summary><b>WaitTracker</b> — Why idle?</summary>

```python
from splinter.coordination import WaitTracker, WaitReason

tracker = WaitTracker()
tracker.start_waiting("writer", WaitReason.WAITING_FOR_INPUT, "researcher")

reason, source = tracker.get_waiting_for("writer")
# WAITING_FOR_INPUT, "researcher"

tracker.stop_waiting("writer")
```

</details>

---

## 🔄 Full Example

```python
from splinter.workflow import Workflow
from splinter.types import AgentConfig, ExecutionLimits, LLMProvider, LoopDetectionConfig

# Create workflow with all protections
workflow = Workflow(
    workflow_id="research-pipeline",
    limits=ExecutionLimits(
        max_budget=20.0,
        max_steps=200,
        max_time_seconds=600,
    ),
    loop_detection=LoopDetectionConfig(
        max_repeated_outputs=3,
        max_no_state_change=5,
    ),
    checkpoint_enabled=True,
)

# Add agents
workflow.add_agent(AgentConfig(
    agent_id="researcher",
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    system_prompt="Research. Output JSON.",
    tools=["web_search"],
    state_ownership=["research.*"],
))

workflow.add_agent(AgentConfig(
    agent_id="writer",
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    system_prompt="Write. Output JSON.",
    state_ownership=["content.*"],
))

# Define order
workflow.add_step("researcher")
workflow.add_step("writer", depends_on=["researcher"])

# Run
result = await workflow.run(initial_state={"topic": "AI trends"})

print(f"Success: {result.success}")
print(f"Cost: ${result.metrics['total_cost']:.4f}")
```

---

## 🌐 Providers

| Provider | Key | Default Model |
|----------|-----|---------------|
| OpenAI | `openai_key="sk-..."` | gpt-4o-mini |
| Anthropic | `anthropic_key="sk-..."` | claude-sonnet-4-20250514 |
| Gemini | `gemini_key="..."` | gemini-1.5-flash |
| Grok | `grok_key="xai-..."` | grok-3-mini-fast |

Or use env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`

---

## 🧪 Testing

```bash
pytest tests/ -v  # 102 tests, MockProvider, no keys needed
```

---

## License

MIT
