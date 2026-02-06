# Setup Guide

## 1. Clone & Open in VSCode

```bash
git clone https://github.com/your-repo/splinter.git
cd splinter
code .
```

## 2. Python Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -e .
pip install openai pytest
```

## 3. VSCode Extensions

Install these extensions (Cmd+Shift+X):
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **MDX** (for docs)

## 4. Configure VSCode

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true
}
```

## 5. Set OpenAI API Key

Create `.env` file (add to .gitignore):
```
OPENAI_API_KEY=sk-...
```

Or export in terminal:
```bash
export OPENAI_API_KEY=sk-...
```

## 6. Run Tests

```bash
# All tests (uses MockProvider, no API key needed)
pytest tests/ -v

# With real OpenAI API
OPENAI_API_KEY=sk-... pytest tests/ -v
```

## 7. Run Mintlify Docs

```bash
# Install Mintlify CLI
npm install -g mintlify

# Run docs locally
cd docs
mintlify dev
```

Open http://localhost:3000

---

# Using with OpenAI

## Single Agent

```python
import asyncio
import os
from splinter import Gateway, AgentBuilder, LLMProvider, ExecutionLimits

async def main():
    # Create gateway with limits
    gateway = Gateway(
        limits=ExecutionLimits(max_budget=1.0, max_steps=10)
    )
    gateway.configure_provider("openai", api_key=os.environ["OPENAI_API_KEY"])

    # Create agent
    agent = (
        AgentBuilder("assistant")
        .with_provider(LLMProvider.OPENAI, "gpt-4o-mini")
        .with_system_prompt("You are a helpful assistant. Output JSON.")
        .build(gateway)
    )

    # Run
    result = await agent.run(task="What is 2+2? Reply with {answer: number}")
    print(result)

asyncio.run(main())
```

## Multi-Agent Pipeline

```python
import asyncio
import os
from splinter import (
    Gateway,
    Workflow,
    AgentConfig,
    LLMProvider,
    ExecutionLimits,
)

async def main():
    # Gateway
    gateway = Gateway(
        limits=ExecutionLimits(max_budget=2.0, max_steps=20)
    )
    gateway.configure_provider("openai", api_key=os.environ["OPENAI_API_KEY"])

    # Workflow
    workflow = Workflow(workflow_id="research-pipeline")
    workflow._gateway = gateway

    # Agent 1: Researcher
    workflow.add_agent(AgentConfig(
        agent_id="researcher",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        system_prompt="""You research topics.
Output JSON: {"findings": ["fact1", "fact2", "fact3"]}""",
        state_ownership=["research.*"],
    ))

    # Agent 2: Writer
    workflow.add_agent(AgentConfig(
        agent_id="writer",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        system_prompt="""You write based on research findings.
Output JSON: {"article": "..."}""",
        state_ownership=["content.*"],
    ))

    # Define flow: researcher -> writer
    workflow.add_step("researcher")
    workflow.add_step("writer", depends_on=["researcher"])

    # Run
    result = await workflow.run(
        initial_state={"topic": "Benefits of exercise"}
    )

    print(f"Status: {result.status}")
    print(f"Cost: ${result.metrics['total_cost']:.4f}")
    print(f"\nResearcher output:")
    print(result.outputs["researcher"])
    print(f"\nWriter output:")
    print(result.outputs["writer"])

asyncio.run(main())
```

## With Tools

```python
import asyncio
import os
from splinter import Gateway, AgentBuilder, LLMProvider

# Define tools
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

def tool_executor(name: str, args: dict) -> str:
    if name == "calculate":
        return calculate(args["expression"])
    elif name == "get_weather":
        return get_weather(args["city"])
    return f"Unknown tool: {name}"

async def main():
    gateway = Gateway()
    gateway.configure_provider("openai", api_key=os.environ["OPENAI_API_KEY"])

    agent = (
        AgentBuilder("assistant")
        .with_provider(LLMProvider.OPENAI, "gpt-4o-mini")
        .with_system_prompt("You help with calculations and weather.")
        .with_tools(["calculate", "get_weather"])
        .build(gateway, tool_executor=tool_executor)
    )

    result = await agent.run(
        task="What is 15 * 7, and what's the weather in Tokyo?"
    )
    print(result)

asyncio.run(main())
```

## Run Examples

```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Run single agent
python -c "
import asyncio
from examples.simple import single_agent
asyncio.run(single_agent())
"

# Run multi-agent
python -c "
import asyncio
from examples.simple import multi_agent_pipeline
asyncio.run(multi_agent_pipeline())
"
```

---

# Project Structure

```
splinter/
├── splinter/
│   ├── __init__.py        # Exports
│   ├── types.py           # Data types
│   ├── exceptions.py      # Errors
│   ├── gateway/
│   │   ├── gateway.py     # THE orchestrator
│   │   └── providers/     # OpenAI, Anthropic, etc.
│   ├── control/           # Limits, loops, tools
│   ├── coordination/      # State, ownership, checkpoints
│   └── workflow/          # Agent, Workflow
├── tests/
├── docs/                  # Mintlify docs
├── examples/
└── pyproject.toml
```

---

# Troubleshooting

## "No module named 'splinter'"
```bash
pip install -e .
```

## "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY=sk-...
```

## "Connection error" with OpenAI
Check your internet connection and API key.

## Tests fail with "mock" errors
```bash
# Run without API (uses MockProvider)
pytest tests/ -v
```
