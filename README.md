# Multi Agent Intelligence Research Hub

A modular, distributed agent system built with Google ADK (Agent Development Kit), LiteLLM (Mistral), and Tavily Search.

## Architecture

```
Planner  ->  Aggregator  ->  Guardian

Planner:
- Produces a compact execution plan.

Aggregator:
- Delegates work to specialist agents.
- Synthesizes results.

Guardian:
- Final safety pass (PII/sensitive-topic checks + safe formatting).
```

## Project Structure

```
google-adk/
├── agent.py                 # ADK web entry point
├── pyproject.toml           # Project dependencies
├── requirements.txt         # Alternative dependency list
├── .env                     # Environment variables (create from .env.example)
├── app/
│   ├── __init__.py
│   ├── config.py            # Settings management
│   ├── model_factory.py     # LiteLLM configuration + retry/backoff
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py    # Shared agent utilities
│   │   ├── planner.py        # Step 1
│   │   ├── aggregator.py     # Step 2
│   │   ├── guardian.py       # Step 3
│   │   ├── data.py           # Data/EDA tools
│   │   ├── rag.py            # Knowledge base retrieval
│   │   ├── scraper.py        # Web search (Tavily)
│   │   └── video.py          # Local video analysis
│   └── tools/
│       ├── __init__.py
│       ├── data_tools.py
│       ├── rag_tools.py
│       ├── tavily_tool.py
│       ├── tavily_search.py
│       └── video_tools.py
```

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys
# - MISTRAL_API_KEY: Get from https://console.mistral.ai/
# - TAVILY_API_KEY: Get from https://tavily.com
# - GROQ_API_KEY: Optional (legacy)

# Optional reliability tuning (defaults are fine)
# - LLM_NUM_RETRIES
# - LLM_RETRY_BASE_DELAY_S
# - LLM_RETRY_MAX_DELAY_S
```

### 3. Run with ADK Web

```bash
# Start the visual debugger
adk web .

# Or explicitly specify the agent file
adk web agent.py
```

### 4. Access the Interface

Open your browser to `http://localhost:8000` to interact with the agent system through ADK's visual debugger.

## Running the Custom UI (Next.js) + API Backend (FastAPI)

This repo includes a custom dashboard UI in `frontend/` that talks to a FastAPI backend in `app/api_server.py`.

### 1) Start the API backend

The UI expects the backend to run on port `8001` by default.

```bash
python -m app.api_server
```

You can change host/port via:

```bash
set ADK_BACKEND_HOST=127.0.0.1
set ADK_BACKEND_PORT=8001
python -m app.api_server
```

### 2) Start the UI

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

### 3) UI → Backend configuration

The Next.js UI proxies streaming requests to the backend using the environment variable:

- `ADK_API_URL` (server-side in Next.js). Default: `http://localhost:8001`

Optional:

- `NEXT_PUBLIC_PERSIST_SESSIONS` (`true`/`false`) enables local conversation history persistence.

If you run the backend on a different host/port, set `ADK_API_URL` in `frontend/.env.local`.

## UI Features

### Simple View

- Sends requests to `/api/adk/run-sse` (SSE streaming)
- Displays per-agent status + logs
- Captures shared monitoring state in `localStorage` while a task is running

### Advanced View (Monitoring)

- Opened from Simple View and receives `session` in the URL query
- Polls shared monitoring state from `localStorage` to show live agent status/logs

### Data Agent Artifacts (Tables + Charts)

The Data Agent tools can emit renderable artifacts in the agent stream using a marker line:

- `ADK_ARTIFACT:{...json...}`

Artifacts supported:

- Table previews (rendered as HTML tables in the UI)
- Plotly charts rendered inline in the UI (no chart HTML files are required)

Note: Inline charts load Plotly from the CDN at runtime.

## ModelFactory Usage

The `ModelFactory` class provides a clean interface for creating LiteLLM-configured models.

```python
from app.model_factory import ModelFactory
from google.adk.agents import LlmAgent

model = ModelFactory.create()
fast_model = ModelFactory.create(tier="fast")

agent = LlmAgent(
    name="my_agent",
    model=model,
    instruction="Your instruction here",
)
```

## Adding New Agents

1. Create a new file in `app/agents/`:

```python
# app/agents/my_agent.py
from app.agents.base_agent import create_agent

MY_AGENT_INSTRUCTION = """Your agent instructions..."""

my_agent = create_agent(
    name="my_agent",
    instruction=MY_AGENT_INSTRUCTION,
    description="What this agent does",
    tools=[],  # Add tools if needed
    tier="default",
    temperature=0.5,
)
```

2. Register in `app/agents/__init__.py`
3. Add as sub-agent to orchestrator if needed

## Available Models

| Tier | Model | Use Case |
|------|-------|----------|
| `default` | `mistral/mistral-medium-latest` | Complex reasoning, orchestration |
| `fast` | `mistral/mistral-small-latest` | Quick responses, simple tasks |

## Data Files (important for UI uploads)

The data agent can only operate on files that exist on disk inside this repo.

- Put CSV/Excel files in `datasets/` (recommended) so `list_data_files()` can discover them.
- ADK Web uploads are treated as session artifacts and are not guaranteed to be written into `datasets/` automatically.
- When building your custom UI, save uploads to `datasets/` to make them immediately usable by the data tools.

## License

MIT
