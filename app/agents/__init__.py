"""
Agents module for the Multi Agent Intelligence Research Hub.

Each agent is a modular component that can be composed into larger systems.
All agents share a common interface and use the ModelFactory for consistent
Groq configuration.

Architecture (Sequential Pipeline):
    decision_pipeline (SequentialAgent) - ROOT
    ├── planner (LlmAgent) → Analyzes query, writes plan to state["plan"]
    ├── aggregator (LlmAgent) → Reads plan, delegates to agents, synthesizes
    │   ├── gathering_layer (ParallelAgent) → Research layer
    │   │   ├── scraper → Web search (Tavily)
    │   │   └── rag → Vector retrieval (ChromaDB)
    │   ├── data → Data/ML tasks (independent)
    │   └── video → Video analysis (independent)
    └── guardian → Safety review

Pipeline Flow:
    User Query → Planner → Aggregator → Guardian → Response
                  (plan)    (execute     (safety
                             + synth)     gate)
"""

# Worker agents
from app.agents.guardian import guardian_agent
from app.agents.rag import rag_agent
from app.agents.scraper import scraper_agent
from app.agents.video import video_agent
from app.agents.data import data_agent
from app.agents.planner import planner_agent

# Pipeline components and root agent
from app.agents.aggregator import aggregator_agent, decision_pipeline, gathering_layer, root_agent

__all__ = [
    # Root agent (Sequential Pipeline)
    "root_agent",
    "decision_pipeline",
    # Pipeline components
    "gathering_layer",
    "planner_agent",
    "aggregator_agent",
    # Worker agents
    "scraper_agent",
    "video_agent",
    "data_agent",
    "rag_agent",
    "guardian_agent",
]
