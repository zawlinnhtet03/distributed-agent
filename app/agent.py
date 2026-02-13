"""
ADK Agent Entry Point for the 'app' agent.

This file exposes the root_agent (decision_pipeline) for ADK web discovery.

Pipeline:
    1. Planner → Decomposes query into an execution plan
    2. Aggregator → Reads plan, delegates to agents, synthesizes results
    3. Guardian → Reviews outputs for safety

Run with: adk web .
"""

from app.agents.aggregator import root_agent

# ADK web looks for 'root_agent' at module level
__all__ = ["root_agent"]
