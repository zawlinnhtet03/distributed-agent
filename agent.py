"""
ADK Web Entry Point

This module exposes the root_agent for use with `adk web` command.
The agent variable at module level is auto-discovered by ADK.

Usage:
    adk web agent.py
    # or
    adk web .
"""

from app.agents import root_agent

# ADK web looks for an 'agent' or 'root_agent' at module level
agent = root_agent

# Alternative export for explicit naming
__all__ = ["agent", "root_agent"]
