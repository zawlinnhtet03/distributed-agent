"""
Base Agent Configuration and Shared Utilities.

Provides common configuration and utilities for all agents in the system.
This module establishes the shared interface that all agents implement.
"""

from typing import Any

from google.adk.agents import LlmAgent
from pydantic import BaseModel

from app.model_factory import ModelFactory


class AgentMetadata(BaseModel):
    """Metadata for agent registration and discovery."""

    name: str
    description: str
    shard_id: str | None = None
    capabilities: list[str] = []
    version: str = "1.0.0"


def create_agent(
    name: str,
    instruction: str,
    description: str,
    tools: list[Any] | None = None,
    sub_agents: list[LlmAgent] | None = None,
    tier: str = "default",
    temperature: float | None = None,
    output_schema: type[BaseModel] | None = None,
    **extra_kwargs: Any,
) -> LlmAgent:
    """
    Create a standardized LlmAgent with Groq configuration.

    This is the recommended way to create agents in the system. It ensures
    consistent configuration and integrates with the ModelFactory.

    Args:
        name: Unique identifier for the agent
        instruction: System instruction/prompt for the agent
        description: Human-readable description of agent's purpose
        tools: List of tool functions the agent can use
        sub_agents: List of sub-agents this agent can delegate to
        tier: Model tier ("default" or "fast")
        temperature: Override default temperature
        output_schema: Optional Pydantic model for structured output
        extra_kwargs: Additional LlmAgent kwargs (callbacks, output_key, etc.)

    Returns:
        Configured LlmAgent instance
    """
    model = ModelFactory.create(tier=tier, temperature=temperature)

    agent_kwargs = {
        "name": name,
        "model": model,
        "instruction": instruction,
        "description": description,
    }

    if tools:
        agent_kwargs["tools"] = tools

    if sub_agents:
        agent_kwargs["sub_agents"] = sub_agents

    if output_schema:
        agent_kwargs["output_schema"] = output_schema

    if extra_kwargs:
        agent_kwargs.update(extra_kwargs)

    return LlmAgent(**agent_kwargs)
