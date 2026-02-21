"""
Main Entry Point - Multi Agent Intelligence Research Hub

This module provides the entry point for running the agent system.
Use this for programmatic access or testing outside of ADK web.

For ADK web UI, run from the project root:
    adk web .

For programmatic usage:
    from app.main import root_agent, run_query
    
    # Run a query
    response = await run_query("What are the latest AI developments?")
"""

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)
load_dotenv(dotenv_path=_REPO_ROOT / ".env.local", override=False)

from app.agents.aggregator import root_agent

# Re-export the root agent
__all__ = ["root_agent", "run_query", "create_runner"]


def create_runner() -> Runner:
    """
    Create a configured Runner for the aggregator agent.
    
    Returns:
        Runner instance ready to process queries.
    """
    session_service = InMemorySessionService()
    
    return Runner(
        agent=root_agent,
        app_name="sharded-retrieval-system",
        session_service=session_service,
    )


async def run_query(
    query: str,
    user_id: str = "default_user",
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Run a query through the multi-agent system.
    
    Args:
        query: The user's query string.
        user_id: User identifier for session management.
        session_id: Optional session ID for conversation continuity.
    
    Returns:
        Dictionary with the agent's response and metadata.
    
    Example:
        >>> import asyncio
        >>> from app.main import run_query
        >>> result = asyncio.run(run_query("Search for AI news"))
        >>> print(result["response"])
    """
    runner = create_runner()
    
    # Create or get session
    if session_id is None:
        session = await runner.session_service.create_session(
            app_name="sharded-retrieval-system",
            user_id=user_id,
        )
        session_id = session.id
    
    # Collect responses
    responses = []
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=query,
    ):
        if hasattr(event, "content") and event.content:
            if hasattr(event.content, "parts"):
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        responses.append(part.text)
    
    return {
        "query": query,
        "response": "\n".join(responses) if responses else "No response generated",
        "session_id": session_id,
        "user_id": user_id,
    }


async def interactive_session():
    """
    Run an interactive session with the agent system.
    
    Useful for testing and debugging.
    """
    print("=" * 60)
    print("Multi Agent Intelligence Research Hub")
    print("=" * 60)
    print("\nType 'quit' or 'exit' to end the session.\n")
    
    runner = create_runner()
    session = await runner.session_service.create_session(
        app_name="sharded-retrieval-system",
        user_id="interactive_user",
    )
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print("\nAgent: ", end="", flush=True)
            
            async for event in runner.run_async(
                user_id="interactive_user",
                session_id=session.id,
                new_message=query,
            ):
                if hasattr(event, "content") and event.content:
                    if hasattr(event.content, "parts"):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                print(part.text, end="", flush=True)
            
            print()  # Newline after response
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break


if __name__ == "__main__":
    # Run interactive session when executed directly
    asyncio.run(interactive_session())
