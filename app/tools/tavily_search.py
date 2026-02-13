"""
Tavily Search Tool for ADK Agents.

Provides web search capabilities using the Tavily API.
"""

from typing import Literal

from tavily import TavilyClient

from app.config import get_settings


def _get_client() -> TavilyClient:
    """Get configured Tavily client."""
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


def tavily_search(
    query: str,
    search_depth: Literal["basic", "advanced"] = "basic",
    max_results: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict:
    """
    Perform a web search using Tavily API.

    This tool searches the web and returns structured results including
    titles, URLs, and content snippets.

    Args:
        query: The search query string
        search_depth: "basic" for fast results, "advanced" for comprehensive
        max_results: Maximum number of results to return (1-10)
        include_domains: Optional list of domains to include
        exclude_domains: Optional list of domains to exclude

    Returns:
        Dictionary containing search results with keys:
        - results: List of result objects with url, title, content
        - query: Original query
        - response_time: Time taken for search
    """
    client = _get_client()

    response = client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        include_domains=include_domains or [],
        exclude_domains=exclude_domains or [],
    )

    return {
        "query": query,
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in response.get("results", [])
        ],
        "response_time": response.get("response_time", 0),
    }


def tavily_search_context(
    query: str,
    search_depth: Literal["basic", "advanced"] = "advanced",
    max_results: int = 5,
) -> str:
    """
    Perform a search and return context-ready text.

    This is optimized for RAG pipelines where you need the search
    results as a single context string.

    Args:
        query: The search query string
        search_depth: Search depth level
        max_results: Maximum number of results

    Returns:
        Formatted string with search results ready for context injection
    """
    client = _get_client()

    context = client.get_search_context(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
    )

    return context
