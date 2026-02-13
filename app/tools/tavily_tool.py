"""Tavily search tools."""

from typing import Literal
from google.adk.tools import FunctionTool
from tavily import TavilyClient
from app.config import get_settings


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


def tavily_web_search(
    query: str,
    search_depth: Literal["basic", "advanced"] = "advanced",
    max_results: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict:
    """Search the web using Tavily API. Returns titles, URLs, content."""
    client = _get_tavily_client()
    try:
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
        )
        return {
            "query": query,
            "results": [{"title": r.get("title", ""), "url": r.get("url", ""),
                        "content": r.get("content", ""), "score": r.get("score", 0)}
                       for r in response.get("results", [])],
        }
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}


def tavily_extract_content(urls: list[str]) -> dict:
    """Extract content from specific URLs using Tavily."""
    client = _get_tavily_client()
    try:
        response = client.extract(urls=urls[:10])
        return {
            "results": [{"url": r.get("url", ""), "content": r.get("raw_content", r.get("content", ""))}
                       for r in response.get("results", [])],
        }
    except Exception as e:
        return {"results": [], "error": str(e)}


def tavily_get_answer(query: str, search_depth: Literal["basic", "advanced"] = "advanced") -> dict:
    """Get AI-generated answer with sources using Tavily."""
    client = _get_tavily_client()
    try:
        response = client.search(query=query, search_depth=search_depth, include_answer=True)
        return {
            "query": query,
            "answer": response.get("answer", "No answer"),
            "sources": [r.get("url", "") for r in response.get("results", [])],
        }
    except Exception as e:
        return {"query": query, "answer": None, "error": str(e)}


# Export tools
tavily_search_tool = FunctionTool(func=tavily_web_search)
tavily_extract_tool = FunctionTool(func=tavily_extract_content)
tavily_answer_tool = FunctionTool(func=tavily_get_answer)
TAVILY_TOOLS = [tavily_search_tool, tavily_extract_tool, tavily_answer_tool]
