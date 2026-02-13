"""
Tools module for the Multi Agent Intelligence Research Hub.

Contains tool implementations for agents to use.
"""

# Legacy tool functions
from app.tools.tavily_search import tavily_search, tavily_search_context

# ADK FunctionTool wrappers
from app.tools.tavily_tool import (
    TAVILY_TOOLS,
    tavily_answer_tool,
    tavily_extract_tool,
    tavily_search_tool,
    tavily_web_search,
)

# Video tools
from app.tools.video_tools import (
    list_videos,
    analyze_video_locally,
    analyze_uploaded_video,
)

# RAG tools
from app.tools.rag_tools import (
    add_document,
    ingest_directory,
    ingest_file,
    ingest_text,
    list_collections,
    vector_search,
)

# Data tools
from app.tools.data_tools import (
    list_data_files,
    load_dataset,
    profile_dataset,
    clean_dataset,
    analyze_trends,
    add_datetime_features,
    train_small_model,
)

__all__ = [
    # Legacy functions
    "tavily_search",
    "tavily_search_context",
    # ADK tools
    "TAVILY_TOOLS",
    "tavily_search_tool",
    "tavily_extract_tool",
    "tavily_answer_tool",
    "tavily_web_search",
    # Video tools
    "list_videos",
    "analyze_video_locally",
    "analyze_uploaded_video",
    # RAG tools
    "vector_search",
    "add_document",
    "ingest_text",
    "ingest_file",
    "ingest_directory",
    "list_collections",
    # Data tools
    "list_data_files",
    "load_dataset",
    "profile_dataset",
    "clean_dataset",
    "analyze_trends",
    "add_datetime_features",
    "train_small_model",
]
