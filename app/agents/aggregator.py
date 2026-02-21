"""
Aggregator Agent - deterministic coordinator and synthesizer.

This module keeps the planner/guardian pipeline, but avoids provider-specific
function-calling failures by executing common worker tasks directly in a
before-model callback.
"""

from __future__ import annotations

import logging
import os
import re

from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

from app.agents.base_agent import create_agent
from app.agents.guardian import guardian_agent, guardian_pre_agent
from app.agents.planner import planner_agent
from app.agents.rag import rag_agent
from app.agents.scraper import scraper_agent
from app.tools.data_tools import analyze_trends, list_data_files, load_dataset, profile_dataset
from app.tools.rag_tools import vector_search
from app.tools.video_tools import analyze_uploaded_video
from common.adk_text_sanitizer import force_text_only_model_input


# ============================================================================
# LAYER: GatheringLayer (ParallelAgent)
# Kept for architecture compatibility and potential future delegation.
# ============================================================================
gathering_layer = ParallelAgent(
    name="gathering_layer",
    description="Research layer - runs scraper and rag in parallel for web search and knowledge retrieval",
    sub_agents=[scraper_agent, rag_agent],
)


AGGREGATOR_INSTRUCTION = """You are the Aggregator Agent.

Primary behavior:
- Read the planner output from state key {plan}.
- Read the sanitized request from {sanitized_request} when available.
- Synthesize the already-collected results into one clear response.
- If there are any related tasks with gathering_layer, run them.

If no delegated outputs are available, provide a concise direct response.
"""


_VIDEO_EXT = r"(?:mp4|mov|avi|mkv|webm)"
_DATA_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}


def _extract_latest_user_text(llm_request) -> str:
    if llm_request is None or not getattr(llm_request, "contents", None):
        return ""

    fallback = ""
    for content in reversed(llm_request.contents):
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            text_val = getattr(part, "text", None)
            if text_val:
                chunks.append(text_val)
        merged = " ".join(chunks).strip()
        role = getattr(content, "role", None)
        if role == "user" and merged:
            return merged
        if merged and not fallback:
            fallback = merged
    return fallback


# Canonical mapping: alias → runnable agent name
_AGENT_ALIASES: dict[str, str] = {
    "gathering_layer": "gathering_layer",
    "gathering layer": "gathering_layer",
    "gatheringlayer": "gathering_layer",
    "scraper": "gathering_layer",
    "rag": "gathering_layer",
    "web_search": "gathering_layer",
    "search": "gathering_layer",
    "research": "gathering_layer",
    "data": "data",
    "data_agent": "data",
    "video": "video",
    "video_agent": "video",
    "aggregator": "aggregator",
}


def _normalize_agent_name(raw: str) -> str:
    """Strip markdown formatting and resolve aliases."""
    # Remove markdown bold, backticks, underscores used as formatting
    cleaned = re.sub(r"[*`]", "", raw).strip().lower()
    # Try exact alias lookup first
    if cleaned in _AGENT_ALIASES:
        return _AGENT_ALIASES[cleaned]
    # Try replacing spaces/hyphens with underscores
    underscore_form = re.sub(r"[\s-]+", "_", cleaned)
    if underscore_form in _AGENT_ALIASES:
        return _AGENT_ALIASES[underscore_form]
    return cleaned


def _extract_plan_agents(plan_text: str) -> list[str]:
    """Extract agent names from planner output. Handles multiple formats."""
    agents: list[str] = []
    if not plan_text:
        return agents
    
    # Pattern 1: "agent: gathering_layer" (standard format)
    for match in re.finditer(r"agent:\s*\**`?([a-z_][a-z0-9_ -]*)`?\**", plan_text, flags=re.IGNORECASE):
        raw_name = match.group(1).strip()
        name = _normalize_agent_name(raw_name)
        if name and name not in agents:
            agents.append(name)
    
    # Pattern 2: "→ agent: data" or "-> agent: data" (arrow format)
    for match in re.finditer(r"[-→]\s*agent:\s*\**`?([a-z_][a-z0-9_ -]*)`?\**", plan_text, flags=re.IGNORECASE):
        raw_name = match.group(1).strip()
        name = _normalize_agent_name(raw_name)
        if name and name not in agents:
            agents.append(name)
    
    logger.info("[aggregator] Extracted from plan: %s", agents)
    return agents


def _infer_agents_from_query(query: str) -> list[str]:
    q = (query or "").lower()
    agents: list[str] = []

    if re.search(rf"\.{_VIDEO_EXT}\b", q) or any(
        token in q for token in ("video", "frame", "clip", "uploaded")
    ):
        agents.append("video")

    if any(
        token in q
        for token in (
            "csv", "tsv", "xlsx", "dataset", "eda", "table", "data",
            "file", "analyze", "column", "row", "excel", "dataframe",
            "profile", "statistic", "trend", "chart", "plot",
        )
    ):
        agents.append("data")

    if any(
        token in q
        for token in (
            "search", "web", "news", "research", "source", "citation",
            "find", "look up", "lookup", "latest", "current", "recent",
            "article", "report", "scrape", "crawl", "discover",
        )
    ):
        agents.append("gathering_layer")

    return agents or ["aggregator"]


def _extract_video_filename(text: str) -> str:
    if not text:
        return ""
    match = re.search(rf"([A-Za-z0-9._-]+\.{_VIDEO_EXT})", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _extract_first_data_path(listing_text: str) -> str:
    if not listing_text:
        return ""
    for line in listing_text.splitlines():
        if "->" not in line:
            continue
        candidate = line.split("->", 1)[1].strip()
        ext = os.path.splitext(candidate)[1].lower()
        if ext in _DATA_EXTS:
            return candidate
    return ""


def _run_video_task(query: str) -> str:
    filename = _extract_video_filename(query)
    if filename:
        return analyze_uploaded_video(filename=filename)
    return analyze_uploaded_video()


def _run_data_task(query: str) -> str:
    listing = list_data_files()
    if "No data files found" in listing:
        return listing

    selected_path = _extract_first_data_path(listing)
    if not selected_path:
        return listing

    blocks = [
        f"Selected dataset: {selected_path}",
        load_dataset(selected_path),
        profile_dataset(selected_path),
    ]

    q = (query or "").lower()
    if any(token in q for token in ("trend", "outlier", "pattern", "eda", "explor")):
        blocks.append(analyze_trends(selected_path))

    return "\n\n".join(blocks)


def _run_gathering_task(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "No query provided for research task."

    sections: list[str] = []

    try:
        from app.tools.tavily_tool import tavily_web_search

        web = tavily_web_search(query=q, search_depth="advanced", max_results=5)
        if isinstance(web, dict) and web.get("error"):
            sections.append(f"Web search error: {web.get('error')}")
        else:
            results = (web or {}).get("results", []) if isinstance(web, dict) else []
            if results:
                lines = ["Web search top results:"]
                for idx, item in enumerate(results[:5], start=1):
                    title = str(item.get("title", "")).strip()
                    url = str(item.get("url", "")).strip()
                    lines.append(f"{idx}. {title} - {url}")
                sections.append("\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        sections.append(f"Web search unavailable: {exc}")

    try:
        rag = vector_search(query=q, top_k=3)
        docs = rag.get("documents", []) if isinstance(rag, dict) else []
        if docs:
            lines = ["Knowledge base hits:"]
            for idx, doc in enumerate(docs[:3], start=1):
                source = str(doc.get("source", "unknown"))
                score = doc.get("score", 0)
                preview = str(doc.get("content", ""))[:180].replace("\n", " ").strip()
                lines.append(f"{idx}. score={score} source={source} preview={preview}")
            sections.append("\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        sections.append(f"RAG search unavailable: {exc}")

    return "\n\n".join(sections) if sections else "No research results available."


def _direct_aggregator_before_model(*args, **kwargs):
    """
    Execute planned worker steps deterministically before model generation.
    This avoids provider-specific transfer_to_agent function-call failures.
    """
    callback_context = kwargs.get("callback_context") if kwargs else None
    llm_request = kwargs.get("llm_request") if kwargs else None

    if callback_context is None and len(args) >= 1:
        callback_context = args[0]
    if llm_request is None and len(args) >= 2:
        llm_request = args[1]

    force_text_only_model_input(callback_context=callback_context, llm_request=llm_request)

    # ── Collect query from ALL available sources ──
    state_query = ""
    original_query = ""
    plan_text = ""
    if callback_context is not None:
        state_query = str(callback_context.state.get("sanitized_request", "")).strip()
        original_query = str(callback_context.state.get("original_user_query", "")).strip()
        plan_text = str(callback_context.state.get("plan", "")).strip()

    llm_query = _extract_latest_user_text(llm_request)

    # Use whichever source is non-empty; prefer state (sanitized) but keep
    # the raw llm_query too — we run inference on BOTH to avoid losing intent.
    query = state_query or original_query or llm_query
    
    # Data/video agents need the ORIGINAL query to detect file references
    # (sanitization may strip PII including file paths like C:\path\file.csv)
    query_for_files = original_query or llm_query or state_query
    
    logger.info(
        "[aggregator] RAW PLAN TEXT:\n%s",
        plan_text if plan_text else "(empty)",
    )

    # ── Determine which agents to run ──
    planned_agents = _extract_plan_agents(plan_text)
    inferred_from_state = _infer_agents_from_query(state_query) if state_query else []
    inferred_from_original = _infer_agents_from_query(original_query) if original_query else []
    inferred_from_llm = _infer_agents_from_query(llm_query) if llm_query else []

    # Merge all sources: planned + inferred from state + inferred from original + inferred from raw query
    seen: set[str] = set()
    agents: list[str] = []
    for name in planned_agents + inferred_from_state + inferred_from_original + inferred_from_llm:
        if name not in seen and name != "aggregator":
            seen.add(name)
            agents.append(name)

    # SAFETY NET: If planner only picked data but query has research keywords, force gathering_layer
    combined_text = f"{original_query} {llm_query}".lower()
    research_keywords = ["research", "news", "search", "web", "latest", "current", "trend", "article"]
    data_keywords = ["csv", "dataset", "file", "analyze", "data"]
    has_research = any(kw in combined_text for kw in research_keywords)
    has_data = any(kw in combined_text for kw in data_keywords)
    
    if has_research and "gathering_layer" not in agents:
        logger.warning("[aggregator] FORCE: Adding gathering_layer (research keywords detected)")
        agents.append("gathering_layer")
    if has_data and "data" not in agents and any(ext in combined_text for ext in [".csv", ".tsv", ".xlsx"]):
        logger.warning("[aggregator] FORCE: Adding data agent (file keywords detected)")
        agents.append("data")

    # If nothing was resolved at all, try inference on the combined text
    if not agents:
        agents = _infer_agents_from_query(f"{state_query} {llm_query}")

    runnable = [a for a in agents if a in {"video", "data", "gathering_layer"}]

    logger.info(
        "[aggregator] planned=%s  inferred_state=%s  inferred_original=%s  inferred_llm=%s  final_runnable=%s",
        planned_agents, inferred_from_state, inferred_from_original, inferred_from_llm, runnable,
    )
    logger.info("[aggregator] query(sanitized)=%r  query_for_files(original)=%r", query[:120], query_for_files[:120])

    # If planner requested only direct aggregator handling, let model answer normally.
    if not runnable:
        return None

    sections: list[str] = []
    executed_agents: list[str] = []
    for agent_name in runnable:
        logger.info("[aggregator] ===== STARTING %s =====", agent_name)
        try:
            if agent_name == "video":
                output = _run_video_task(query_for_files)
            elif agent_name == "data":
                output = _run_data_task(query_for_files)
            elif agent_name == "gathering_layer":
                output = _run_gathering_task(query)
            else:
                output = f"Unknown agent: {agent_name}"
            executed_agents.append(agent_name)
            logger.info("[aggregator] ===== COMPLETED %s (output len=%d) =====", agent_name, len(output))
        except Exception as exc:  # noqa: BLE001
            logger.error("[aggregator] ===== FAILED %s: %s =====", agent_name, exc)
            output = f"Execution error in {agent_name}: {exc}"
            executed_agents.append(f"{agent_name}(failed)")
        sections.append(f"[{agent_name}]\n{output}")

    logger.info("[aggregator] Executed agents: %s", executed_agents)

    final_text = "\n\n".join(sections)

    if callback_context is not None:
        callback_context.state["aggregator_execution"] = {
            "agents": runnable,
            "query": query,
            "plan": plan_text,
        }
        # Store aggregator output so the guardian can pass it through
        callback_context.state["aggregator_output"] = final_text
        # IMPORTANT: Clear the plan so the LLM doesn't try to call agents as tools
        callback_context.state["plan"] = ""

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=final_text)],
        )
    )


aggregator_agent = create_agent(
    name="aggregator",
    instruction=AGGREGATOR_INSTRUCTION,
    description="Executes planner steps and synthesizes outputs",
    tier="default",
    temperature=0.2,
    before_model_callback=_direct_aggregator_before_model,
)


# ============================================================================
# ROOT: DecisionPipeline (SequentialAgent)
# Step 1: Planner -> Step 2: Guardian Pre -> Step 3: Aggregator -> Step 4: Guardian
# ============================================================================
decision_pipeline = SequentialAgent(
    name="decision_pipeline",
    description="Sequential pipeline: planner -> guardian_pre -> aggregator -> guardian",
    sub_agents=[
        planner_agent,      # Step 1: Analyze query and produce a plan
        guardian_pre_agent, # Step 2: Sanitize user request (PII redaction) before delegation
        aggregator_agent,   # Step 3: Execute plan and synthesize results
        guardian_agent,     # Step 4: Safety review
    ],
)

# Export for ADK
root_agent = decision_pipeline

