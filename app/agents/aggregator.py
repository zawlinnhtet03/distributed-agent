"""
Aggregator Agent - deterministic coordinator and synthesizer.

This module keeps the planner/guardian pipeline, but avoids provider-specific
function-calling failures by executing common worker tasks directly in a
before-model callback.
"""

from __future__ import annotations

import os
import re

from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

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


def _extract_plan_agents(plan_text: str) -> list[str]:
    agents: list[str] = []
    for match in re.finditer(r"agent:\s*([a-z_]+)", plan_text or "", flags=re.IGNORECASE):
        name = match.group(1).strip().lower()
        if name and name not in agents:
            agents.append(name)
    return agents


def _infer_agents_from_query(query: str) -> list[str]:
    q = (query or "").lower()
    agents: list[str] = []

    if re.search(rf"\.{_VIDEO_EXT}\b", q) or any(
        token in q for token in ("video", "frame", "clip", "uploaded")
    ):
        agents.append("video")

    if any(token in q for token in ("csv", "tsv", "xlsx", "dataset", "eda", "table", "data")):
        agents.append("data")

    if any(token in q for token in ("search", "web", "news", "research", "source", "citation")):
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

    query = ""
    plan_text = ""
    if callback_context is not None:
        query = str(callback_context.state.get("sanitized_request", "")).strip()
        plan_text = str(callback_context.state.get("plan", "")).strip()
    if not query:
        query = _extract_latest_user_text(llm_request)

    planned_agents = _extract_plan_agents(plan_text)
    agents = planned_agents or _infer_agents_from_query(query)
    runnable = [a for a in agents if a in {"video", "data", "gathering_layer"}]

    # If planner requested only direct aggregator handling, let model answer normally.
    if not runnable:
        return None

    sections: list[str] = []
    for agent_name in runnable:
        try:
            if agent_name == "video":
                output = _run_video_task(query)
            elif agent_name == "data":
                output = _run_data_task(query)
            else:
                output = _run_gathering_task(query)
        except Exception as exc:  # noqa: BLE001
            output = f"Execution error in {agent_name}: {exc}"
        sections.append(f"[{agent_name}]\n{output}")

    final_text = "\n\n".join(sections)

    if callback_context is not None:
        callback_context.state["aggregator_execution"] = {
            "agents": runnable,
            "query": query,
            "plan": plan_text,
        }

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

