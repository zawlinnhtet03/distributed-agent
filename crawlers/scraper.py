import os
import json
import re

from tavily import TavilyClient
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types
from dotenv import load_dotenv
import litellm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_response import LlmResponse
from common.adk_text_sanitizer import force_text_only_model_input

litellm.set_verbose = False
litellm.suppress_debug_info = True

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if google_api_key:
    llm_model = Gemini(model="gemini-2.5-flash-lite")
elif mistral_api_key:
    llm_model = LiteLlm(
        model="mistral/mistral-medium-latest",
        api_key=mistral_api_key,
    )
else:
    raise ValueError("Missing model key: set GOOGLE_API_KEY or MISTRAL_API_KEY")

tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("Missing TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_api_key)
retry_config = types.HttpRetryOptions(attempts=5, initial_delay=1, http_status_codes=[429, 500, 503, 504])

def analyze_market_trends(query: str) -> str:
    """Performs market research using Tavily."""
    try:
        normalized_query = (query or "").strip()
        lower_q = normalized_query.lower()
        if (
            not normalized_query
            or lower_q.endswith((".mp4", ".mov", ".avi", ".mkv"))
            or "file attachment" in lower_q
            or "inline attachment" in lower_q
            or "video attachment" in lower_q
        ):
            return json.dumps(
                {
                    "query": normalized_query,
                    "answer": "Not available from tool output (invalid or non-market query).",
                    "results": [],
                },
                ensure_ascii=False,
            )

        response = tavily_client.search(
            query=normalized_query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )
        normalized = {
            "query": normalized_query,
            "answer": response.get("answer"),
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                    "score": r.get("score"),
                    "published_date": r.get("published_date"),
                }
                for r in response.get("results", [])
            ],
        }
        return json.dumps(normalized, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


def _build_grounded_scrape_report(raw_output: str, query: str | None) -> str:
    if not raw_output:
        return (
            "# Trend Summary\n"
            "- Not available from tool output.\n\n"
            "# Key Evidence\n"
            "- Not available from tool output.\n\n"
            "# Sources\n"
            "- Sources: None"
        )

    if raw_output.startswith("Error:"):
        return (
            "# Trend Summary\n"
            f"- {raw_output}\n"
            f"- Query: {query or 'Not available from tool output.'}\n\n"
            "# Key Evidence\n"
            "- Not available from tool output.\n\n"
            "# Sources\n"
            "- Sources: None"
        )

    data = None
    try:
        data = json.loads(raw_output)
    except Exception:
        # Backward compatibility for legacy string format.
        summary_match = re.search(r"SUMMARY:\s*(.*?)\s*DETAILS:", raw_output, flags=re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "Not available from tool output."
        source_lines = []
        if "SOURCES:" in raw_output:
            source_lines = [
                ln.strip() for ln in raw_output.split("SOURCES:", 1)[1].splitlines() if ln.strip().startswith("-")
            ]
        if not source_lines:
            source_lines = ["- Sources: None"]
        return (
            "# Trend Summary\n"
            f"- Query: {query or 'Not available from tool output.'}\n"
            f"- {summary}\n\n"
            "# Key Evidence\n"
            "- Not available from tool output.\n\n"
            "# Sources\n"
            + "\n".join(source_lines)
        )

    query_used = data.get("query") or query or "Not available from tool output."
    answer = data.get("answer") or "Not available from tool output."
    results = data.get("results") or []

    trend_lines = [
        f"- Query: {query_used}",
        f"- Answer: {answer}",
        f"- Result count: {len(results)}",
    ]

    evidence_lines = []
    for item in results[:5]:
        title = item.get("title") or "Untitled source"
        snippet = (item.get("content") or "No content").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        evidence_lines.append(f"- {title}: {snippet}")
    if not evidence_lines:
        evidence_lines = ["- Not available from tool output."]

    source_lines = []
    for item in results:
        url = item.get("url")
        if url:
            source_lines.append(f"- {url}")
    if not source_lines:
        source_lines = ["- Sources: None"]

    return (
        "# Trend Summary\n"
        + "\n".join(trend_lines)
        + "\n\n# Key Evidence\n"
        + "\n".join(evidence_lines)
        + "\n\n# Sources\n"
        + "\n".join(source_lines)
    )


def _capture_scrape_tool_output(*args, **kwargs):
    tool_context = kwargs.get("tool_context") if kwargs else None
    tool_args = kwargs.get("args") if kwargs else None
    tool_response = kwargs.get("tool_response") if kwargs else None

    if tool_context is None and len(args) >= 3:
        tool_context = args[2]
    if tool_args is None and len(args) >= 2:
        tool_args = args[1]
    if tool_response is None and len(args) >= 4:
        tool_response = args[3]

    if tool_context is None:
        return None

    query = None
    if isinstance(tool_args, dict):
        query = tool_args.get("query")

    tool_context.state["scrape_tool_output"] = str(tool_response)
    tool_context.state["scrape_query"] = query
    return None


def _force_grounded_scrape_response(*args, **kwargs):
    callback_context = kwargs.get("callback_context") if kwargs else None
    llm_response = kwargs.get("llm_response") if kwargs else None

    if callback_context is None and len(args) >= 1:
        callback_context = args[0]
    if llm_response is None and len(args) >= 2:
        llm_response = args[1]
    if callback_context is None or llm_response is None:
        return llm_response

    raw_output = callback_context.state.get("scrape_tool_output")
    query = callback_context.state.get("scrape_query")
    if raw_output is None:
        return llm_response

    grounded_report = _build_grounded_scrape_report(raw_output, query)
    llm_response.content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=grounded_report)],
    )
    return llm_response


def _extract_latest_user_query_from_request(llm_request) -> str:
    if llm_request is None or not getattr(llm_request, "contents", None):
        return ""

    for content in reversed(llm_request.contents):
        if getattr(content, "role", None) != "user":
            continue
        parts = getattr(content, "parts", None) or []
        text_chunks = []
        for part in parts:
            text_val = getattr(part, "text", None)
            if text_val:
                text_chunks.append(text_val)
        query = " ".join(text_chunks).strip()
        if query:
            return query
    return ""


def _direct_scrape_before_model(*args, **kwargs):
    """
    Deterministic path for robustness:
    - Extract latest user query
    - Run Tavily tool directly
    - Return grounded markdown as final model response
    This bypasses fragile provider-specific function-calling behavior.
    """
    callback_context = kwargs.get("callback_context") if kwargs else None
    llm_request = kwargs.get("llm_request") if kwargs else None

    if callback_context is None and len(args) >= 1:
        callback_context = args[0]
    if llm_request is None and len(args) >= 2:
        llm_request = args[1]

    # Keep OpenAI-compatible providers safe from non-text structured parts.
    force_text_only_model_input(callback_context=callback_context, llm_request=llm_request)

    query = _extract_latest_user_query_from_request(llm_request)
    raw_output = analyze_market_trends(query=query)

    if callback_context is not None:
        callback_context.state["scrape_tool_output"] = raw_output
        callback_context.state["scrape_query"] = query

    grounded_report = _build_grounded_scrape_report(raw_output, query)
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=grounded_report)],
        )
    )


# 4. Define the Agent
scraping_agent = LlmAgent(
    model=llm_model,
    name="scraping_agent_shard",
    instruction="""
    You are the Scraping Agent.
    ALWAYS call 'analyze_market_trends' exactly once to find live data.


    Rules:
    1) Make 1-3 targeted queries.
    2) After the tool returns, produce ONE final structured Markdown report.
    3) Do not output planning text like "I will call...", partial tool-call text, or phrases like "pending".
    4) Never output placeholder text such as "[insert ...]".
    5) ONLY use information from the tool output. Do NOT infer from any video content or any other agent output.
    6) Do not claim "AI"/"machine learning" unless the tool output explicitly supports it.
    7) If a field is missing, write "Not available from tool output." instead of placeholders.

    Output format (Markdown):
    - Trend Summary (3-6 bullets)
    - Key Evidence (2-5 bullets with concrete facts)
    - Sources (bulleted list of URLs strictly copied from the tool output; if none are present, write "Sources: None")
    """,
    tools=[analyze_market_trends],
    before_model_callback=_direct_scrape_before_model,
    after_tool_callback=_capture_scrape_tool_output,
    after_model_callback=_force_grounded_scrape_response,
)

# 5. EXPOSE THE APP (Crucial Step!)
# This variable 'app' is what uvicorn looks for
app = to_a2a(scraping_agent, port=8001)

if __name__ == "__main__":
    # Optional: Allow running with `python scraping_server.py` for testing
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
