import os
from pathlib import Path
import uuid
from urllib.parse import unquote, urlparse

from a2a.types import TextPart
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.a2a.converters.event_converter import convert_genai_part_to_a2a_part
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm

use_gemini_aggregator = bool(os.getenv("GOOGLE_API_KEY"))

if use_gemini_aggregator:
    aggregator_model = Gemini(model="gemini-2.5-flash-lite")
else:
    aggregator_model = LiteLlm(
        model="groq/llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
    )


UPLOAD_DIR = Path(os.getenv("ADK_VIDEO_UPLOAD_DIR", "shared_data/videos/uploads"))


aggregator_agent = LlmAgent(
    model=aggregator_model,
    name="AggregatorAgent",
    instruction=r"""
    You are the Lead Market Analyst.

    You will receive outputs from two upstream agents:
    - scraping_agent_shard: web trend report
    - video_agent_shard: visual/video analysis

    Hard rules:
    - Do NOT invent sources.
    - If the scraping report has \"Sources: None\" or indicates an error/limited data, you MUST lower confidence and you MAY NOT make strong claims.
    - Do not claim \"AI\"/\"machine learning\" unless the scraping report explicitly supports it.

    Task:
    1) Summarize each input separately.
    2) Correlate: do visuals support the web trend report?
    3) Produce a verdict tag from EXACTLY one of:
       - Real & Observable
       - Hype vs Reality Mismatch
       - Insufficient Evidence
    4) Provide Confidence: High / Medium / Low.

    Output format (Markdown):
    - Scraping Agent Summary
    - Video Agent Summary
    - Correlation (include 2-5 bullets of evidence; only cite evidence present in inputs)
    - Verdict Tag
    - Confidence
    - Executive Summary (120-180 words)
    - Final Verdict (one line)
    """,
)


def _mime_to_ext(mime: str) -> str:
    mime = (mime or "").lower().strip()
    mapping = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
    }
    return mapping.get(mime, ".bin")


def _file_uri_to_local_path(uri: str) -> str | None:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    local_path = unquote(parsed.path or "")
    # Windows drive path normalization: /C:/foo -> C:/foo
    if local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
        local_path = local_path[1:]
    return local_path or None


def _save_inline_video(inline_data) -> str | None:
    data = getattr(inline_data, "data", None)
    mime = getattr(inline_data, "mime_type", None) or "application/octet-stream"
    if not data:
        return None

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        out_path = UPLOAD_DIR / f"adk_upload_{uuid.uuid4().hex}{_mime_to_ext(mime)}"
        out_path.write_bytes(data)
        return str(out_path)
    except Exception:
        return None


def _compact_genai_part_for_scraper_a2a(part):
    """
    Keep A2A requests small:
    - Replace binary/attachment parts with short text markers.
    - Truncate oversized text parts.
    """
    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        uri = getattr(file_data, "file_uri", None) or getattr(file_data, "uri", None) or "unknown"
        mime = getattr(file_data, "mime_type", None) or "unknown"
        return TextPart(text=f"[File attachment: {uri} ({mime})]")

    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        mime = getattr(inline_data, "mime_type", None) or "unknown"
        return TextPart(text=f"[Inline attachment omitted ({mime})]")

    converted = convert_genai_part_to_a2a_part(part)
    if isinstance(converted, TextPart) and len(converted.text) > 6000:
        return TextPart(text=converted.text[:6000] + "\n...[truncated for transport size]")
    if converted is None:
        return TextPart(text="[Unsupported content omitted]")
    return converted


def _compact_genai_part_for_video_a2a(part):
    """
    Preserve video attachments for the video shard:
    - file_data(file://...): pass resolved local path when possible.
    - inline_data(video/*): persist to shared_data/videos/uploads and pass path.
    """
    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        uri = getattr(file_data, "file_uri", None) or getattr(file_data, "uri", None) or ""
        local_path = _file_uri_to_local_path(uri)
        if local_path:
            return TextPart(text=local_path)
        mime = getattr(file_data, "mime_type", None) or "unknown"
        return TextPart(text=f"[File attachment: {uri or 'unknown'} ({mime})]")

    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        saved_path = _save_inline_video(inline_data)
        if saved_path:
            return TextPart(text=saved_path)
        mime = getattr(inline_data, "mime_type", None) or "unknown"
        return TextPart(text=f"[Inline attachment omitted ({mime})]")

    converted = convert_genai_part_to_a2a_part(part)
    if isinstance(converted, TextPart) and len(converted.text) > 6000:
        return TextPart(text=converted.text[:6000] + "\n...[truncated for transport size]")
    if converted is None:
        return TextPart(text="[Unsupported content omitted]")
    return converted


remote_scraping_agent = RemoteA2aAgent(
    name="scraping_agent_shard",
    agent_card="http://localhost:8001/.well-known/agent-card.json",
    genai_part_converter=_compact_genai_part_for_scraper_a2a,
    full_history_when_stateless=False,
)

remote_video_agent = RemoteA2aAgent(
    name="video_agent_shard",
    agent_card="http://localhost:8002/.well-known/agent-card.json",
    genai_part_converter=_compact_genai_part_for_video_a2a,
    full_history_when_stateless=False,
)


gathering_squad = ParallelAgent(
    name="GatheringLayer",
    sub_agents=[remote_scraping_agent, remote_video_agent],
)


if use_gemini_aggregator:
    root_agent = SequentialAgent(
        name="ResearchPipeline",
        sub_agents=[gathering_squad, aggregator_agent],
    )
else:
    # Groq OpenAI-compatible endpoints can reject ADK's structured message parts
    # in the final aggregation hop. In that case, return parallel shard outputs directly.
    root_agent = SequentialAgent(
        name="ResearchPipelineNoAggregator",
        sub_agents=[gathering_squad],
    )
