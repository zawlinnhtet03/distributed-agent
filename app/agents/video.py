"""
Video Agent - Local video analysis.

This agent uses deterministic pre-model routing for compatibility.
"""

from __future__ import annotations

import re

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from app.agents.base_agent import create_agent
from app.tools.video_tools import analyze_uploaded_video, analyze_video_locally, list_videos
from common.adk_text_sanitizer import force_text_only_model_input


VIDEO_INSTRUCTION = """You are the Video Agent, specialized in local video analysis.

Workflow:
1. If user provides a full video path, analyze that path.
2. If user mentions a specific filename like sample.mp4, analyze by filename.
3. If user asks for available videos, list videos.
4. If user says "analyze this video" without a filename, analyze uploaded video automatically.
"""


_VIDEO_EXT = r"(?:mp4|mov|avi|mkv|webm)"


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
    if fallback:
        return fallback
    return ""


def _extract_video_path(text: str) -> str:
    if not text:
        return ""

    tagged_path = re.search(rf"Path:\s*([^\r\n]+\.{_VIDEO_EXT})", text, flags=re.IGNORECASE)
    if tagged_path:
        return tagged_path.group(1).strip().strip("\"'")

    quoted_path = re.search(rf"[\"']([^\"']+\.{_VIDEO_EXT})[\"']", text, flags=re.IGNORECASE)
    if quoted_path:
        return quoted_path.group(1)

    file_uri = re.search(r"(file://[^\s\"'<>]+)", text, flags=re.IGNORECASE)
    if file_uri:
        return file_uri.group(1)

    windows_path = re.search(
        rf"([A-Za-z]:[\\/][^\s\"'<>]+\.{_VIDEO_EXT})",
        text,
        flags=re.IGNORECASE,
    )
    if windows_path:
        return windows_path.group(1)

    posix_path = re.search(
        rf"((?:/|\.?/)[^\s\"'<>]+\.{_VIDEO_EXT})",
        text,
        flags=re.IGNORECASE,
    )
    if posix_path:
        return posix_path.group(1)

    return ""


def _extract_video_filename(text: str) -> str:
    if not text:
        return ""
    match = re.search(rf"([A-Za-z0-9._-]+\.{_VIDEO_EXT})", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _direct_video_before_model(*args, **kwargs):
    """
    Run video analysis deterministically before model call to avoid
    provider-specific tool-calling parse issues.
    """
    callback_context = kwargs.get("callback_context") if kwargs else None
    llm_request = kwargs.get("llm_request") if kwargs else None

    if callback_context is None and len(args) >= 1:
        callback_context = args[0]
    if llm_request is None and len(args) >= 2:
        llm_request = args[1]

    force_text_only_model_input(callback_context=callback_context, llm_request=llm_request)
    user_text = _extract_latest_user_text(llm_request)
    if not user_text and callback_context is not None:
        user_text = str(callback_context.state.get("sanitized_request", "")).strip()
    lowered = user_text.lower()

    path = _extract_video_path(user_text)
    filename = _extract_video_filename(user_text)

    if path:
        tool_output = analyze_video_locally(video_path=path)
    elif filename:
        tool_output = analyze_uploaded_video(filename=filename)
    elif any(token in lowered for token in ("what videos", "list videos", "available videos")):
        tool_output = list_videos()
    else:
        tool_output = analyze_uploaded_video()

    if callback_context is not None:
        callback_context.state["video_tool_output"] = tool_output

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=tool_output)],
        )
    )


video_agent = create_agent(
    name="video",
    instruction=VIDEO_INSTRUCTION,
    description="Analyzes local video files using OpenCV and Ollama",
    tools=[list_videos, analyze_video_locally, analyze_uploaded_video],
    tier="default",
    temperature=0.3,
    before_model_callback=_direct_video_before_model,
)
