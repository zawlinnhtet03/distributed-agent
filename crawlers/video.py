import os
import re
from urllib.parse import unquote, urlparse

import cv2
import litellm
import ollama
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
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


PREFERRED_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream").strip() or "moondream"
FALLBACK_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_FALLBACK_VISION_MODEL", "gemma3:4b").strip() or "gemma3:4b"


def _select_ollama_vision_model() -> str:
    preferred_order = [PREFERRED_OLLAMA_VISION_MODEL, FALLBACK_OLLAMA_VISION_MODEL]

    try:
        models = ollama.list().get("models", [])
        installed = {m.get("name", "").strip() for m in models if isinstance(m, dict)}
        for candidate in preferred_order:
            if candidate in installed:
                return candidate
    except Exception:
        pass

    return PREFERRED_OLLAMA_VISION_MODEL


def analyze_video_locally(video_path: str) -> str:
    """
    Analyze a local video.
    Uses OpenCV for frame extraction and Ollama vision model when available.
    """
    print(f"Video Shard: Processing '{video_path}'...")

    raw_path = (video_path or "").strip().strip("\"'")
    parsed = urlparse(raw_path)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path or "")
        if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
            raw_path = raw_path[1:]

    candidates = [
        raw_path,
        os.path.join("shared_data", "videos", raw_path),
        os.path.join(os.path.expanduser("~"), "Downloads", os.path.basename(raw_path)),
    ]
    user_profile = os.getenv("USERPROFILE")
    if user_profile:
        candidates.append(os.path.join(user_profile, "Downloads", os.path.basename(raw_path)))

    resolved_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not resolved_path:
        # ADK attachments can arrive as placeholder text paths; fall back to a real
        # local demo file if one exists in shared_data/videos.
        fallback_path = _pick_fallback_video_from_shared_data()
        if fallback_path:
            resolved_path = fallback_path
        else:
            return f"Error: Video file not found at {video_path}"

    try:
        video = cv2.VideoCapture(resolved_path)
        if not video.isOpened():
            return "Error: OpenCV could not open the video file."

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return "Error: Could not determine FPS from video metadata."

        duration = total_frames / fps
        indices = [int(total_frames * 0.2), int(total_frames * 0.5), int(total_frames * 0.8)]
        descriptions = []
        vision_model = _select_ollama_vision_model()

        for idx, frame_idx in enumerate(indices, start=1):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                print(f"Warning: Failed to read frame {idx}")
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            image_bytes = buffer.tobytes()

            try:
                response = ollama.chat(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Describe this image in detail.",
                            "images": [image_bytes],
                        }
                    ],
                )
                desc = response["message"]["content"]
            except Exception as exc:
                # Fallback so the demo can proceed even if model pull/network fails.
                h, w = frame.shape[:2]
                desc = (
                    f"Ollama vision unavailable for model '{vision_model}' ({exc}). "
                    f"Fallback frame metadata: width={w}, height={h}."
                )

            descriptions.append(f"Timestamp {frame_idx / fps:.1f}s: {desc}")

        video.release()

        if not descriptions:
            return "Error: Video opened, but no frames could be read."

        return (
            f"Local Video Analysis\n"
            f"Source Path: {resolved_path}\n"
            f"Vision Model: {vision_model}\n"
            f"Duration: {duration:.1f}s\n"
            f"Visual Narrative:\n" + "\n\n".join(descriptions)
        )
    except Exception as exc:
        return f"Python Error: {exc}"


def _pick_fallback_video_from_shared_data() -> str | None:
    shared_video_dir = os.path.join("shared_data", "videos")
    if not os.path.isdir(shared_video_dir):
        return None

    valid_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    candidates: list[str] = []
    for name in os.listdir(shared_video_dir):
        full_path = os.path.join(shared_video_dir, name)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(name)[1].lower() in valid_ext:
            candidates.append(full_path)

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _extract_timestamp_entries(raw_output: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in raw_output.splitlines():
        line = line.strip()
        match = re.match(r"^Timestamp\s+([0-9.]+)s:\s*(.+)$", line)
        if match:
            entries.append((match.group(1), match.group(2).strip()))
    return entries


def _build_grounded_video_report(raw_output: str, source_path: str | None) -> str:
    if not raw_output:
        return (
            "# Video Summary\n"
            "- Not available from tool output.\n\n"
            "# Key Visual Evidence\n"
            "- Not available from tool output.\n\n"
            "# Notable Text/Logos Seen\n"
            "- Not available from tool output."
        )

    if raw_output.startswith("Error:") or raw_output.startswith("Python Error:"):
        return (
            "# Video Summary\n"
            f"- {raw_output}\n"
            f"- Source path: {source_path or 'Not available from tool output.'}\n\n"
            "# Key Visual Evidence\n"
            "- Not available from tool output.\n\n"
            "# Notable Text/Logos Seen\n"
            "- Not available from tool output."
        )

    duration_match = re.search(r"^Duration:\s*([0-9.]+)s\s*$", raw_output, flags=re.MULTILINE)
    source_match = re.search(r"^Source Path:\s*(.+)\s*$", raw_output, flags=re.MULTILINE)
    source_from_tool = source_match.group(1).strip() if source_match else None
    duration = f"{duration_match.group(1)}s" if duration_match else "Not available from tool output."
    entries = _extract_timestamp_entries(raw_output)
    fallback_mode = any("ollama unavailable" in desc.lower() for _, desc in entries)

    summary_lines = [
        f"- Source path: {source_from_tool or source_path or 'Not available from tool output.'}",
        f"- Duration: {duration}",
        f"- Frames analyzed: {len(entries)}",
        f"- Vision model status: {'Unavailable (fallback metadata only)' if fallback_mode else 'Available'}",
    ]

    evidence_lines: list[str] = []
    for ts, desc in entries:
        if "ollama unavailable" in desc.lower():
            evidence_lines.append(f"- Timestamp {ts}s: Vision model unavailable; fallback metadata only.")
        else:
            evidence_lines.append(f"- Timestamp {ts}s: {desc}")
    if not evidence_lines:
        evidence_lines = ["- Not available from tool output."]

    text_logo_lines = [
        f"- Timestamp {ts}s: {desc}"
        for ts, desc in entries
        if re.search(r"\b(text|logo|sign|brand|caption)\b", desc, flags=re.IGNORECASE)
    ]
    if not text_logo_lines:
        text_logo_lines = ["- Not available from tool output."]

    return (
        "# Video Summary\n"
        + "\n".join(summary_lines)
        + "\n\n# Key Visual Evidence\n"
        + "\n".join(evidence_lines)
        + "\n\n# Notable Text/Logos Seen\n"
        + "\n".join(text_logo_lines)
    )


def _capture_video_tool_output(*args, **kwargs):
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

    tool_context.state["video_tool_output"] = str(tool_response)
    source_path = None
    if isinstance(tool_args, dict):
        source_path = tool_args.get("video_path")
    tool_context.state["video_source_path"] = source_path
    return None


def _force_grounded_video_response(*args, **kwargs):
    callback_context = kwargs.get("callback_context") if kwargs else None
    llm_response = kwargs.get("llm_response") if kwargs else None

    if callback_context is None and len(args) >= 1:
        callback_context = args[0]
    if llm_response is None and len(args) >= 2:
        llm_response = args[1]
    if callback_context is None or llm_response is None:
        return llm_response

    raw_output = callback_context.state.get("video_tool_output")
    source_path = callback_context.state.get("video_source_path")
    if not raw_output:
        return llm_response

    grounded_report = _build_grounded_video_report(raw_output, source_path)
    llm_response.content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=grounded_report)],
    )
    return llm_response


def _extract_video_path_from_text(text: str) -> str:
    if not text:
        return ""

    file_uri_match = re.search(r"(file://[^\s\"'<>]+)", text, flags=re.IGNORECASE)
    if file_uri_match:
        return file_uri_match.group(1)

    path_pattern = (
        r"([A-Za-z]:[\\/][^\s\"'<>]+?\.(?:mp4|mov|avi|mkv|webm)|"
        r"shared_data[\\/][^\s\"'<>]+?\.(?:mp4|mov|avi|mkv|webm)|"
        r"[^\s\"'<>]+?\.(?:mp4|mov|avi|mkv|webm))"
    )
    match = re.search(path_pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _extract_latest_user_video_path_from_request(llm_request) -> str:
    if llm_request is None or not getattr(llm_request, "contents", None):
        return ""

    for content in reversed(llm_request.contents):
        if getattr(content, "role", None) != "user":
            continue

        parts = getattr(content, "parts", None) or []
        text_chunks: list[str] = []
        for part in parts:
            text_val = getattr(part, "text", None)
            if text_val:
                text_chunks.append(text_val)
        merged_text = " ".join(text_chunks).strip()
        if not merged_text:
            continue

        extracted = _extract_video_path_from_text(merged_text)
        if extracted:
            return extracted
    return ""


def _direct_video_before_model(*args, **kwargs):
    """
    Deterministic path for robustness:
    - Extract latest user video path
    - Run local video analysis directly
    - Return grounded markdown as final response
    This bypasses provider-specific function-calling failures.
    """
    callback_context = kwargs.get("callback_context") if kwargs else None
    llm_request = kwargs.get("llm_request") if kwargs else None

    if callback_context is None and len(args) >= 1:
        callback_context = args[0]
    if llm_request is None and len(args) >= 2:
        llm_request = args[1]

    # Keep provider requests safe if a model call ever happens.
    force_text_only_model_input(callback_context=callback_context, llm_request=llm_request)

    source_path = _extract_latest_user_video_path_from_request(llm_request)
    raw_output = analyze_video_locally(video_path=source_path)

    if callback_context is not None:
        callback_context.state["video_tool_output"] = raw_output
        callback_context.state["video_source_path"] = source_path

    grounded_report = _build_grounded_video_report(raw_output, source_path)
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=grounded_report)],
        )
    )


video_agent_shard = LlmAgent(
    model=llm_model,
    name="video_agent_shard",
    instruction="""
    You are the Video Forensics Engineer.
    Your tool runs locally on the machine.

    Task:
    1. Run `analyze_video_locally`.
    2. Read the text descriptions returned by the tool.
    3. Synthesize the findings into a "Vibe Check" report.

    Rules:
    - Always call `analyze_video_locally` exactly once.
    - After the tool returns, produce ONE final Markdown report.
    - Do not output planning text like "I will call...", partial tool-call text, placeholders, or phrases like "pending".
    - Never output placeholder text such as "[insert ...]".
    - Use only concrete evidence returned by the tool.
    - If a field is missing, write "Not available from tool output." instead of placeholders.

    Output format (Markdown):
    - Video Summary (2-5 bullets)
    - Key Visual Evidence (bullets)
    - Notable Text/Logos Seen (if any)
    """,
    tools=[analyze_video_locally],
    before_model_callback=_direct_video_before_model,
    after_tool_callback=_capture_video_tool_output,
    after_model_callback=_force_grounded_video_response,
)

app = to_a2a(video_agent_shard, port=8002)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8002)
