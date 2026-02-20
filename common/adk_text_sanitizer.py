from __future__ import annotations

from typing import Any

from google.genai import types


def _part_to_text(part: types.Part) -> str:
    if getattr(part, "text", None):
        return part.text

    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        file_uri = getattr(file_data, "file_uri", None) or getattr(file_data, "uri", None)
        mime_type = getattr(file_data, "mime_type", None) or "unknown"
        if file_uri:
            return f"[Attached file: {file_uri} ({mime_type})]"
        return f"[Attached file ({mime_type})]"

    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        mime_type = getattr(inline_data, "mime_type", None) or "unknown"
        return f"[Inline attachment ({mime_type})]"

    function_call = getattr(part, "function_call", None)
    if function_call is not None:
        name = getattr(function_call, "name", "unknown")
        return f"[Function call: {name}]"

    function_response = getattr(part, "function_response", None)
    if function_response is not None:
        name = getattr(function_response, "name", "unknown")
        return f"[Function response: {name}]"

    executable_code = getattr(part, "executable_code", None)
    if executable_code is not None:
        return "[Executable code content]"

    code_execution_result = getattr(part, "code_execution_result", None)
    if code_execution_result is not None:
        return "[Code execution result]"

    return "[Unsupported non-text content]"


def force_text_only_model_input(*args: Any, **kwargs: Any):
    """
    Normalize all incoming message parts to text-only to keep OpenAI-compatible
    providers (like Groq chat completions) from rejecting structured content.
    """
    llm_request = kwargs.get("llm_request")
    if llm_request is None and len(args) >= 2:
        llm_request = args[1]
    if llm_request is None:
        return None

    normalized_contents: list[types.Content] = []
    for content in llm_request.contents:
        normalized_parts: list[types.Part] = []
        for part in content.parts or []:
            normalized_parts.append(types.Part.from_text(text=_part_to_text(part)))

        if not normalized_parts:
            normalized_parts.append(types.Part.from_text(text="[No content]"))

        normalized_contents.append(types.Content(role=content.role, parts=normalized_parts))

    llm_request.contents = normalized_contents
    return None
