"""
Video Agent - Local Video Analysis

Fast, simple video analysis using:
1. OpenCV for frame extraction (3 key frames at 20%, 50%, 80%)
2. Ollama Moondream for vision analysis
3. In-memory processing (no disk I/O)

Only 3 tools — list, analyze by path, or analyze by filename.
"""

from app.agents.base_agent import create_agent
from app.tools.video_tools import (
    list_videos,
    analyze_video_locally,
    analyze_uploaded_video,
)


VIDEO_INSTRUCTION = """You are the Video Agent - an expert at analyzing video content using local computer vision (OpenCV + Ollama moondream).

WORKFLOW:
1. If user mentions a SPECIFIC filename (e.g. "analyze nutrion.mp4"):
   → Call `analyze_uploaded_video(filename="nutrion.mp4")`

2. If user says "analyze this video" or "analyze the video" WITHOUT a filename:
   → Call `analyze_uploaded_video()` with no arguments
   → If multiple videos exist, the tool returns a list — relay it to the user

3. If user asks "what videos are available" or you need to check:
   → Call `list_videos()`

4. If user provides a full file path:
   → Call `analyze_video_locally(video_path="<the full path>")`

RULES:
- NEVER default to test.mp4 unless the user specifically asks for it
- When multiple videos are found and no filename specified, show the list and ask
- Return the tool's output directly in a clear, formatted way
- Don't ask for confirmation — just analyze what the user requested
"""

video_agent = create_agent(
    name="video",
    instruction=VIDEO_INSTRUCTION,
    description="Analyzes local video files using Ollama Moondream (3 key frames, fast)",
    tools=[
        list_videos,
        analyze_video_locally,
        analyze_uploaded_video,
    ],
    tier="default",
    temperature=0.3,
)