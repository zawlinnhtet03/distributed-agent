"""Video analysis tools using OpenCV and Ollama Moondream."""

import os
import glob
from datetime import datetime

cv2 = None
ollama = None


def _ensure_deps():
    global cv2, ollama
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    if ollama is None:
        import ollama as _ollama
        ollama = _ollama


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_videos() -> list[dict]:
    project_root = _get_project_root()
    cwd = os.getcwd()
    all_videos = []
    seen = set()

    for directory in list(set([cwd, project_root])):
        if not os.path.exists(directory):
            continue
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            for vp in glob.glob(os.path.join(directory, f"*{ext}")):
                abs_path = os.path.abspath(vp)
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                stat = os.stat(abs_path)
                all_videos.append({
                    "path": _posix(abs_path),
                    "filename": os.path.basename(abs_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                })
    all_videos.sort(key=lambda v: v["filename"].lower())
    return all_videos


def list_videos() -> str:
    """List available video files."""
    videos = _find_videos()
    if not videos:
        return "No video files found."
    lines = [f"Found {len(videos)} video(s):"]
    for i, v in enumerate(videos, 1):
        lines.append(f"  {i}. {v['filename']} ({v['size_mb']} MB)")
    return "\n".join(lines)


def analyze_video_locally(video_path: str) -> str:
    """Analyze video using OpenCV and Moondream (3 frames at 20%, 50%, 80%)."""
    try:
        _ensure_deps()
    except ImportError as e:
        return f"Error: Missing dependency: {e}"

    video_path = video_path.replace("/", os.sep).replace("\\", os.sep)
    if not os.path.isabs(video_path):
        for base in [_get_project_root(), os.getcwd()]:
            candidate = os.path.join(base, video_path)
            if os.path.exists(candidate):
                video_path = candidate
                break

    if not os.path.exists(video_path):
        videos = _find_videos()
        available = ", ".join(v["filename"] for v in videos) if videos else "none"
        return f"Video not found. Available: {available}"

    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            return "Error: Could not open video file."

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or total_frames <= 0:
            video.release()
            return "Error: Invalid video metadata."

        duration = total_frames / fps
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Extract frames at 20%, 50%, 80%
        positions = [0.2, 0.5, 0.8]
        descriptions = []

        for i, pos in enumerate(positions):
            frame_idx = int(total_frames * pos)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            image_bytes = buffer.tobytes()
            timestamp = frame_idx / fps

            try:
                response = ollama.chat(
                    model="moondream",
                    messages=[{"role": "user", "content": "Describe this image.", "images": [image_bytes]}],
                )
                desc = response["message"]["content"]
            except Exception as e:
                desc = f"(error: {e})"

            descriptions.append(f"Frame {i+1} @ {timestamp:.1f}s: {desc}")

        video.release()

        if not descriptions:
            return "Error: Could not analyze any frames."

        return f"**Video: {os.path.basename(video_path)}**\nDuration: {duration:.1f}s | {width}x{height}\n\n" + "\n\n".join(descriptions)

    except Exception as e:
        return f"Error: {e}"


def analyze_uploaded_video(filename: str = "") -> str:
    """Auto-detect and analyze video by filename, or list available."""
    videos = _find_videos()
    if not videos:
        return "No video files found."

    if filename:
        match = next((v for v in videos if v["filename"].lower() == filename.lower()), None)
        if not match:
            match = next((v for v in videos if filename.lower() in v["filename"].lower()), None)
        if match:
            return analyze_video_locally(match["path"])
        return f"Video '{filename}' not found. Available: {', '.join(v['filename'] for v in videos)}"

    if len(videos) == 1:
        return analyze_video_locally(videos[0]["path"])

    lines = [f"Multiple videos found. Please specify:"]
    for i, v in enumerate(videos, 1):
        lines.append(f"  {i}. {v['filename']}")
    return "\n".join(lines)
