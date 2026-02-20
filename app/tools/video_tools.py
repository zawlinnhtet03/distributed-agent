"""Video analysis tools using OpenCV and an Ollama vision model."""

import os
import glob
from datetime import datetime

cv2 = None
ollama = None

_VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


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


def _vision_model_name() -> str:
    # Supports runtime switching without code changes.
    # Example (PowerShell): $env:VIDEO_VISION_MODEL="gemma3:4b"
    return (
        os.getenv("VIDEO_VISION_MODEL")
        or os.getenv("OLLAMA_VISION_MODEL")
        or "moondream"
    )


def _fallback_vision_model_name() -> str | None:
    value = (os.getenv("VIDEO_VISION_MODEL_FALLBACK") or "").strip()
    return value or None


def _max_frame_side() -> int:
    try:
        return max(256, int(os.getenv("VIDEO_FRAME_MAX_SIDE", "512")))
    except Exception:
        return 512


def _ollama_num_ctx() -> int:
    try:
        return max(128, int(os.getenv("VIDEO_OLLAMA_NUM_CTX", "256")))
    except Exception:
        return 256


def _resize_frame_for_vision(frame):
    h, w = frame.shape[:2]
    max_side = _max_frame_side()
    current_max = max(h, w)
    if current_max <= max_side:
        return frame
    scale = max_side / float(current_max)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _call_vision_model(image_bytes: bytes, model_name: str) -> str:
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": "Describe this image.", "images": [image_bytes]}],
        options={"num_ctx": _ollama_num_ctx(), "temperature": 0.1},
    )
    return response["message"]["content"]


def _uploads_dir() -> str:
    return os.path.join(_get_project_root(), "shared_data", "videos", "uploads")


def _find_videos() -> list[dict]:
    project_root = _get_project_root()
    cwd = os.getcwd()
    all_videos = []
    seen = set()

    candidate_dirs = {
        cwd,
        project_root,
        os.path.join(project_root, "shared_data", "videos"),
        os.path.join(project_root, "shared_data", "videos", "uploads"),
    }

    for directory in sorted(candidate_dirs):
        if not os.path.exists(directory):
            continue
        for ext in _VIDEO_EXTS:
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
                    "mtime": float(stat.st_mtime),
                })
    all_videos.sort(key=lambda v: v["filename"].lower())
    return all_videos


def _find_uploaded_videos() -> list[dict]:
    uploads = _uploads_dir()
    if not os.path.exists(uploads):
        return []

    found: list[dict] = []
    for ext in _VIDEO_EXTS:
        for vp in glob.glob(os.path.join(uploads, f"*{ext}")):
            abs_path = os.path.abspath(vp)
            stat = os.stat(abs_path)
            found.append({
                "path": _posix(abs_path),
                "filename": os.path.basename(abs_path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "mtime": float(stat.st_mtime),
            })

    found.sort(key=lambda v: v["mtime"], reverse=True)
    return found


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
    """Analyze video using OpenCV and Ollama (3 frames at 20%, 50%, 80%)."""
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

        vision_model = _vision_model_name()

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

            frame = _resize_frame_for_vision(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            image_bytes = buffer.tobytes()
            timestamp = frame_idx / fps

            try:
                desc = _call_vision_model(image_bytes=image_bytes, model_name=vision_model)
            except Exception as e:
                err_text = str(e)
                fallback = _fallback_vision_model_name()
                if (
                    fallback
                    and fallback != vision_model
                    and "requires more system memory" in err_text.lower()
                ):
                    try:
                        desc = _call_vision_model(image_bytes=image_bytes, model_name=fallback)
                        desc = f"[fallback {fallback}] {desc}"
                    except Exception as e2:
                        desc = f"(model={vision_model} error: {e}; fallback={fallback} error: {e2})"
                else:
                    desc = f"(model={vision_model} error: {e})"

            descriptions.append(f"Frame {i+1} @ {timestamp:.1f}s: {desc}")

        video.release()

        if not descriptions:
            return "Error: Could not analyze any frames."

        return (
            f"**Video: {os.path.basename(video_path)}**\n"
            f"Model: {_vision_model_name()}\n"
            f"Duration: {duration:.1f}s | {width}x{height}\n\n"
            + "\n\n".join(descriptions)
        )

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

    # Default behavior for UI uploads: analyze the most recent uploaded video.
    uploaded = _find_uploaded_videos()
    if uploaded:
        chosen = uploaded[0]
        analyzed = analyze_video_locally(chosen["path"])
        return f"Auto-selected latest uploaded video: {chosen['filename']}\n\n{analyzed}"

    # Fallback when no upload folder videos exist: analyze most recently modified overall.
    chosen = max(videos, key=lambda v: v.get("mtime", 0.0))
    analyzed = analyze_video_locally(chosen["path"])
    return f"Auto-selected latest available video: {chosen['filename']}\n\n{analyzed}"
