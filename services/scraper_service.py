from __future__ import annotations

from pathlib import Path

from common.models import VideoTask

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def discover_local_video_paths(video_dir: str = "shared_data/videos") -> list[str]:
    root = Path(video_dir)
    if not root.exists():
        return []

    video_paths = [
        str(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
    ]
    return video_paths


def build_video_tasks(video_paths: list[str], source: str = "local") -> list[VideoTask]:
    tasks: list[VideoTask] = []
    for idx, video_path in enumerate(video_paths, start=1):
        path = Path(video_path)
        task = VideoTask(
            video_id=f"{path.stem}_{idx:03d}",
            video_path=str(path),
            metadata={"source": source},
        )
        tasks.append(task)
    return tasks


def scrape_videos(url: str) -> VideoTask:
    paths = discover_local_video_paths()
    if not paths:
        raise FileNotFoundError("No local videos found in shared_data/videos")

    return VideoTask(
        video_id=Path(paths[0]).stem,
        video_path=paths[0],
        metadata={"source": url},
    )
