from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import ray

from orchestrator.config import init_ray
from orchestrator.scheduler import pipeline
from services.scraper_service import build_video_tasks, discover_local_video_paths

BASE_DIR = Path(os.getenv("BASE_DIR", "shared_data"))
VIDEO_DIR = Path(os.getenv("VIDEO_DIR", str(BASE_DIR / "videos")))
CSV_PATH = Path(os.getenv("VIDEO_METADATA_CSV", str(BASE_DIR / "metadata" / "video_metadata.csv")))
FINAL_REPORT_PATH = Path(os.getenv("FINAL_REPORT_PATH", str(BASE_DIR / "outputs" / "final_report.json")))


def _load_videos_from_csv(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []

    video_paths: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        if "video_path" in fieldnames:
            for row in reader:
                raw_path = row.get("video_path", "").strip()
                if raw_path:
                    video_paths.append(str((BASE_DIR / raw_path).resolve()))
        elif "filename" in fieldnames:
            for row in reader:
                filename = row.get("filename", "").strip()
                if filename:
                    video_paths.append(str((VIDEO_DIR / filename).resolve()))

    return video_paths


def discover_video_paths(limit: int | None = None) -> list[str]:
    candidates = _load_videos_from_csv(CSV_PATH)
    if not candidates:
        candidates = discover_local_video_paths(str(VIDEO_DIR))

    existing_paths = [path for path in candidates if Path(path).exists()]
    if limit is not None:
        existing_paths = existing_paths[:limit]
    return existing_paths


def run_pipeline(video_paths: list[str], source: str) -> list[dict]:
    tasks = build_video_tasks(video_paths, source=source)
    futures = [pipeline.remote(task) for task in tasks]
    return ray.get(futures)


def write_final_report(results: list[dict], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the distributed video processing pipeline.")
    parser.add_argument("--video", action="append", default=[], help="Specific video path(s) to process.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N discovered videos.")
    parser.add_argument("--source", default="local_files", help="Source metadata label for generated tasks.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    init_ray()

    if args.video:
        video_paths = [str(Path(path).resolve()) for path in args.video]
    else:
        video_paths = discover_video_paths(limit=args.limit)

    if not video_paths:
        raise RuntimeError(
            f"No videos found. Checked CSV at {CSV_PATH} and directory {VIDEO_DIR}."
        )

    print(f"Found {len(video_paths)} video(s)")
    results = run_pipeline(video_paths, source=args.source)

    for item in results:
        print("=" * 40)
        print("VIDEO ID:", item["video_id"])
        print("VIDEO PATH:", item["video_path"])
        print("EMBEDDING SIZE:", item["embedding_size"])
        print("SUMMARY:", item["summary"])

    write_final_report(results, FINAL_REPORT_PATH)
    print(f"Final report written to: {FINAL_REPORT_PATH}")
    ray.shutdown()


if __name__ == "__main__":
    main()
