from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import ray

from orchestrator.config import init_ray
from orchestrator.scheduler import pipeline
from services.scraper_service import build_video_tasks, discover_local_video_paths
from services.vector_service import text_to_embedding

BASE_DIR = Path(os.getenv("BASE_DIR", "shared_data"))
VIDEO_DIR = Path(os.getenv("VIDEO_DIR", str(BASE_DIR / "videos")))
IMAGE_METADATA_CSV = Path(
    os.getenv("IMAGE_METADATA_CSV", str(BASE_DIR / "metadata" / "image_metadata_all.csv"))
)
FINAL_PROJECT_REPORT_PATH = Path(
    os.getenv("FINAL_PROJECT_REPORT_PATH", str(BASE_DIR / "outputs" / "final_project_report.json"))
)
DEFAULT_ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8000")
FRAME_CLIP_DIM = int(os.getenv("FRAME_CLIP_DIM", "512"))
TEXT_EMBED_DIM = int(os.getenv("TEXT_EMBED_DIM", "768"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sharded multi-agent retrieval final project flow."
    )
    parser.add_argument("--router-url", default=DEFAULT_ROUTER_URL)
    parser.add_argument("--limit-videos", type=int, default=3)
    parser.add_argument("--limit-images", type=int, default=3)
    parser.add_argument("--source", default="final_project")
    return parser.parse_args()


def _http_get_json(url: str, timeout_s: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: int = 30) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        content = response.read().decode("utf-8")
    return json.loads(content)


def wait_for_router(router_url: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            health = _http_get_json(f"{router_url}/health", timeout_s=5)
            if health.get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover - depends on runtime service startup
            last_error = exc
        time.sleep(2)

    raise RuntimeError(f"Router was not ready at {router_url}") from last_error


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _image_orientation(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "square"
    if width == height:
        return "square"
    return "horizontal" if width > height else "vertical"


def load_image_candidates(limit: int) -> list[dict[str, Any]]:
    if not IMAGE_METADATA_CSV.exists() or limit <= 0:
        return []

    rows: list[dict[str, Any]] = []
    with IMAGE_METADATA_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            width = _safe_int(row.get("width"), default=0)
            height = _safe_int(row.get("height"), default=0)
            rows.append(
                {
                    "image_id": str(row.get("image_id", "")).strip(),
                    "image_url": str(row.get("image_url", "")).strip(),
                    "file_path": str(row.get("file_path", "")).strip(),
                    "source": str(row.get("source", "image_metadata")).strip(),
                    "orientation": _image_orientation(width, height),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def run_video_analyst(video_paths: list[str], source: str) -> list[dict[str, Any]]:
    if not video_paths:
        return []

    tasks = build_video_tasks(video_paths, source=source)
    for task in tasks:
        task.metadata["actor"] = "analyst_video"

    futures = [pipeline.remote(task) for task in tasks]
    return ray.get(futures)


def _build_video_point(item: dict[str, Any], index: int, now_ts: int) -> dict[str, Any]:
    video_id = str(item["video_id"])
    transcript = str(item.get("transcript", ""))
    summary = str(item.get("summary", ""))
    # Alternating routing-friendly metadata ensures shard1 and shard2 are both exercised.
    orientation = "vertical" if index % 2 == 1 else "horizontal"
    duration_sec = 45 if orientation == "vertical" else 180

    payload = {
        "global_id": f"video_{video_id}",
        "video_id": video_id,
        "platform": "local",
        "content_type": "video",
        "orientation": orientation,
        "duration_sec": duration_sec,
        "created_at": now_ts - (index * 60),
        "ingested_at": now_ts,
        "frame_ts_ms": 1000 * index,
        "source_url": f"file://{item.get('video_path', '')}",
        "brand_tags": ["demo", "local"],
        "embedding_model_version": "hash-v1",
    }
    vectors = {
        "frame_clip": text_to_embedding(summary or transcript or video_id, dim=FRAME_CLIP_DIM),
        "text_embed": text_to_embedding(transcript or summary or video_id, dim=TEXT_EMBED_DIM),
    }
    return {
        "id": _stable_point_uuid("video", video_id),
        "vectors": vectors,
        "payload": payload,
    }


def _build_image_point(row: dict[str, Any], index: int, now_ts: int) -> dict[str, Any]:
    image_id = row["image_id"] or f"img_{index}"
    text_seed = f"{row['image_url']} {row['file_path']} {row['source']}"
    payload = {
        "global_id": f"image_{image_id}",
        "video_id": f"image_{image_id}",
        "platform": "metadata",
        "content_type": "image",
        "orientation": row["orientation"],
        "duration_sec": 0,
        "created_at": now_ts - (index * 30),
        "ingested_at": now_ts,
        "frame_ts_ms": 0,
        "source_url": row["image_url"] or row["file_path"] or "about:blank",
        "brand_tags": ["image"],
        "embedding_model_version": "hash-v1",
    }
    vectors = {
        "frame_clip": text_to_embedding(text_seed, dim=FRAME_CLIP_DIM),
        "text_embed": text_to_embedding(text_seed, dim=TEXT_EMBED_DIM),
    }
    return {
        "id": _stable_point_uuid("image", image_id),
        "vectors": vectors,
        "payload": payload,
    }


def _stable_point_uuid(prefix: str, key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}:{key}"))


def build_router_points(
    video_analysis: list[dict[str, Any]],
    image_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    now_ts = int(time.time())
    points: list[dict[str, Any]] = []

    for index, item in enumerate(video_analysis, start=1):
        points.append(_build_video_point(item, index=index, now_ts=now_ts))

    base = len(points)
    for offset, row in enumerate(image_candidates, start=1):
        points.append(_build_image_point(row, index=base + offset, now_ts=now_ts))

    return points


def run_retrieval(router_url: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"targeted": {}, "global": {}, "warnings": ["No points available for retrieval."]}

    query_vector = points[0]["vectors"]["frame_clip"]
    targeted_payload = {
        "mode": "targeted",
        "vector_name": "frame_clip",
        "query_vector": query_vector,
        "top_k": 5,
        "filters": {
            "content_type": ["video"],
            "orientation": ["vertical"],
            "duration_sec": {"lte": 90},
        },
        "with_payload": True,
    }
    global_payload = {
        "mode": "global",
        "vector_name": "frame_clip",
        "query_vector": query_vector,
        "top_k": 10,
        "filters": {"platform": ["local", "metadata"]},
        "with_payload": True,
    }

    targeted_response = _http_post_json(f"{router_url}/search", targeted_payload, timeout_s=45)
    global_response = _http_post_json(f"{router_url}/search", global_payload, timeout_s=45)

    return {
        "targeted": targeted_response,
        "global": global_response,
        "warnings": [],
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = _parse_args()
    router_url = args.router_url.rstrip("/")

    video_paths = discover_local_video_paths(str(VIDEO_DIR))
    if args.limit_videos > 0:
        video_paths = video_paths[: args.limit_videos]

    image_candidates = load_image_candidates(limit=args.limit_images)

    report: dict[str, Any] = {
        "project": "Sharded Multi-Agent Retrieval System",
        "generated_at_epoch": int(time.time()),
        "stages": {},
    }

    report["stages"]["scout_actors"] = {
        "video_scout_count": len(video_paths),
        "image_scout_count": len(image_candidates),
        "video_paths": video_paths,
    }

    init_ray()
    try:
        video_analysis = run_video_analyst(video_paths=video_paths, source=args.source)
    finally:
        if ray.is_initialized():
            ray.shutdown()

    report["stages"]["analyst_actors"] = {
        "processed_videos": len(video_analysis),
        "results": video_analysis,
    }

    points = build_router_points(video_analysis, image_candidates)
    report["stages"]["router_ingest"] = {
        "points_prepared": len(points),
    }

    if not points:
        report["stages"]["router_ingest"]["upsert_response"] = {
            "status": "skipped",
            "reason": "No points prepared from scout/analyst stages.",
        }
        report["stages"]["retrieval"] = {
            "status": "skipped",
            "reason": "No points available for retrieval.",
        }
        write_report(report, FINAL_PROJECT_REPORT_PATH)
        print(f"Final project report written to: {FINAL_PROJECT_REPORT_PATH}")
        return

    wait_for_router(router_url)
    upsert_response = _http_post_json(
        f"{router_url}/upsert",
        {"points": points, "batch_size": 64},
        timeout_s=60,
    )
    report["stages"]["router_ingest"]["upsert_response"] = upsert_response
    shards_written = upsert_response.get("shards_written", {})
    total_written = sum(int(v) for v in shards_written.values()) if isinstance(shards_written, dict) else 0
    if total_written == 0:
        warnings = upsert_response.get("warnings", [])
        raise RuntimeError(f"Router upsert wrote 0 points. warnings={warnings}")

    retrieval = run_retrieval(router_url, points)
    report["stages"]["retrieval"] = retrieval

    write_report(report, FINAL_PROJECT_REPORT_PATH)
    print(f"Final project report written to: {FINAL_PROJECT_REPORT_PATH}")


if __name__ == "__main__":
    main()
