"""Generate mock JSONL data for ingestion testing."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from router.app.qdrant_schema import FRAME_CLIP_DIM, TEXT_EMBED_DIM


CONTENT_TYPES = ["video", "image", "carousel"]
ORIENTATIONS = ["vertical", "horizontal", "square"]
PLATFORMS = ["tiktok", "instagram", "youtube"]
BRANDS = ["nike", "adidas", "puma", "none"]


def _random_vector(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]


def _random_payload(index: int) -> dict:
    content_type = random.choice(CONTENT_TYPES)
    orientation = random.choice(ORIENTATIONS)
    duration_sec = random.randint(5, 180)
    created_at = int(time.time()) - random.randint(0, 86400)
    ingested_at = created_at + random.randint(0, 600)
    brand = random.choice(BRANDS)
    payload = {
        "global_id": f"global_{index}",
        "video_id": f"video_{index}",
        "platform": random.choice(PLATFORMS),
        "content_type": content_type,
        "orientation": orientation,
        "duration_sec": duration_sec,
        "created_at": created_at,
        "ingested_at": ingested_at,
        "frame_ts_ms": random.randint(0, 30000),
        "source_url": f"https://example.com/{index}",
        "embedding_model_version": "clip-v1",
    }
    if brand != "none":
        payload["brand_tags"] = [brand]
    return payload


def _random_vectors() -> dict:
    vectors = {"frame_clip": _random_vector(FRAME_CLIP_DIM)}
    if random.random() < 0.7:
        vectors["text_embed"] = _random_vector(TEXT_EMBED_DIM)
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock JSONL data")
    parser.add_argument("--output", default="mock_data.jsonl")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in range(args.count):
            record = {
                "id": f"global_{idx}",
                "vectors": _random_vectors(),
                "payload": _random_payload(idx),
            }
            handle.write(json.dumps(record))
            handle.write("\n")

    print(f"wrote {args.count} records to {output_path}")


if __name__ == "__main__":
    main()
