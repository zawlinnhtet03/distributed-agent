from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from common.models import ProcessingResult

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "32"))
OUTPUT_PATH = Path(os.getenv("EMBEDDINGS_OUTPUT_PATH", "shared_data/outputs/embeddings.jsonl"))


def text_to_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    values: list[float] = []
    seed = text.encode("utf-8")
    counter = 0

    # Deterministic hash embedding for local pipeline wiring.
    while len(values) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(4, byteorder="big")).digest()
        for idx in range(0, len(digest), 4):
            chunk = digest[idx : idx + 4]
            if len(chunk) < 4:
                break
            number = int.from_bytes(chunk, byteorder="big", signed=False)
            values.append((number / 4294967295.0) * 2.0 - 1.0)
            if len(values) >= dim:
                break
        counter += 1

    return values


def _append_record(record: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def store_embeddings(result: ProcessingResult, metadata: dict | None = None) -> ProcessingResult:
    if not result.embeddings:
        result.embeddings = text_to_embedding(result.transcript)

    record = {
        "video_id": result.video_id,
        "embedding_dim": len(result.embeddings),
        "transcript_chars": len(result.transcript),
        "metadata": metadata or {},
    }
    _append_record(record)
    print(f"[VectorDB] Stored embeddings for video {result.video_id} at {OUTPUT_PATH}")
    return result
