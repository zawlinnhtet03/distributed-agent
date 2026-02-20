"""Generate random vectors for local testing."""
from __future__ import annotations

import json
import random

from router.app.qdrant_schema import FRAME_CLIP_DIM, TEXT_EMBED_DIM


def random_vector(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]


def main() -> None:
    vectors = {
        "frame_clip": random_vector(FRAME_CLIP_DIM),
        "text_embed": random_vector(TEXT_EMBED_DIM),
    }
    print(json.dumps(vectors))


if __name__ == "__main__":
    main()
