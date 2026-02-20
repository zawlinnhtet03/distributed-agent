"""Qdrant collection and payload schema definitions.

created_at and ingested_at are stored as integer epoch milliseconds.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .qdrant_client import collection_exists, create_collection, create_payload_index


COLLECTION_NAME = "multimodal_items"

FRAME_CLIP_DIM = int(os.getenv("FRAME_CLIP_DIM", "512"))
TEXT_EMBED_DIM = int(os.getenv("TEXT_EMBED_DIM", "768"))


def _vectors_config() -> Dict[str, models.VectorParams]:
    return {
        "frame_clip": models.VectorParams(size=FRAME_CLIP_DIM, distance=models.Distance.COSINE),
        "text_embed": models.VectorParams(size=TEXT_EMBED_DIM, distance=models.Distance.COSINE),
    }


def _payload_indexes() -> Iterable[Tuple[str, models.PayloadSchemaType]]:
    return [
        ("global_id", models.PayloadSchemaType.KEYWORD),
        ("video_id", models.PayloadSchemaType.KEYWORD),
        ("platform", models.PayloadSchemaType.KEYWORD),
        ("content_type", models.PayloadSchemaType.KEYWORD),
        ("orientation", models.PayloadSchemaType.KEYWORD),
        ("duration_sec", models.PayloadSchemaType.INTEGER),
        ("created_at", models.PayloadSchemaType.INTEGER),
        ("ingested_at", models.PayloadSchemaType.INTEGER),
        ("frame_ts_ms", models.PayloadSchemaType.INTEGER),
        ("source_url", models.PayloadSchemaType.KEYWORD),
        ("brand_tags", models.PayloadSchemaType.KEYWORD),
        ("embedding_model_version", models.PayloadSchemaType.KEYWORD),
    ]


def ensure_schema(client: QdrantClient) -> None:
    if not collection_exists(client, COLLECTION_NAME):
        create_collection(client, COLLECTION_NAME, vectors_config=_vectors_config())


def ensure_payload_indexes(client: QdrantClient) -> None:
    if not collection_exists(client, COLLECTION_NAME):
        create_collection(client, COLLECTION_NAME, vectors_config=_vectors_config())

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    payload_schema = collection_info.payload_schema or {}
    existing_fields = set(payload_schema.keys())

    for field_name, field_schema in _payload_indexes():
        if field_name in existing_fields:
            continue
        create_payload_index(client, COLLECTION_NAME, field_name, field_schema)
