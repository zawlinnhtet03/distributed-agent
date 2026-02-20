"""Thin wrapper helpers for the qdrant-client SDK."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models


def get_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> QdrantClient:
    return QdrantClient(url=base_url, api_key=api_key, timeout=timeout_s)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception:
        return False


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vectors_config: Any,
) -> None:
    client.create_collection(collection_name=collection_name, vectors_config=vectors_config)


def create_payload_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
    field_schema: Any,
) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=field_schema,
    )


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: Iterable[models.PointStruct],
) -> None:
    client.upsert(collection_name=collection_name, points=list(points), wait=True)


def search_points(
    client: QdrantClient,
    collection_name: str,
    vector_name: str,
    query_vector: list[float],
    limit: int,
    query_filter: Optional[models.Filter],
    with_payload: bool,
    score_threshold: Optional[float],
) -> list[models.ScoredPoint]:
    # qdrant-client >=1.16 uses query_points(); older versions expose search().
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=(vector_name, query_vector),
            limit=limit,
            with_payload=with_payload,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
        limit=limit,
        with_payload=with_payload,
        query_filter=query_filter,
        score_threshold=score_threshold,
    )
    return list(response.points or [])
