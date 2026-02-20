"""FastAPI entrypoint for the router service."""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from qdrant_client.http import models as qdrant_models

from .config import settings
from .aggregation import merge_topk
from .filters import build_filter
from .models import SearchHit, SearchRequest, SearchResponse, UpsertPoint, UpsertRequest
from .qdrant_client import get_client, search_points, upsert_points
from .qdrant_schema import (
    COLLECTION_NAME,
    FRAME_CLIP_DIM,
    TEXT_EMBED_DIM,
    ensure_payload_indexes,
    ensure_schema,
)
from .router_policy import choose_shard, choose_shard_from_filters
from .shard_registry import get_registry

app = FastAPI(title="Vector Router", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "router"}


@app.post("/upsert")
def upsert(request: UpsertRequest) -> dict:
    if not request.points:
        raise HTTPException(status_code=422, detail="points must not be empty")

    shard_batches: dict[str, list[UpsertPoint]] = {}
    warnings: list[str] = []

    for point in request.points:
        _validate_point(point)
        shard_name = choose_shard(point.payload, request.target_shard)
        shard_batches.setdefault(shard_name, []).append(point)

    shards_written: dict[str, int] = {name: 0 for name in get_registry().list_shards()}

    for shard_name, points in shard_batches.items():
        shard_url = get_registry().get_url(shard_name)
        client = get_client(
            shard_url,
            api_key=settings.qdrant_api_key,
            timeout_s=settings.request_timeout_s,
        )
        try:
            for batch in _iter_batches(points, request.batch_size):
                qdrant_points = [
                    qdrant_models.PointStruct(
                        id=item.id,
                        vector={
                            "frame_clip": item.vectors.frame_clip,
                            **({"text_embed": item.vectors.text_embed} if item.vectors.text_embed else {}),
                        },
                        payload=item.payload,
                    )
                    for item in batch
                ]
                try:
                    _upsert_with_retry(client, qdrant_points)
                    shards_written[shard_name] += len(batch)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"shard {shard_name} upsert failed: {exc}")
                    break
        finally:
            client.close()

    return {"status": "ok", "shards_written": shards_written, "warnings": warnings}


@app.post("/search")
def search(request: SearchRequest) -> SearchResponse:
    _validate_search_request(request)
    if request.mode == "targeted":
        return _search_targeted(request)
    if request.mode == "global":
        return _search_global(request)
    raise HTTPException(status_code=400, detail="mode must be targeted or global")


@app.get("/shards")
async def shards() -> dict:
    registry = get_registry()
    health = await registry.healthcheck_all()
    shard_list = [
        {"name": name, "url": registry.get_url(name), "healthy": health.get(name, False)}
        for name in registry.list_shards()
    ]
    return {"shards": shard_list}


def _log_config() -> None:
    # Ensures settings are loaded during startup; useful for debugging env.
    _ = settings


@app.on_event("startup")
def on_startup() -> None:
    _log_config()
    registry = get_registry()
    for shard_name in registry.list_shards():
        shard_url = registry.get_url(shard_name)
        client = get_client(
            shard_url,
            api_key=settings.qdrant_api_key,
            timeout_s=settings.request_timeout_s,
        )
        try:
            ensure_schema(client)
            ensure_payload_indexes(client)
        finally:
            client.close()


def _validate_point(point: UpsertPoint) -> None:
    if not point.id:
        raise HTTPException(status_code=422, detail="point.id is required")

    frame_clip = point.vectors.frame_clip
    if not frame_clip or len(frame_clip) != FRAME_CLIP_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"vectors.frame_clip must be length {FRAME_CLIP_DIM}",
        )

    if point.vectors.text_embed is not None and len(point.vectors.text_embed) != TEXT_EMBED_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"vectors.text_embed must be length {TEXT_EMBED_DIM}",
        )

    payload = point.payload
    for field_name in ("global_id", "content_type", "orientation", "duration_sec"):
        if field_name not in payload or payload[field_name] is None:
            raise HTTPException(status_code=422, detail=f"payload.{field_name} is required")


def _validate_search_request(request: SearchRequest) -> None:
    if request.vector_name not in {"frame_clip", "text_embed"}:
        raise HTTPException(status_code=422, detail="vector_name must be frame_clip or text_embed")

    if request.vector_name == "frame_clip" and len(request.query_vector) != FRAME_CLIP_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"query_vector must be length {FRAME_CLIP_DIM} for frame_clip",
        )

    if request.vector_name == "text_embed" and len(request.query_vector) != TEXT_EMBED_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"query_vector must be length {TEXT_EMBED_DIM} for text_embed",
        )


def _search_targeted(request: SearchRequest) -> SearchResponse:
    shard_name = choose_shard_from_filters(request.filters, request.target_shard)
    shard_url = get_registry().get_url(shard_name)
    query_filter = build_filter(request.filters)

    started = time.time()
    warnings: list[str] = []
    client = get_client(
        shard_url,
        api_key=settings.qdrant_api_key,
        timeout_s=settings.request_timeout_s,
    )
    try:
        results = search_points(
            client,
            COLLECTION_NAME,
            request.vector_name,
            request.query_vector,
            request.top_k,
            query_filter,
            request.with_payload,
            request.score_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"shard {shard_name} search failed: {exc}")
        results = []
    finally:
        client.close()

    took_ms = int((time.time() - started) * 1000)
    hits = [
        SearchHit(
            id=item.id,
            score=item.score,
            payload=item.payload if request.with_payload else None,
            shard=shard_name,
        )
        for item in results
    ]
    return SearchResponse(
        mode="targeted",
        shards_queried=[shard_name],
        took_ms=took_ms,
        results=hits,
        warnings=warnings,
    )


def _search_global(request: SearchRequest) -> SearchResponse:
    registry = get_registry()
    warnings: list[str] = []
    started = time.time()

    try:
        healthy_shards = _run_async(registry.get_healthy_shards())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"healthcheck failed: {exc}") from exc

    if not healthy_shards:
        raise HTTPException(status_code=503, detail="no healthy shards available")

    query_filter = build_filter(request.filters)
    shard_results: dict[str, list[SearchHit]] = {}
    top_k_shard = min(request.top_k * 2, 1000)

    def _query_shard(shard_name: str) -> list[SearchHit]:
        shard_url = registry.get_url(shard_name)
        client = get_client(
            shard_url,
            api_key=settings.qdrant_api_key,
            timeout_s=settings.request_timeout_s,
        )
        try:
            results = search_points(
                client,
                COLLECTION_NAME,
                request.vector_name,
                request.query_vector,
                top_k_shard,
                query_filter,
                request.with_payload,
                request.score_threshold,
            )
            return [
                SearchHit(
                    id=item.id,
                    score=item.score,
                    payload=item.payload if request.with_payload else None,
                    shard=shard_name,
                )
                for item in results
            ]
        finally:
            client.close()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=len(healthy_shards)) as executor:
        future_map = {executor.submit(_query_shard, shard): shard for shard in healthy_shards}
        for future in as_completed(future_map):
            shard_name = future_map[future]
            try:
                shard_results[shard_name] = future.result()
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"shard {shard_name} search failed: {exc}")

    if not shard_results:
        raise HTTPException(status_code=503, detail="all shards failed to respond")

    merged = merge_topk(shard_results, request.top_k)
    took_ms = int((time.time() - started) * 1000)
    return SearchResponse(
        mode="global",
        shards_queried=sorted(shard_results.keys()),
        took_ms=took_ms,
        results=merged,
        warnings=warnings,
    )


def _run_async(coro):
    import asyncio

    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _iter_batches(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _upsert_with_retry(client, points):
    last_exc = None
    for attempt in range(settings.retry_count + 1):
        try:
            upsert_points(client, COLLECTION_NAME, points)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= settings.retry_count:
                break
            time.sleep(settings.retry_backoff_s * (attempt + 1))
    raise last_exc
