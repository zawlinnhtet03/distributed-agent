# Sharded Vector DB Router

## System Overview

This project provides a vector storage and retrieval router on top of three Qdrant shards:

- shard1: short vertical videos (orientation=vertical, duration_sec <= 90)
- shard2: everything else (default)
- shard3: image + carousel content

The router exposes:

- `/upsert` for inserting points (routes each point to a shard)
- `/search` with:
  - `mode=targeted` (single shard)
  - `mode=global` (scatter-gather across all healthy shards)

## How To Run

```bash
docker compose up --build
```

Router is available at `http://localhost:8000`.

## API Usage

### Upsert

Request:
```bash
curl -X POST http://localhost:8000/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "global_1",
        "vectors": {
          "frame_clip": [0.1, 0.2, 0.3],
          "text_embed": [0.1, 0.2, 0.3]
        },
        "payload": {
          "global_id": "global_1",
          "video_id": "video_1",
          "platform": "tiktok",
          "content_type": "video",
          "orientation": "vertical",
          "duration_sec": 12,
          "created_at": 1730000000,
          "ingested_at": 1730000100,
          "frame_ts_ms": 1200,
          "source_url": "https://example.com/1",
          "brand_tags": ["nike"],
          "embedding_model_version": "clip-v1"
        }
      }
    ],
    "target_shard": null,
    "batch_size": 64
  }'
```

Response shape:
```json
{
  "status": "ok",
  "shards_written": {
    "shard1": 1,
    "shard2": 0,
    "shard3": 0
  },
  "warnings": []
}
```

### Targeted Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "targeted",
    "vector_name": "frame_clip",
    "query_vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "filters": {
      "platform": ["tiktok"],
      "content_type": ["video"],
      "orientation": ["vertical"],
      "duration_sec": {"lte": 90}
    },
    "target_shard": null,
    "with_payload": true,
    "score_threshold": null
  }'
```

Response shape:
```json
{
  "mode": "targeted",
  "shards_queried": ["shard1"],
  "took_ms": 12,
  "results": [
    {
      "id": "global_1",
      "score": 0.77,
      "payload": {"global_id": "global_1"},
      "shard": "shard1"
    }
  ],
  "warnings": []
}
```

### Global Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "global",
    "vector_name": "frame_clip",
    "query_vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "filters": {
      "platform": ["tiktok", "instagram"]
    },
    "with_payload": true
  }'
```

Response shape:
```json
{
  "mode": "global",
  "shards_queried": ["shard1", "shard2"],
  "took_ms": 18,
  "results": [
    {
      "id": "global_1",
      "score": 0.83,
      "payload": {"global_id": "global_1"},
      "shard": "shard1"
    }
  ],
  "warnings": []
}
```

## Sharding Rules

Routing rules for upsert and targeted search:

- If `content_type` in {"image", "carousel"} => shard3
- Else if `orientation == "vertical"` AND `duration_sec <= 90` => shard1
- Else => shard2

You can override routing with `target_shard` in both `/upsert` and `/search`.

## Supported Filter Fields

The router translates these into Qdrant filters:

- `platform` (keyword list)
- `content_type` (keyword list)
- `orientation` (keyword list)
- `duration_sec` (range)
- `brand_tags` (keyword list)
- `created_at` (range)
- `ingested_at` (range)
- `global_id` (keyword)
- `video_id` (keyword)
- `frame_ts_ms` (range)
- `source_url` (keyword)
- `embedding_model_version` (keyword)

Range fields support: `gte`, `lte`, `gt`, `lt`.

## Failure Behavior

- Router uses timeouts and retries for shard calls.
- For `/upsert`, a shard failure yields a warning but other shards still ingest.
- For `/search` in global mode, shard failures yield warnings and partial results.
- If all shards fail, `/search` returns HTTP 503.

## Ingestion CLI

Generate mock data:
```bash
python tools/gen_mock_data.py --output mock_data.jsonl --count 50
```

Dry-run (routing stats only):
```bash
python tools/ingest.py --input mock_data.jsonl --router http://localhost:8000 --batch-size 64 --dry-run
```

Ingest data:
```bash
python tools/ingest.py --input mock_data.jsonl --router http://localhost:8000 --batch-size 64
```

## Analyst Agent Integration Notes

Recommended flow for an Analyst Agent:

- Use `/upsert` to insert points with payload fields and named vectors.
- Use `/search` with `mode=targeted` for precise shard querying.
- Use `/search` with `mode=global` when shard is unknown or for broad retrieval.

Expected payload fields:
```json
{
  "global_id": "string",
  "video_id": "string",
  "platform": "tiktok|instagram|youtube",
  "content_type": "video|image|carousel",
  "orientation": "vertical|horizontal|square",
  "duration_sec": 12,
  "created_at": 1730000000,
  "ingested_at": 1730000100,
  "frame_ts_ms": 1200,
  "source_url": "https://example.com/1",
  "brand_tags": ["nike"],
  "embedding_model_version": "clip-v1"
}
```
